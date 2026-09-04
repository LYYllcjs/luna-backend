import hashlib
import json
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from asr_service import recognize_audio
from elex_service import ElexServiceError, analyze_health, pet_chat
from health_analysis_models import HealthAnalysisRequest, HealthAnalysisResponse, request_payload


MAX_AUDIO_BYTES = int(os.getenv("ASR_MAX_AUDIO_BYTES", str(25 * 1024 * 1024)))
MIN_AUDIO_SECONDS = float(os.getenv("ASR_MIN_AUDIO_SECONDS", "0.2"))
EXPECTED_SAMPLE_RATE = 16000
EXPECTED_CHANNELS = 1


def _load_dotenv() -> None:
    """Load key=value pairs from backend .env so ELEX_API_KEY 等配置
    可以保存在文件里，无需每次启动前手动 set 环境变量。"""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as error:
        print(f".env 读取失败：{type(error).__name__}")


_load_dotenv()

# /analyze_health 结果文件缓存：相同 payload 24 小时内直接返回，不再调用付费 LLM
ANALYSIS_CACHE_TTL_SECONDS = float(os.getenv("ANALYSIS_CACHE_TTL_SECONDS", str(24 * 3600)))
ANALYSIS_CACHE_VERSION = "v3"  # v3: 治愈系语气硬规则上线（禁「建议」开头等），旧缓存失效
ANALYSIS_CACHE_PATH = Path(
    os.getenv("ANALYSIS_CACHE_PATH", str(Path(__file__).resolve().parent / "health_analysis_cache.json"))
)
_analysis_cache_lock = threading.Lock()


def _preheat_asr() -> None:
    """Warm up the ASR model in a background thread so the first
    /upload_voice request does not pay the model loading cost."""
    try:
        from asr_service import get_asr_model

        started = time.monotonic()
        get_asr_model()
        print(f"ASR 预热完成，耗时 {time.monotonic() - started:.1f}s")
    except Exception as error:  # noqa: BLE001 - 预热失败只记录，首次请求时再重试
        print(f"ASR 预热失败（首次请求时会重试）：{type(error).__name__}: {error}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    preheat = os.getenv("ASR_PREHEAT", "1").strip().lower() in {"1", "true", "yes", "on"}
    if preheat:
        threading.Thread(target=_preheat_asr, daemon=True, name="asr-preheat").start()
    yield


app = FastAPI(lifespan=lifespan)


def _analysis_cache_key(request: HealthAnalysisRequest) -> str:
    payload = json.dumps(
        request_payload(request),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest_input = f"{ANALYSIS_CACHE_VERSION}:{payload}"
    return hashlib.sha256(digest_input.encode("utf-8")).hexdigest()


def _load_analysis_cache() -> dict:
    try:
        with ANALYSIS_CACHE_PATH.open("r", encoding="utf-8") as cache_file:
            data = json.load(cache_file)
        if isinstance(data, dict) and isinstance(data.get("entries"), dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"version": ANALYSIS_CACHE_VERSION, "entries": {}}


def _save_analysis_cache(cache: dict) -> None:
    try:
        with ANALYSIS_CACHE_PATH.open("w", encoding="utf-8") as cache_file:
            json.dump(cache, cache_file, ensure_ascii=False)
    except OSError as error:
        print(f"分析缓存写入失败：{type(error).__name__}")


def _read_cached_analysis(key: str) -> HealthAnalysisResponse | None:
    with _analysis_cache_lock:
        cache = _load_analysis_cache()
        entry = cache["entries"].get(key)
    if not isinstance(entry, dict):
        return None
    created_at = entry.get("created_at")
    response_data = entry.get("response")
    if not isinstance(created_at, (int, float)) or not isinstance(response_data, dict):
        return None
    if time.time() - created_at >= ANALYSIS_CACHE_TTL_SECONDS:
        return None
    try:
        return HealthAnalysisResponse.model_validate(response_data)
    except ValidationError:
        return None


def _write_cached_analysis(key: str, response: HealthAnalysisResponse) -> None:
    now = time.time()
    with _analysis_cache_lock:
        cache = _load_analysis_cache()
        # 顺带清理过期条目，避免缓存文件无限增长
        fresh_entries = {
            entry_key: entry
            for entry_key, entry in cache["entries"].items()
            if isinstance(entry, dict)
            and isinstance(entry.get("created_at"), (int, float))
            and now - entry["created_at"] < ANALYSIS_CACHE_TTL_SECONDS
        }
        fresh_entries[key] = {
            "created_at": now,
            "response": response.model_dump(mode="json"),
        }
        _save_analysis_cache({"version": ANALYSIS_CACHE_VERSION, "entries": fresh_entries})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/upload_voice")
async def upload_voice(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="未提供录音文件")

    temp_path: str | None = None
    total_size = 0
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_size += len(chunk)
                if total_size > MAX_AUDIO_BYTES:
                    raise HTTPException(status_code=413, detail="录音文件过大")
                temp_file.write(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="录音文件为空")
        if total_size <= 44:
            raise HTTPException(status_code=400, detail="录音时间太短")

        with open(temp_path, "rb") as audio_file:
            header = audio_file.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise HTTPException(status_code=400, detail="仅支持 WAV 格式录音")

        try:
            info = sf.info(temp_path)
        except Exception as error:
            raise HTTPException(status_code=400, detail="WAV 文件无法读取") from error

        if info.samplerate != EXPECTED_SAMPLE_RATE:
            raise HTTPException(status_code=400, detail="录音采样率必须为 16000 Hz")
        if info.channels != EXPECTED_CHANNELS:
            raise HTTPException(status_code=400, detail="录音必须为单声道")
        if info.duration < MIN_AUDIO_SECONDS:
            raise HTTPException(status_code=400, detail="录音时间太短")

        try:
            text = recognize_audio(temp_path)
        except Exception as error:
            print(f"语音识别失败：{type(error).__name__}")
            raise HTTPException(status_code=500, detail="语音识别失败，请稍后重试") from error

        return {"text": text}
    finally:
        await file.close()
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                print("临时录音文件清理失败")


@app.post("/analyze_health")
async def analyze_health_route(request: HealthAnalysisRequest):
    cache_key = _analysis_cache_key(request)
    cached = _read_cached_analysis(cache_key)
    if cached is not None:
        print("analyze_health cache=hit")
        return cached
    try:
        result = await analyze_health(request)
    except ElexServiceError as error:
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "message": error.message}},
        )
    _write_cached_analysis(cache_key, result)
    print("analyze_health cache=miss")
    return result


# ============================================================
# 宠物聊天端点
# ============================================================


class PetChatHistoryItem(BaseModel):
    role: str = "user"
    content: str = ""


class PetChatRequest(BaseModel):
    pet_type: str = "cat"
    pet_name: str = ""
    message: str
    tier: str = ""
    history: list[PetChatHistoryItem] = []


@app.post("/pet_chat")
async def pet_chat_route(request: PetChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空。")
    try:
        reply = await pet_chat(
            pet_type=request.pet_type,
            pet_name=request.pet_name,
            message=request.message,
            history=[item.model_dump() for item in request.history],
            tier=request.tier,
        )
    except ElexServiceError as error:
        return JSONResponse(
            status_code=error.status_code,
            content={"error": {"code": error.code, "message": error.message}},
        )
    return {"reply": reply}


# ============================================================
# 云同步端点
# ============================================================

SYNC_STORE_PATH = Path(os.getenv("SYNC_STORE_PATH", str(Path(__file__).resolve().parent / "sync_store.json")))
_sync_lock = threading.Lock()


class SyncPushRequest(BaseModel):
    device_id: str
    data: dict
    version: int = 1


def _load_sync_store() -> dict:
    try:
        with SYNC_STORE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_sync_store(store: dict) -> None:
    try:
        with SYNC_STORE_PATH.open("w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False)
    except OSError as error:
        print(f"sync_store 写入失败：{error}")


@app.post("/sync")
async def sync_push(request: SyncPushRequest):
    """推送设备数据到云端存储。"""
    if not request.device_id.strip():
        raise HTTPException(status_code=400, detail="device_id 不能为空")
    with _sync_lock:
        store = _load_sync_store()
        store[request.device_id] = {
            "data": request.data,
            "version": request.version,
            "updated_at": time.time(),
        }
        _save_sync_store(store)
    return {"status": "ok", "device_id": request.device_id}


@app.get("/sync")
async def sync_pull(device_id: str):
    """拉取指定设备的最新同步数据。"""
    if not device_id.strip():
        raise HTTPException(status_code=400, detail="device_id 不能为空")
    with _sync_lock:
        store = _load_sync_store()
        entry = store.get(device_id)
    if entry is None:
        return {"status": "not_found", "device_id": device_id}
    return {"status": "ok", "device_id": device_id, **entry}
