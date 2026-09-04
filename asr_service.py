import os
import threading
from typing import Any

import numpy as np
import soundfile as sf

# sherpa-onnx 运行时：ONNX int8 量化模型推理，无需 torch/funasr，
# 内存占用从 ~2GB 降到 ~450MB，可在 1GB 内存免费平台运行。
import sherpa_onnx

_model: Any | None = None
_model_lock = threading.Lock()

_MODEL_DIR = os.getenv("ASR_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
_MODEL_PATH = os.getenv("ASR_ONNX_MODEL", os.path.join(_MODEL_DIR, "model.int8.onnx"))
_TOKENS_PATH = os.getenv("ASR_TOKENS", os.path.join(_MODEL_DIR, "tokens.txt"))


def _available() -> bool:
    return os.path.exists(_MODEL_PATH) and os.path.exists(_TOKENS_PATH)


def get_asr_model() -> Any:
    """Lazily create one recognizer per worker process (int8 CPU)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not _available():
                    raise RuntimeError(
                        f"ASR 模型文件缺失：{_MODEL_PATH} / {_TOKENS_PATH}"
                    )
                _model = sherpa_onnx.OfflineRecognizer.from_paraformer(
                    paraformer=_MODEL_PATH,
                    tokens=_TOKENS_PATH,
                    num_threads=int(os.getenv("ASR_NCPU", "2")),
                    sample_rate=16000,
                    feature_dim=80,
                    decoding_method="greedy_search",
                )
    return _model


def _load_samples(audio_path: str) -> tuple[np.ndarray, int]:
    """Read wav audio, resample to 16 kHz mono if needed."""
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    # 立体声转单声道
    samples = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    target_sr = 16000
    if sr != target_sr:
        # 线性重采样足够满足短语音需求，避免引入重依赖
        duration = samples.shape[0] / sr
        target_len = max(1, int(duration * target_sr))
        x_old = np.linspace(0.0, 1.0, samples.shape[0], endpoint=False)
        x_new = np.linspace(0.0, 1.0, target_len, endpoint=False)
        samples = np.interp(x_new, x_old, samples).astype("float32")
        sr = target_sr
    return samples, sr


def _text_from_result(result: Any) -> str:
    if isinstance(result, list):
        parts = [_text_from_result(item) for item in result]
        return "".join(parts).strip()
    if isinstance(result, dict):
        return str(result.get("text", "") or "").strip()
    return str(getattr(result, "text", "") or "").strip()


def recognize_audio(audio_path: str) -> str:
    """Recognize a local audio file and return non-empty text or raise an error."""
    samples, sr = _load_samples(audio_path)
    if samples.size == 0:
        raise ValueError("未识别到有效语音内容")

    recognizer = get_asr_model()
    stream = recognizer.create_stream()
    stream.accept_waveform(sr, samples)
    recognizer.decode_stream(stream)
    text = _text_from_result(stream.result)
    if not text:
        raise ValueError("未识别到有效语音内容")
    return text
