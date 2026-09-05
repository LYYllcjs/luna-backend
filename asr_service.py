import os
import threading
from typing import Any

import numpy as np
import soundfile as sf

# sherpa-onnx 运行时：ONNX int8 量化模型推理，无需 torch/funasr。
# 模型有两种，由环境变量 ASR_MODEL_TYPE 切换：
#   "transducer"（默认）：流式 Zipformer 14M（~24MB，进程 RSS ~110MB），
#       适配 1GB 内存小服务器（阿里云轻量 2核1G 实测稳定）。
#   "paraformer"：非流式 Paraformer 220M（233MB，推理峰值 ~805MB），
#       识别精度更高，需 2GB+ 内存，大内存/本地开发可用。
# sherpa_onnx 为惰性导入——小内存平台（ASR_ENABLED=0）即使没装
# sherpa-onnx 也能正常启动并提供文字服务。

_model: Any | None = None
_model_lock = threading.Lock()

_MODEL_DIR = os.getenv("ASR_MODEL_DIR", os.path.join(os.path.dirname(__file__), "models"))
_PARAFORMER_PATH = os.getenv("ASR_ONNX_MODEL", os.path.join(_MODEL_DIR, "model.int8.onnx"))
_TOKENS_PATH = os.getenv("ASR_TOKENS", os.path.join(_MODEL_DIR, "tokens.txt"))

# 流式 Zipformer 默认放在 models/14m/ 下（encoder/decoder/joiner 三件套）
_ZHIPU_DIR = os.getenv("ASR_ZHIPU_DIR", os.path.join(_MODEL_DIR, "14m"))
_ZHIPU_ENCODER = os.getenv(
    "ASR_TRANS_ENCODER", os.path.join(_ZHIPU_DIR, "encoder-epoch-99-avg-1.int8.onnx")
)
_ZHIPU_DECODER = os.getenv(
    "ASR_TRANS_DECODER", os.path.join(_ZHIPU_DIR, "decoder-epoch-99-avg-1.int8.onnx")
)
_ZHIPU_JOINER = os.getenv(
    "ASR_TRANS_JOINER", os.path.join(_ZHIPU_DIR, "joiner-epoch-99-avg-1.int8.onnx")
)
_ZHIPU_TOKENS = os.getenv("ASR_TRANS_TOKENS", os.path.join(_ZHIPU_DIR, "tokens.txt"))


def asr_enabled() -> bool:
    """语音识别总开关：ASR_ENABLED=0/false/no 时关闭（小内存平台用）。"""
    return os.getenv("ASR_ENABLED", "1").strip().lower() not in {"0", "false", "no", "off"}


def _model_type() -> str:
    """识别模型类型：'transducer'（小，默认）或 'paraformer'（大）。"""
    return os.getenv("ASR_MODEL_TYPE", "transducer").strip().lower()


def _available() -> bool:
    if _model_type() == "paraformer":
        return os.path.exists(_PARAFORMER_PATH) and os.path.exists(_TOKENS_PATH)
    return (
        os.path.exists(_ZHIPU_ENCODER)
        and os.path.exists(_ZHIPU_DECODER)
        and os.path.exists(_ZHIPU_JOINER)
        and os.path.exists(_ZHIPU_TOKENS)
    )


def get_asr_model() -> Any:
    """Lazily create one recognizer per worker process (int8 CPU)。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not asr_enabled():
                    raise RuntimeError("语音识别已被 ASR_ENABLED=0 关闭")
                if not _available():
                    raise RuntimeError(
                        "ASR 模型文件缺失，请检查 models/ 目录（transducer 需要 "
                        "models/14m/ 下 encoder/decoder/joiner/tokens 四件）"
                    )
                try:
                    import sherpa_onnx
                except ImportError as error:
                    raise RuntimeError(
                        "sherpa-onnx 未安装，语音识别不可用"
                    ) from error

                ncpu = int(os.getenv("ASR_NCPU", "2"))
                if _model_type() == "paraformer":
                    _model = sherpa_onnx.OfflineRecognizer.from_paraformer(
                        paraformer=_PARAFORMER_PATH,
                        tokens=_TOKENS_PATH,
                        num_threads=ncpu,
                        sample_rate=16000,
                        feature_dim=80,
                        decoding_method="greedy_search",
                    )
                else:
                    # 流式 Zipformer-transducer：内存占用约为 paraformer 的 1/7，
                    # 文件解码时尾部补静音 flush 最后一段结果即可。
                    _model = sherpa_onnx.OnlineRecognizer.from_transducer(
                        tokens=_ZHIPU_TOKENS,
                        encoder=_ZHIPU_ENCODER,
                        decoder=_ZHIPU_DECODER,
                        joiner=_ZHIPU_JOINER,
                        num_threads=ncpu,
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
    if isinstance(result, str):
        return result.strip()
    return str(getattr(result, "text", "") or "").strip()


def recognize_audio(audio_path: str) -> str:
    """Recognize a local audio file and return non-empty text or raise an error。"""
    samples, sr = _load_samples(audio_path)
    if samples.size == 0:
        raise ValueError("未识别到有效语音内容")

    recognizer = get_asr_model()
    stream = recognizer.create_stream()
    stream.accept_waveform(sr, samples)

    if _model_type() == "paraformer":
        recognizer.decode_stream(stream)
        text = _text_from_result(stream.result)
    else:
        # 流式模型：尾部补 0.5s 静音把最后一帧结果冲出来，再循环解码到空
        tail = np.zeros(int(0.5 * sr), dtype=np.float32)
        stream.accept_waveform(sr, tail)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        text = _text_from_result(recognizer.get_result(stream))

    if not text:
        raise ValueError("未识别到有效语音内容")
    return text
