# Luna 后端（轻量版：ONNX int8 语音识别，无需 torch）
# 构建时从镜像源下载 int8 模型，运行时内存 ~0.8GB（推荐 1GB+ 容器）。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ASR_MODEL_DIR=/app/models

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载已量化的 paraformer-zh int8 模型（约 240MB）
RUN mkdir -p /app/models \
    && curl -fsSL -o /app/models/model.int8.onnx \
      "https://hf-mirror.com/csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14/resolve/main/model.int8.onnx" \
    && curl -fsSL -o /app/models/tokens.txt \
      "https://hf-mirror.com/csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14/resolve/main/tokens.txt"

COPY main.py elex_service.py asr_service.py health_analysis_models.py ./

# .env 由部署时注入（环境变量）或复制进去
COPY .env* ./

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
