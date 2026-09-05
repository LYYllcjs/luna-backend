# Luna 后端（轻量版：ONNX int8 语音识别，无需 torch）
# 适配 ClawCloud Run 等容器平台：从 Git 仓库直接构建，密钥通过平台环境变量注入。
# 运行时内存 ~0.8GB（语音峰值），推荐 1GB+ 容器。
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ASR_MODEL_DIR=/app/models \
    ASR_PREHEAT=0 \
    PORT=8000

WORKDIR /app

# 依赖（海外构建用官方 PyPI）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 下载已量化的 paraformer-zh int8 模型（约 240MB）
# 海外构建走 HuggingFace 官方源，失败时回退国内镜像
RUN mkdir -p /app/models \
    && (curl -fsSL -o /app/models/model.int8.onnx \
        "https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14/resolve/main/model.int8.onnx" \
      || curl -fsSL -o /app/models/model.int8.onnx \
        "https://hf-mirror.com/csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14/resolve/main/model.int8.onnx") \
    && (curl -fsSL -o /app/models/tokens.txt \
        "https://huggingface.co/csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14/resolve/main/tokens.txt" \
      || curl -fsSL -o /app/models/tokens.txt \
        "https://hf-mirror.com/csukuangfj/sherpa-onnx-paraformer-zh-2023-09-14/resolve/main/tokens.txt")

COPY main.py elex_service.py asr_service.py health_analysis_models.py ./

# 注意：.env 含密钥不进仓库，由 Claw 控制台环境变量注入（ELEX_API_KEY / ELEX_MODEL）

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+__import__('os').getenv('PORT','8000')+'/health',timeout=5).status==200 else 1)"

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
