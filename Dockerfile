# Luna 后端（轻量版：ONNX int8 语音识别，无需 torch）
# 容器平台通用：从 Git 仓库直接构建，密钥通过平台环境变量注入。
# 全功能版（含语音）：默认，或构建参数 ASR_ENABLED=true
#   默认使用流式 Zipformer 14M 小模型（24MB，进程 RSS ~110MB），1GB 内存可跑
# 文字版（512MB 极小内存平台）：构建参数 ASR_ENABLED=false
#   - 构建时跳过模型下载；运行时环境变量 ASR_ENABLED=0 关闭语音端点
# 如需更高识别精度（2GB+ 内存）：设 ASR_MODEL_TYPE=paraformer 并挂载大模型
FROM python:3.11-slim

ARG ASR_ENABLED=true

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ASR_MODEL_DIR=/app/models \
    ASR_PREHEAT=1 \
    ASR_MODEL_TYPE=transducer \
    PORT=8000

WORKDIR /app

# 依赖（海外构建用官方 PyPI）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 下载流式 Zipformer-zh 14M int8 模型（encoder+decoder+joiner+tokens，共约 24MB）
# 仅全功能版下载；文字版跳过，镜像更小、构建更快
# 海外构建走 HuggingFace 官方源，失败时回退国内镜像
RUN if [ "$ASR_ENABLED" = "true" ]; then \
        mkdir -p /app/models/14m \
        && BASE_HF="https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/resolve/main" \
        && BASE_MIRROR="https://hf-mirror.com/csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23/resolve/main" \
        && for f in encoder-epoch-99-avg-1.int8.onnx decoder-epoch-99-avg-1.int8.onnx joiner-epoch-99-avg-1.int8.onnx tokens.txt; do \
            curl -fsSL -o /app/models/14m/$f "$BASE_HF/$f" \
              || curl -fsSL -o /app/models/14m/$f "$BASE_MIRROR/$f"; \
        done && ls -lh /app/models/14m; \
    else \
        echo "ASR_ENABLED=false: 跳过语音模型下载（文字版镜像）"; \
    fi

COPY main.py elex_service.py asr_service.py health_analysis_models.py ./

# 注意：.env 含密钥不进仓库，由平台环境变量注入（ELEX_API_KEY / ELEX_MODEL / ASR_ENABLED）

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+__import__('os').getenv('PORT','8000')+'/health',timeout=5).status==200 else 1)"

CMD ["sh", "-c", "python -m uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
