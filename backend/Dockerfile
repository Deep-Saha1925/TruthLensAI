# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Build args: INSTALL_TORCH=true adds PyTorch (CPU wheels by default; GPU: see docker-compose.gpu.yml).
ARG INSTALL_TORCH=false
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENVIRONMENT=production \
    HOME=/tmp

WORKDIR /srv

COPY requirements.txt requirements-torch.txt ./
RUN pip install -r requirements.txt \
    && if [ "$INSTALL_TORCH" = "true" ]; then \
         pip install torch torchvision --index-url "$TORCH_INDEX_URL" \
         && pip install -r requirements-torch.txt; \
       fi

COPY app ./app

RUN useradd --system --uid 10001 --no-create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

# One worker: the result store and rate limiter live in process memory.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]