FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ── Instalar PyTorch CPU (mucho más liviano ~200MB vs ~5GB CUDA) ──
COPY requirements.txt .
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt && \
    pip cache purge && \
    find /usr/local/lib/python3.12 -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    rm -rf /root/.cache /tmp/*

COPY . .

EXPOSE ${PORT:-8000}

CMD ["sh", "-c", "python manage.py migrate && python local_ia/trainer.py && uvicorn core.asgi:app --host 0.0.0.0 --port ${PORT:-8000}"]