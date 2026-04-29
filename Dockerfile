FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first to improve Docker layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE ${PORT:-8000}

CMD ["sh", "-c", "python manage.py migrate && python local_ia/trainer.py && uvicorn core.asgi:app --host 0.0.0.0 --port ${PORT:-8000}"]