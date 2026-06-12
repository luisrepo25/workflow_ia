# ═══════════════════════════════════════════════════════════
# STAGE 1: dependencias — instala solo lo necesario para runtime
# ═══════════════════════════════════════════════════════════
FROM python:3.12-slim AS deps

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements-runtime.txt && \
    pip cache purge && \
    find /usr/local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    rm -rf /root/.cache /tmp/*


# ═══════════════════════════════════════════════════════════
# STAGE 2: entrenamiento — instala deps extra + entrena modelos
# ═══════════════════════════════════════════════════════════
FROM python:3.12-slim AS trainer

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# Copiar librerías base desde stage anterior
COPY --from=deps /usr/local /usr/local

# Instalar dependencias extra de entrenamiento
COPY requirements-train.txt requirements-runtime.txt ./
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements-train.txt && \
    pip cache purge && \
    find /usr/local -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    rm -rf /root/.cache /tmp/*

# Copiar código fuente
COPY . .

# Migrar DB + entrenar modelos
RUN python manage.py migrate && \
    python local_ia/trainer.py && \
    python local_ia/dataset_generator.py && \
    python local_ia/train_recommender.py --epochs 10 --batch-size 8 && \
    echo "✅ Entrenamiento completado"


# ═══════════════════════════════════════════════════════════
# STAGE 3: runtime — imagen final LIGERA para producción
# ═══════════════════════════════════════════════════════════
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# Copiar solo las librerías de runtime (NO training extras)
COPY --from=deps /usr/local /usr/local

# Copiar modelos entrenados + índices
COPY --from=trainer /app/local_ia/models/ ./local_ia/models/
COPY --from=trainer /app/local_ia/index/ ./local_ia/index/

# Copiar solo el código necesario para servir la API
COPY api_ia/ ./api_ia/
COPY core/ ./core/
COPY local_ia/__init__.py ./local_ia/
COPY local_ia/inference.py ./local_ia/
COPY local_ia/recommender.py ./local_ia/
COPY manage.py .
COPY datos.json .

EXPOSE ${PORT:-8000}

# Solamente migrar + arrancar servidor (sin entrenar)
CMD ["sh", "-c", "\
	python manage.py migrate && \
	uvicorn core.asgi:app --host 0.0.0.0 --port ${PORT:-8000} \
"]