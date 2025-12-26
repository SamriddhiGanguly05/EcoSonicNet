# ---------- Frontend build ----------
FROM node:22-alpine AS frontend_build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---------- Backend runtime ----------
FROM python:3.10-slim AS runtime
WORKDIR /app

# System deps + curl for model download
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    SERVE_UI=1 \
    MODEL_PATH=/app/best_model.pth

# Python deps
COPY requirements.prod.txt /app/requirements.prod.txt
RUN pip install --no-cache-dir -r /app/requirements.prod.txt

# App code + data
COPY backend/ /app/backend/
COPY train.csv /app/train.csv
COPY taxonomy.csv /app/taxonomy.csv

# Download model from GitHub Releases
RUN curl -L \
  https://github.com/SamriddhiGanguly05/EcoSonicNet/releases/download/v1.0/best_model.pth \
  -o /app/best_model.pth

# Built frontend
COPY --from=frontend_build /app/frontend/dist /app/frontend/dist

EXPOSE 10000

CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:10000", "backend.wsgi:app"]
