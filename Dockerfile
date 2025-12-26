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

# System deps for audio decoding (soundfile/libsndfile) and common libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    SERVE_UI=1

# Python deps
COPY requirements.prod.txt /app/requirements.prod.txt
RUN pip install --no-cache-dir -r /app/requirements.prod.txt

# App code + model/assets
COPY backend/ /app/backend/
COPY best_model.pth /app/best_model.pth
COPY train.csv /app/train.csv
COPY taxonomy.csv /app/taxonomy.csv

# Built frontend
COPY --from=frontend_build /app/frontend/dist /app/frontend/dist

EXPOSE 10000

# Gunicorn in production
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:10000", "backend.wsgi:app"]






