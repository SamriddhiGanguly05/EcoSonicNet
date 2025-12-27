# ---------- Frontend build ----------
FROM node:22-alpine AS frontend_build
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ---------- Backend runtime ----------
FROM python:3.10-slim
WORKDIR /app

# System dependencies (audio + curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SERVE_UI=1 \
    MODEL_PATH=/app/best_model.pth

# Install Python dependencies (CPU-only torch)
COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

# Backend source
COPY backend/ ./backend/

# Data files
COPY train.csv .
COPY taxonomy.csv .

# Download trained model (GitHub Release)
RUN curl -L \
  https://github.com/SamriddhiGanguly05/EcoSonicNet/releases/download/v1.0/best_model.pth \
  -o /app/best_model.pth

# Copy built frontend into backend-served static directory
COPY --from=frontend_build /app/frontend/dist /app/frontend/dist

# Render provides PORT automatically
EXPOSE 10000

# IMPORTANT: use $PORT (Render injects it)
CMD gunicorn backend.wsgi:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 4
