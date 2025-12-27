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

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SERVE_UI=1 \
    MODEL_PATH=/app/best_model.pth

# Python deps
COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

# App code
COPY backend/ ./backend/
COPY train.csv .
COPY taxonomy.csv .

# Download model
RUN curl -L \
  https://github.com/SamriddhiGanguly05/EcoSonicNet/releases/download/v1.0/best_model.pth \
  -o /app/best_model.pth

# Frontend
COPY --from=frontend_build /app/frontend/dist /app/frontend/dist

# 🚨 DO NOT hardcode PORT
CMD ["sh", "-c", "gunicorn backend.wsgi:app -w 1 -k gthread --threads 4 -b 0.0.0.0:$PORT"]
