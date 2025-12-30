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

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SERVE_UI=1 \
    MODEL_PATH=/app/best_model.pth

COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

COPY backend backend
COPY train.csv .
COPY taxonomy.csv .

RUN curl -L \
  https://github.com/SamriddhiGanguly05/EcoSonicNet/releases/download/v1.0/best_model.pth \
  -o /app/best_model.pth

COPY --from=frontend_build /app/frontend/dist /app/frontend/dist

CMD ["sh", "-c", "gunicorn backend.wsgi:app -w 1 -k gthread --threads 4 -b 0.0.0.0:$PORT"]
