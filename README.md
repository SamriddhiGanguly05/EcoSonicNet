# EcoSonicNet — Bioacoustic Detection Web App (React + Flask + PyTorch)

EcoSonicNet is a local web app that lets a user **upload an audio file** and get **Top‑K species predictions** with confidence and taxonomy metadata.

This repo contains:
- **ML model inference** (PyTorch + timm Vision Transformer)
- **Backend API** (Flask)
- **Frontend UI** (React + Vite)

---

## Contents
- [Project layout](#project-layout)
- [ML model overview](#ml-model-overview)
- [Audio preprocessing (inference)](#audio-preprocessing-inference)
- [Label mapping & taxonomy](#label-mapping--taxonomy)
- [Backend API (Flask)](#backend-api-flask)
- [Frontend (React)](#frontend-react)
- [Run locally](#run-locally)
- [Troubleshooting](#troubleshooting)

---

## Project layout

- `best_model.pth`: trained model weights (classification head has **206 classes**)
- `train.csv`: training metadata (used to build the **class list / index mapping**)
- `taxonomy.csv`: taxonomy metadata (common/scientific names + class group)
- `backend/`: Flask API + inference utilities
  - `backend/app.py`: API server (`/api/health`, `/api/predict`)
  - `backend/inference.py`: preprocessing + model load + prediction helpers
- `frontend/`: React app (Vite)
  - `frontend/src/App.jsx`: upload + settings + results UI
  - `frontend/vite.config.js`: dev proxy (`/api` → `http://localhost:5000`)

---

## ML model overview

- **Architecture**: Vision Transformer (ViT) from `timm` (`vit_base_patch16_224`)
- **Input**: a **224×224 mel-spectrogram** treated like a 1‑channel image
- **Output**: softmax probability over **206 classes**
- **Weights**: loaded from `best_model.pth`
- **Device**: CPU (works without CUDA)

### Why a Vision Transformer?

The model treats the mel-spectrogram as an image and performs image-style classification to predict the most likely class for the uploaded recording.

---

## Audio preprocessing (inference)

Implemented in `backend/inference.py`:

- **Resample** to 32 kHz (`sample_rate=32000`)
- **Mel spectrogram**:
  - `n_fft=1024`
  - `hop_length=320`
  - `n_mels=224`
- Convert to **dB scale** (`power_to_db`)
- **Normalize** using mean/std
- **Pad/crop** time axis to 224 and ensure final tensor shape is:
  - **(1, 1, 224, 224)**

---

## Label mapping & taxonomy

### Class index mapping

To map model output indices → `primary_label`, the backend builds a stable class list:

- Take unique `primary_label` values from `train.csv`
- Convert to string
- Sort deterministically (numeric IDs as strings) and use the index as the class id

This is done in:
- `backend/inference.py` → `load_class_list()`

### Taxonomy enrichment

Predicted labels are merged with `taxonomy.csv` to provide:
- `common_name`
- `scientific_name`
- `class_name` (e.g., Aves / Amphibia / Insecta / Mammalia, etc.)

---

## Backend API (Flask)

Source: `backend/app.py`

### `GET /api/health`

Returns:
- `ok`
- `num_classes`
- `model_path`

### `POST /api/predict`

**Request** (`multipart/form-data`):
- `file`: audio file (`.wav`, `.mp3`, `.ogg`, `.m4a`, `.flac`, …)
- `top_k`: integer (optional, default `5`, max `50`)

**Response** (`application/json`):
- `top_k`
- `sample_rate`
- `num_samples`
- `results`: array of objects:
  - `primary_label`
  - `confidence` (raw softmax probability)
  - `confidence_pct` (confidence × 100)
  - `common_name`, `scientific_name`, `class_name` (when taxonomy matches)

**Important**: the backend is designed to **always return JSON**, even on error, so the frontend doesn’t crash on parsing.

---

## Frontend (React)

Source: `frontend/src/App.jsx`

Features:
- Upload an audio file + **audio preview**
- Settings:
  - Top‑K slider
  - Confidence threshold filter (filters what’s displayed)
- Results tabs:
  - **Summary** (top prediction + table)
  - **Chart** (confidence bars)
  - **Details** (raw JSON response)

### “No-fail UI” behavior

If live inference fails (network, decoding, server error, etc.), the UI can show **simulated demo results** (clearly labeled) so the user never hits a dead end.

### Confidence display rule

The UI displays a **scaled confidence** in the range **(75%, 100%)** for presentation, while keeping the **raw model probabilities** intact in the Details/JSON.

---

## Run locally

### 1) Start backend

From the repo root:

```bash
python -m pip install -r requirements.txt
python -m backend.app
```

Backend URL: `http://127.0.0.1:5000`

### 2) Start frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend URL: `http://127.0.0.1:5173`

> Vite dev proxy is already configured: `/api/*` → `http://localhost:5000`

---

## Troubleshooting

### “Site can’t be reached”

- Confirm backend is running:
  - Open `http://127.0.0.1:5000/api/health` (should return JSON)
- Confirm frontend is running:
  - Open `http://127.0.0.1:5173`

### Windows note (multiple Python versions)

If you have multiple Python versions installed, you may already have torch installed for one interpreter (e.g., Python 3.9) but not another (e.g., Python 3.13). Prefer running the backend with the interpreter that already has:
- `torch`
- `timm`
- `librosa`

Example:

```bash
py -0p
```

---

## Legacy Streamlit UI

The original Streamlit UI is still in `streamlit_app.py`, but the recommended interface is now the React + Flask app described above.






