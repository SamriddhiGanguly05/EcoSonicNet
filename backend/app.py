import os
import urllib.request
import gradio as gr
import torch

from inference import (
    InferenceConfig,
    build_results,
    load_class_list,
    load_model,
    load_taxonomy,
    predict_topk,
    preprocess_audio_to_tensor,
)

# -------------------------------------------------------
# 🔽 Download model from GitHub Releases (once)
# -------------------------------------------------------
MODEL_URL = "https://github.com/SamriddhiGanguly05/EcoSonicNet/releases/download/v1.0/best_model.pth"
MODEL_PATH = "best_model.pth"

if not os.path.exists(MODEL_PATH):
    print("⬇️ Downloading model weights...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Model downloaded")

# -------------------------------------------------------
# 🧠 Inference Configuration
# -------------------------------------------------------
cfg = InferenceConfig(
    model_path=MODEL_PATH,
    train_csv_path="train.csv",
    taxonomy_csv_path="taxonomy.csv",
)

print("📂 Loading class list and taxonomy...")
class_list = load_class_list(cfg)
taxonomy = load_taxonomy(cfg)

print("🧠 Loading HTSAT-Swin model...")
model = load_model(cfg, class_list)
model.eval()

print("✅ Model ready")

# -------------------------------------------------------
# 🎵 Prediction Function (Gradio)
# -------------------------------------------------------
def predict(audio, top_k=5):
    """
    audio: filepath from Gradio
    """
    if audio is None:
        return {"error": "No audio file provided"}

    try:
        x, sr, num_samples = preprocess_audio_to_tensor(cfg, audio)
        idx, probs = predict_topk(model, x, top_k=top_k)
        results = build_results(idx, probs, class_list, taxonomy)

        # Convert to Gradio Label format
        output = {
            r["species"]: float(r["probability"])
            for r in results
        }

        return output

    except Exception as e:
        return {"error": str(e)}

# -------------------------------------------------------
# 🖥️ Gradio UI
# -------------------------------------------------------
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Audio(type="filepath", label="Upload Audio"),
        gr.Slider(1, 10, value=5, step=1, label="Top-K Predictions"),
    ],
    outputs=gr.Label(num_top_classes=5, label="Predicted Species"),
    title="EcoSonicNet – Bioacoustic Species Classifier",
    description=(
        "HTSAT-Swin Transformer for multi-species bioacoustic monitoring. "
        "Upload an audio clip to identify species."
    ),
    allow_flagging="never",
)

interface.launch()
