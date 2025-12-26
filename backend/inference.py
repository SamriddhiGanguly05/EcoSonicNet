import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
import timm
import torch


@dataclass(frozen=True)
class InferenceConfig:
    model_path: str = "best_model.pth"
    train_csv_path: str = "train.csv"
    taxonomy_csv_path: str = "taxonomy.csv"
    sample_rate: int = 32000
    n_fft: int = 1024
    hop_length: int = 320
    n_mels: int = 224
    spec_size: int = 224  # 224x224


def load_taxonomy(cfg: InferenceConfig) -> pd.DataFrame:
    if not os.path.exists(cfg.taxonomy_csv_path):
        return pd.DataFrame()
    df = pd.read_csv(cfg.taxonomy_csv_path)
    if "primary_label" in df.columns:
        df["primary_label"] = df["primary_label"].astype(str)
    return df


def load_class_list(cfg: InferenceConfig) -> Tuple[str, ...]:
    """
    Stable class ordering for numeric IDs: sorted unique primary_label from train.csv.
    This mirrors the deterministic ordering used in the Streamlit app.
    """
    if not os.path.exists(cfg.train_csv_path):
        tax = load_taxonomy(cfg)
        if tax.empty or "primary_label" not in tax.columns:
            # best-effort fallback
            return tuple(str(i) for i in range(264))
        labels = tax["primary_label"].astype(str).unique().tolist()
        return tuple(sorted(labels, key=lambda x: (len(x), x)))

    df = pd.read_csv(cfg.train_csv_path, usecols=["primary_label"])
    labels = df["primary_label"].astype(str).unique().tolist()
    return tuple(sorted(labels, key=lambda x: (len(x), x)))


def create_model(num_classes: int) -> torch.nn.Module:
    return timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=num_classes,
        in_chans=1,
    )


def load_model(cfg: InferenceConfig, class_list: Tuple[str, ...]) -> torch.nn.Module:
    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Model not found at {cfg.model_path}")

    model = create_model(num_classes=len(class_list))
    checkpoint = torch.load(cfg.model_path, map_location=torch.device("cpu"))
    state_dict = checkpoint.get("state_dict", checkpoint)

    model_state_dict = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in state_dict.items() if k in model_state_dict and model_state_dict[k].shape == v.shape
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()
    return model


def preprocess_audio_to_tensor(cfg: InferenceConfig, audio_path: str) -> Tuple[torch.Tensor, int, int]:
    """
    Returns:
      - input tensor: (1,1,224,224)
      - sample_rate (int)
      - num_samples (int)
    """
    y, sr = librosa.load(audio_path, sr=cfg.sample_rate, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)

    x = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float()
    if x.shape[-1] < cfg.spec_size:
        x = torch.nn.functional.pad(x, (0, cfg.spec_size - x.shape[-1]))
    x = x[:, :, :, : cfg.spec_size]

    if x.shape[-2] != cfg.spec_size:
        x = torch.nn.functional.interpolate(x, size=(cfg.spec_size, cfg.spec_size), mode="bilinear", align_corners=False)

    return x, sr, int(y.shape[0])


def predict_topk(model: torch.nn.Module, x: torch.Tensor, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    idx = probs.argsort()[-top_k:][::-1]
    return idx, probs[idx]


def build_results(
    top_idx: np.ndarray,
    top_probs: np.ndarray,
    class_list: Tuple[str, ...],
    taxonomy: pd.DataFrame,
) -> List[Dict[str, Any]]:
    labels = [class_list[i] if i < len(class_list) else f"class_{i}" for i in top_idx.tolist()]
    df = pd.DataFrame({"primary_label": labels, "confidence": top_probs})

    if not taxonomy.empty and "primary_label" in taxonomy.columns:
        df = df.merge(taxonomy, on="primary_label", how="left")

    df["confidence_pct"] = (df["confidence"] * 100).round(2)

    # Convert NaN to None for JSON
    records: List[Dict[str, Any]] = df.replace({np.nan: None}).to_dict(orient="records")
    return records


def save_bytes_to_temp(data: bytes, filename: str) -> str:
    suffix = os.path.splitext(filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name


