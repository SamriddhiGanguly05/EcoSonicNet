import os
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from .inference import (
    InferenceConfig,
    build_results,
    load_class_list,
    load_model,
    load_taxonomy,
    predict_topk,
    preprocess_audio_to_tensor,
    save_bytes_to_temp,
)


def create_app() -> Flask:
    # ---- Frontend build directory ----
    dist_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
    )

    serve_ui = os.getenv("SERVE_UI", "1") == "1"

    app = Flask(
        __name__,
        static_folder=dist_dir if serve_ui else None,
        static_url_path="" if serve_ui else None,
    )

    # ---- Upload size limit (default: 200MB) ----
    app.config["MAX_CONTENT_LENGTH"] = int(
        os.getenv("MAX_CONTENT_LENGTH", str(200 * 1024 * 1024))
    )

    # ---- CORS (API only) ----
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ---- Inference configuration ----
    cfg = InferenceConfig(
        model_path=os.getenv("MODEL_PATH", "/app/best_model.pth"),
        train_csv_path=os.getenv("TRAIN_CSV", "train.csv"),
        taxonomy_csv_path=os.getenv("TAXONOMY_CSV", "taxonomy.csv"),
    )

    # ---- Load model + metadata ONCE at startup ----
    class_list = load_class_list(cfg)
    taxonomy = load_taxonomy(cfg)
    model = load_model(cfg, class_list)

    # ------------------------------------------------------------------
    # API ROUTES
    # ------------------------------------------------------------------

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "num_classes": len(class_list),
                "model_loaded": True,
                "model_path": cfg.model_path,
            }
        )

    @app.post("/api/predict")
    def predict():
        if "file" not in request.files:
            return jsonify({"error": "missing file"}), 400

        f = request.files["file"]
        if not f or not f.filename:
            return jsonify({"error": "empty upload"}), 400

        try:
            top_k = int(request.form.get("top_k", "5"))
        except ValueError:
            return jsonify({"error": "top_k must be an integer"}), 400

        top_k = max(1, min(top_k, 50))

        temp_path = None
        try:
            data = f.read()
            temp_path = save_bytes_to_temp(data, f.filename)

            x, sr, num_samples = preprocess_audio_to_tensor(cfg, temp_path)
            idx, probs = predict_topk(model, x, top_k=top_k)
            results = build_results(idx, probs, class_list, taxonomy)

            payload: Dict[str, Any] = {
                "top_k": top_k,
                "sample_rate": sr,
                "num_samples": num_samples,
                "results": results,
            }
            return jsonify(payload)

        except Exception as e:
            return jsonify({"error": str(e), "results": []}), 500

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # FRONTEND (React SPA)
    # ------------------------------------------------------------------

    if serve_ui:
        @app.route("/", defaults={"path": ""})
        @app.route("/<path:path>")
        def serve_react(path: str):
            target = os.path.join(dist_dir, path)
            if path and os.path.isfile(target):
                return send_from_directory(dist_dir, path)
            return send_from_directory(dist_dir, "index.html")

    return app
