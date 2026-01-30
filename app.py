import os
from flask import Flask, request, jsonify
from PIL import Image
import torch
import clip

# ===== 設定 =====
device = "cpu"
torch.set_num_threads(1)  # CPU のみで軽量化

_model = None
_preprocess = None

def load_model():
    """CLIPモデルをロード（最初のリクエスト時のみ）"""
    global _model, _preprocess
    if _model is None:
        print("LOADING LIGHTWEIGHT CLIP RN50...")
        _model, _preprocess = clip.load("RN50", device=device)
        _model.eval()
        for p in _model.parameters():
            p.requires_grad = False
    return _model, _preprocess

def encode_image(image: Image.Image):
    model, preprocess = load_model()
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
        return embedding.half()  # float16 に変換

def encode_text(texts):
    model, _ = load_model()
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        embedding = model.encode_text(text_tokens)
        return embedding.half()  # float16 に変換

# ===== Flask Web サービス =====
app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({"message": "CLIP API is running!"})

@app.route("/encode-image", methods=["POST"])
def encode_image_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        image = Image.open(request.files["file"].stream).convert("RGB")
        embedding = encode_image(image)
        return jsonify({"embedding": embedding.cpu().numpy().astype(float).tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/encode-text", methods=["POST"])
def encode_text_endpoint():
    data = request.get_json(force=True)
    if not data or "texts" not in data:
        return jsonify({"error": "No texts provided"}), 400
    try:
        embedding = encode_text(data["texts"])
        return jsonify({"embedding": embedding.cpu().numpy().astype(float).tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===== 本番は gunicorn で起動 =====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Warming up CLIP model...")
    load_model()
    app.run(host="0.0.0.0", port=port)
