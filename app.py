import os
from flask import Flask, request, jsonify
from PIL import Image
import torch
import clip

# ===== Render 安全設定 =====
device = "cpu"
torch.set_num_threads(1)

_model = None
_preprocess = None

def load_model():
    """CLIPモデルを最初のリクエスト時にロード"""
    global _model, _preprocess
    if _model is None:
        print("LOADING LIGHTWEIGHT CLIP RN50...")
        try:
            _model, _preprocess = clip.load("RN50", device=device)
            _model.eval()
            for p in _model.parameters():
                p.requires_grad = False
        except Exception as e:
            print("Failed to load CLIP:", e)
            raise
    return _model, _preprocess

def encode_image(image: Image.Image):
    model, preprocess = load_model()
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return model.encode_image(image_input)

def encode_text(texts):
    model, _ = load_model()
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        return model.encode_text(text_tokens)

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
        # numpy array をリストに変換
        return jsonify({"embedding": embedding.cpu().numpy().tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/encode-text", methods=["POST"])
def encode_text_endpoint():
    # Flask では request.json でも request.get_json() でもOK
    data = request.get_json(force=True)  # force=True で Content-Type が不正でも強制的にパース
    if not data or "texts" not in data:
        return jsonify({"error": "No texts provided"}), 400
    try:
        embedding = encode_text(data["texts"])
        return jsonify({"embedding": embedding.cpu().numpy().tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # 初回ロード
    print("Warming up CLIP model...")
    load_model()
    app.run(host="0.0.0.0", port=port)
