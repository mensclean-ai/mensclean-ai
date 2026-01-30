import os
from flask import Flask, request, jsonify
from PIL import Image
import torch
import clip
import json

# ===== 設定 =====
device = "cpu"
torch.set_num_threads(1)

_model = None
_preprocess = None

def load_model():
    global _model, _preprocess
    if _model is None:
        print("Loading LIGHT CLIP model...")
        _model, _preprocess = clip.load("RN50", device=device)

        _model.eval()
        for p in _model.parameters():
            p.requires_grad = False

        # メモリ節約
        torch.set_grad_enabled(False)

    return _model, _preprocess

def encode_image(image: Image.Image):
    model, preprocess = load_model()
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_input)
        return embedding.float()


def encode_text(texts):
    model, _ = load_model()
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        embedding = model.encode_text(text_tokens)
        return embedding.float()




# ===== Flask =====
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
        return jsonify({"embedding": embedding.cpu().numpy().tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/encode-text", methods=["POST"])
def encode_text_endpoint():
    try:
        data = request.get_json(force=True, silent=True) or {}
        print("BODY:", data)

        texts = data.get("texts") or [data.get("text")]

        if not texts or not texts[0]:
            return jsonify({"error": "text missing"}), 400

        embedding = encode_text(texts)

        return jsonify({
            "embedding": embedding.cpu().numpy().astype(float).tolist()
        })

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ===== ローカル実行用（本番では呼ばれない）=====
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=port)
