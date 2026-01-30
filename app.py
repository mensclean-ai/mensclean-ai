import os
from flask import Flask, request, jsonify
from PIL import Image
import torch
import clip

device = "cpu"
torch.set_num_threads(1)
torch.set_grad_enabled(False)

print("=== PRELOADING CLIP MODEL ===")

# 🔥 起動時ロード（ここが最重要）
model, preprocess = clip.load("RN50", device="cpu", jit=True)

model.eval()
model = model.half()   # ← メモリ半減
for p in model.parameters():
    p.requires_grad = False

print("=== CLIP LOADED SUCCESSFULLY ===")
# ================= Flask =================
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
        image_input = preprocess(image).unsqueeze(0).to(device)

        with torch.inference_mode():
            embedding = model.encode_image(image_input)

        return jsonify({"embedding": embedding.cpu().numpy().tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/encode-text", methods=["POST"])
def encode_text_endpoint():
    try:
        data = request.get_json(force=True, silent=True) or {}
        texts = data.get("texts") or [data.get("text")]

        if not texts or not texts[0]:
            return jsonify({"error": "text missing"}), 400

        text_tokens = clip.tokenize(texts).to(device)

        with torch.inference_mode():
            embedding = model.encode_text(text_tokens)

        return jsonify({"embedding": embedding.cpu().numpy().tolist()})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ローカル用
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
