from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch

from openai import OpenAI, RateLimitError
from comment_templates import COMMENT_TEMPLATES

client = OpenAI()

app = Flask(__name__)
print("FLASK IMPORTED")

# ==================== プロンプト ====================
PART_PROMPTS = { "眉": [ 
"眉が整っていて清潔感のある男性の顔", 
"眉の形が整い自然に手入れされている", 
"眉毛が濃すぎず薄すぎずバランスが良い", 
" 眉周りの輪郭がはっきりしていて無駄毛が処理されている", 
"無駄な毛がなく整えられた眉" 
], 
"髪": [ 
"髪型が整っていて清潔感のある男性の顔", 
"髪がセットされていて自然なツヤがある", 
"寝癖がなく整えられたヘアスタイル", 
"重すぎず軽やかな印象の髪型", 
"清潔に洗われ整った髪" 
], 
"肌": [
"肌がきれいで清潔感のある男性の顔",
"肌がなめらかで健康的な印象",
"テカリが少なく自然な肌質",
"透明感のあるきれいな肌",
"清潔で整ったスキンコンディション" 
],
"フェイスライン": [
"フェイスラインがすっきりして清潔感のある男性の顔",
"輪郭がはっきりして引き締まった印象",
"あご周りが整っていてだらしなくない",
"顔のラインがシャープで爽やか",
"もたつきのない整ったフェイスライン" 
], 
}
PART_PROMPTS_NEG = {
"眉": [ 
"眉が整っておらず不潔な印象の男性の顔",
"眉毛がボサボサで手入れされていない",
"眉の形が崩れている",
"眉周りに無駄毛が多い",
"眉毛の形が不揃いで輪郭がぼやけている" 
],
"髪":[
"髪が乱れていて不潔な印象の男性の顔",
"髪がベタついて整っていない",
"寝癖がついたまま",
"重く野暮ったい髪型",
"手入れされていない髪"
], 
"肌": [
"肌荒れが目立ち不潔な印象の男性の顔",
"脂っぽくテカリが強い肌","ニキビや赤みが目立つ",
"くすみがある肌",
"清潔感に欠ける肌状態"
], 
"フェイスライン": [
"フェイスラインがだらしなく不潔な印象の男性の顔",
"輪郭がぼやけてもたついている",
"あご周りが緩んで見える",
"シャープさのない輪郭",
"すっきり感のない顔立ち"
],}
# ==================== CLIP ====================
print("PRELOADING CLIP...")
from clip_model import encode_image, encode_text
from scorer import calc_similarity, score_to_rank
print("CLIP PRELOADED")

TEXT_FEATURES_POS = None
TEXT_FEATURES_NEG = None
def get_text_features():
    global TEXT_FEATURES_POS, TEXT_FEATURES_NEG

    if TEXT_FEATURES_POS is None:
        TEXT_FEATURES_POS = {
            part: encode_text(PART_PROMPTS[part])
            for part in PART_PROMPTS
        }

        TEXT_FEATURES_NEG = {
            part: encode_text(PART_PROMPTS_NEG[part])
            for part in PART_PROMPTS_NEG
        }

    return TEXT_FEATURES_POS, TEXT_FEATURES_NEG
# ==================== Routes ====================
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/upload")
def upload():
    return app.send_static_file("upload.html")

@app.route("/loading")
def loading():
    return render_template("loading.html")

@app.route("/result")
def result():
    return render_template("result.html")

@app.route("/whats.html")
def whats():
    return render_template("whats.html")

@app.route("/imageup.html")
def imageup():
    return render_template("imageup.html")

@app.route("/girlofimage.html")
def girlofimage():
    return render_template("girlofimage.html")

@app.route("/scent.html")
def scent():
    return render_template("scent.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    print("ANALYZE START")

    if "image" not in request.files:
        return jsonify({"error": "画像がありません"}), 400

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
        image = image.resize((224, 224))
    except Exception:
        return jsonify({"error": "画像を読み込めません"}), 400

    try:
        image_feat = encode_image(image)
        text_features_pos, text_features_neg = get_text_features()
    except Exception as e:
        print("CLIP ERROR:", e)
        return jsonify({"error": "画像解析に失敗しました"}), 500

    scores = {}
    ranks = {}

    for part in text_features_pos:
        score = calc_similarity(
            image_feat,
            text_features_pos[part],
            text_features_neg[part],
        )
        rank = score_to_rank(score)

        scores[part] = round(score, 3)
        ranks[part] = rank

        print(part, score, rank)

    diagnosis_result = "\n".join(
        [f"{part}: {ranks[part]}評価" for part in ranks]
    )

    prompt = f"""
やさしく前向きな清潔感アドバイスを2〜4文で。

診断結果:
{diagnosis_result}
"""
    try:
        res = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
            max_output_tokens=200,
        )
        comment = res.output_text
    except Exception:
        comment = "全体的に清潔感のある印象です。少し意識するとさらに良くなります。"

    part_comments = {
        part: COMMENT_TEMPLATES[part][ranks[part]]
        for part in ranks
    }

    print("ANALYZE END")

    return jsonify(
        {
            "scores": scores,
            "ranks": ranks,
            "comments": part_comments,
            "comment": comment,
        }
    )

if __name__ == "__main__":
    app.run(debug=False, threaded=False)
