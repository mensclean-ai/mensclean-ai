import os, random
from flask import Flask, request, jsonify
from PIL import Image
import torch
import clip
import numpy as np
from scorer import calc_similarity, score_to_rank
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

device = "cpu"
torch.set_num_threads(1)
torch.set_grad_enabled(False)

print("=== PRELOADING CLIP MODEL ===")
model, preprocess = clip.load("RN50", device=device, jit=True)
model.eval().half()
for p in model.parameters():
    p.requires_grad = False
print("=== CLIP LOADED SUCCESSFULLY ===")

app = Flask(__name__)

# ================== テキスト定義 ==================
CLEAN_PROMPTS = [
    "清潔感のある男性の顔",
    "肌が綺麗で爽やかな男性",
    "髪が整っている男性",
    "眉が整っている男性",
    "清楚で好印象な男性"
]

DIRTY_PROMPTS = [
    "不潔に見える男性",
    "肌が荒れている男性",
    "髪が乱れている男性",
    "だらしない印象の男性"
]

ADVICE_DB = {
"A":[
"今の清潔習慣をキープしましょう",
"ヘアスタイルの維持が好印象を保つ鍵",
"眉の形が綺麗なので維持を意識",
"肌の透明感が強みです",
"服の清潔感も合わせると最強",
"定期的な散髪を継続しましょう",
"保湿ケアを続けるとさらに安定",
"写真写りがとても良いタイプ",
"横顔の印象も整っています",
"第一印象で損しない顔立ちです"
],
"B":[
"眉を少し整えるとAに近づきます",
"肌の保湿ケアで印象UP",
"前髪のボリューム調整がおすすめ",
"ヒゲの処理を丁寧にすると◎",
"髪のツヤを出すスタイリングを",
"洗顔後の保湿を習慣化",
"目元の印象が整うとさらに好印象",
"フェイスラインの清潔感が鍵",
"寝ぐせ対策で評価UP",
"光の当たり方で印象が変わるタイプ"
],
"C":[
"髪型を整えると一気に印象UP",
"肌ケアを始めると清潔感が出ます",
"眉の形を整えるだけで変わるタイプ",
"前髪の重さが少しもったいない",
"洗顔習慣を見直すと良い",
"髪のボサつきを抑えると改善",
"ヒゲ剃り後の保湿が大事",
"目元の明るさを意識すると◎",
"髪にツヤを出すと清潔感増",
"清潔習慣で化けるポテンシャルあり"
],
"D":[
"まずは眉と髪の手入れが最優先",
"肌ケアを始めると大きく改善",
"寝ぐせ対策が必要",
"ヒゲ処理を丁寧に",
"髪のボリューム整理を",
"洗顔不足が印象に影響",
"清潔感の土台作りが重要",
"前髪の重さが印象を下げている",
"眉周りを整えるだけで変わる",
"今は伸びしろが大きい状態"
],
"E":[
"髪と眉を整えるのが最優先",
"肌ケアを始めると劇的改善",
"洗顔・保湿を習慣に",
"ヒゲの処理で印象が激変",
"髪型を変えるだけで別人級",
"寝ぐせの改善が必要",
"まずは基本の清潔習慣から",
"髪のボリューム整理が鍵",
"眉周りを整えると清潔感UP",
"伸びしろMAXの状態です"
]
}

# ================== 女性コメント10種 ==================
COMMENT_DB = {
"A":[
"すごく爽やか！清潔感あります✨",
"第一印象かなり良いです",
"自然体で好印象なタイプ",
"一緒に歩きたい清潔感",
"写真写りめちゃくちゃ良い",
"好感度高い雰囲気",
"安心感ある見た目",
"ちゃんとしてる感ある",
"清潔男子って感じ",
"女子ウケ安定タイプ"
],
"B":[
"かなり好印象です！",
"ちょっと整えたらもっと良くなる",
"清潔感ある方だと思う",
"普通以上の印象",
"悪い印象は全くない",
"少しの改善で化けそう",
"雰囲気は良い感じ",
"ちゃんとすればモテそう",
"爽やか寄り",
"あと一歩で上位層"
],
"C":[
"普通くらいの印象",
"ちょっと整えたら良くなりそう",
"悪くはないけど惜しい",
"もう少し清潔感欲しいかも",
"整えたら全然変わりそう",
"ポテンシャルは感じる",
"今は平均的かな",
"改善したら好印象になりそう",
"雰囲気は悪くない",
"清潔感を足したい感じ"
],
"D":[
"少し清潔感が足りないかも",
"整えたら絶対変わるタイプ",
"今はもったいない印象",
"清潔感があると全然違うと思う",
"改善すれば良くなりそう",
"少しだらしなく見えるかも",
"印象が弱いかも",
"ケアすればかなり変わる",
"今は損してる感じ",
"整えたら好印象になりそう"
],
"E":[
"第一印象がもったいない",
"整えたら絶対良くなる",
"今は清潔感が不足気味",
"伸びしろはかなりある",
"手入れすれば印象激変タイプ",
"今は損してる状態",
"改善で大きく変われる",
"ポテンシャル高いのにもったいない",
"清潔習慣で化けそう",
"整えたら別人レベル"
]
}

# ================== 診断API ==================
@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    try:
        image = Image.open(request.files["image"].stream).convert("RGB")
        image.thumbnail((224, 224))
        image_input = preprocess(image).unsqueeze(0).to(device)
        text_inputs = clip.tokenize(CLEAN_PROMPTS + DIRTY_PROMPTS).to(device)

        with torch.inference_mode():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        sims = (image_features @ text_features.T).squeeze().cpu().numpy()

        clean_score = np.mean(sims[:len(CLEAN_PROMPTS)])
        dirty_score = np.mean(sims[len(CLEAN_PROMPTS):])
        score = round(float(clean_score - dirty_score), 3)

        rank = score_to_rank(score)

        result = {
            "rank": rank,
            "score": score,
            "advice": random.sample(ADVICE_DB[rank], 2),
            "female_comment": random.choice(COMMENT_DB[rank])
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
