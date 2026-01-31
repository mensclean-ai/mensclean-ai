import torch

# ===== スコア計算 =====
def calc_similarity(image_feat, pos_feats, neg_feats):
    # cosine similarity
    pos_sim = (image_feat @ pos_feats.T).mean()
    neg_sim = (image_feat @ neg_feats.T).mean()

    raw_score = (pos_sim - neg_sim).item()

    # 🔥 安定化補正（揺れ防止）
    # スコアを中央に寄せることで極端な結果を減らす
    stabilized = raw_score * 0.7

    return stabilized
# ===== ランク変換（Cが平均）=====
def score_to_rank(score):
    if score >= 0.075:
        return "A"   # 超レア
    elif score >= 0.045:
        return "B"
    elif score >= 0.010:
        return "C"   # 基本ここに集中
    elif score >= -0.030:
        return "D"
    else:
        return "E"   # 超レア
