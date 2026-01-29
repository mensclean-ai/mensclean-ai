import torch

def calc_similarity(image_feat, pos_feats, neg_feats):
    # cosine similarity
    pos_sim = (image_feat @ pos_feats.T).mean()
    neg_sim = (image_feat @ neg_feats.T).mean()

    score = (pos_sim - neg_sim) * 10
    return float(score.item())

def score_to_rank(score):
    if score   >= 0.080:
        return "A"
    elif score >= 0.065:
        return "B"
    elif score >= 0.035:
        return "C"
    elif score >=-0.020:
        return "D"
    else:
        return "E"