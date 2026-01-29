import torch
import open_clip
from PIL import Image

# ====================
# Device（CPU固定）
# ====================
device = torch.device("cpu")
# ====================
# Load CLIP model（起動時1回）
# ====================
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="openai",
    device=device,
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

model.eval()

# ====================
# Encode image（高速）
# ====================
def encode_image(image: Image.Image) -> torch.Tensor:
    """
    image: PIL.Image (RGB, 224x224 推奨)
    return: (1, D) 正規化済み特徴
    """
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(image)

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb
# ====================
# Encode text（高速・複数対応）
# ====================
def encode_text(texts) -> torch.Tensor:
    """
    texts: str | list[str]
    return: (N, D) 正規化済み特徴
    """
    if isinstance(texts, str):
        texts = [texts]

    tokens = tokenizer(texts)

    with torch.inference_mode():
        text_features = model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features