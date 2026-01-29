import torch
import clip
from PIL import Image

device = "cpu"
torch.set_num_threads(1)

_model = None
_preprocess = None

def load_model():
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
        return model.encode_image(image_input)


def encode_text(texts):
    model, _ = load_model()
    text_tokens = clip.tokenize(texts).to(device)
    with torch.no_grad():
        return model.encode_text(text_tokens)
