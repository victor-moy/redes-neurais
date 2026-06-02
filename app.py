import base64
import io

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageFilter
from pydantic import BaseModel

from model import MLP

# ── model setup ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = MLP().to(device)
model_loaded = False

try:
    state = torch.load("models/mlp_mnist.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    model_loaded = True
    print(f"Modelo carregado com sucesso! (device: {device})")
except FileNotFoundError:
    print("AVISO: models/mlp_mnist.pth não encontrado. Execute train.py primeiro.")

# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Classificador de Dígitos MNIST")


class ImagePayload(BaseModel):
    image: str  # base64 data-URL


def preprocess(image_b64: str) -> torch.Tensor:
    # decode base64 → PIL image
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    raw   = base64.b64decode(image_b64)
    img   = Image.open(io.BytesIO(raw)).convert("RGBA")

    # white canvas → black background, colored strokes → white
    bg    = Image.new("RGBA", img.size, (0, 0, 0, 255))
    bg.paste(img, mask=img.split()[3])          # alpha compositing
    gray  = bg.convert("L")

    arr   = np.array(gray, dtype=np.float32)

    # if background is mostly light, invert (user drew dark on light)
    if arr.mean() > 128:
        arr = 255.0 - arr

    # find bounding box of the digit
    mask  = arr > 30
    rows  = np.any(mask, axis=1)
    cols  = np.any(mask, axis=0)
    if not rows.any():
        raise ValueError("Nada desenhado")

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # crop with padding
    pad   = max(4, int(0.1 * max(rmax - rmin, cmax - cmin)))
    rmin  = max(0, rmin - pad)
    rmax  = min(arr.shape[0] - 1, rmax + pad)
    cmin  = max(0, cmin - pad)
    cmax  = min(arr.shape[1] - 1, cmax + pad)

    crop  = arr[rmin:rmax+1, cmin:cmax+1]

    # place crop into a square canvas (center it like MNIST)
    side  = max(crop.shape)
    sq    = np.zeros((side, side), dtype=np.float32)
    r0    = (side - crop.shape[0]) // 2
    c0    = (side - crop.shape[1]) // 2
    sq[r0:r0+crop.shape[0], c0:c0+crop.shape[1]] = crop

    # resize to 28×28 with anti-aliasing
    pil28 = Image.fromarray(sq.astype(np.uint8)).resize((28, 28), Image.LANCZOS)
    arr28 = np.array(pil28, dtype=np.float32) / 255.0

    # MNIST normalisation
    arr28 = (arr28 - 0.1307) / 0.3081

    tensor = torch.tensor(arr28).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,28,28)
    return tensor


@app.get("/health")
def health():
    return {"model_loaded": model_loaded, "device": str(device)}


@app.post("/predict")
def predict(payload: ImagePayload):
    if not model_loaded:
        raise HTTPException(503, "Modelo não carregado. Execute train.py primeiro.")

    try:
        tensor = preprocess(payload.image)
    except ValueError as e:
        raise HTTPException(400, str(e))

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    digit      = int(probs.argmax())
    confidence = float(probs[digit])

    return {
        "digit":         digit,
        "confidence":    confidence,
        "probabilities": probs.tolist(),
    }


# serve static files last so API routes take priority
app.mount("/", StaticFiles(directory="static", html=True), name="static")
