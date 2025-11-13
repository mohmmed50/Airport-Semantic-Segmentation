"""
Run inference using saved `best_model.pth`.

Usage:
  python scripts/infer_smp.py --image path/to/image.jpg --out out_mask.png

It will output a mask PNG (pixel values are predicted class ids) and a colored overlay.
"""

import argparse
import torch
from PIL import Image
import numpy as np
import os
from segmentation_models_pytorch import Unet


def colorize_mask(mask, num_classes=None):
    # simple color map: random per class (deterministic)
    if num_classes is None:
        num_classes = int(mask.max()) + 1
    np.random.seed(42)
    colors = (np.random.randint(0, 255, size=(num_classes, 3))).astype(np.uint8)
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        out[mask == c] = colors[c]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-classes", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("best_model.pth"):
        raise FileNotFoundError("best_model.pth not found. Train first.")

    # num_classes optional; if None we will assume 20 (roboflow) but better to pass explicitly
    num_classes = args.num_classes or 20
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        classes=num_classes,
        activation=None,
    )
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device).eval()

    img = Image.open(args.image).convert("RGB")
    img_resized = img.resize((512, 512))
    x = np.array(img_resized).transpose(2, 0, 1).astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        # out shape [1, C, H, W]
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # resize back to original
    pred_img = Image.fromarray(pred).resize(img.size, resample=Image.NEAREST)
    pred_img.save(args.out)
    # color overlay
    colored = colorize_mask(np.array(pred_img), num_classes)
    overlay = Image.fromarray(colored).convert("RGBA")
    base = img.convert("RGBA")
    blended = Image.blend(base, overlay, alpha=0.5)
    out_overlay = os.path.splitext(args.out)[0] + "_overlay.png"
    blended.save(out_overlay)
    print("Saved mask:", args.out)
    print("Saved overlay:", out_overlay)


if __name__ == "__main__":
    main()
