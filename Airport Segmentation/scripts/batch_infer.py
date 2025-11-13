"""
Batch inference script

Run the trained model on all images in an input folder and save three outputs per image:
 - {name}_mask.png      : raw predicted class mask (pixel value = class id)
 - {name}_overlay.png   : colored overlay blended with original image
 - {name}_boxes.png     : original image annotated with boxes + labels + confidence

Usage:
  python scripts/batch_infer.py --input-dir test --out-dir output_tester --model best_model.pth --confidence 0.5

If images are in the project root (e.g. `test/`), pass that path. The script will create `output_tester` if missing.
"""

import os
import sys
import argparse
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# ensure repo root on sys.path so we can import sibling scripts when run directly
if __package__ is None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from segmentation_models_pytorch import Unet
from scripts.infer_smp_viz import get_class_names, colorize_mask, draw_detections


def run_on_image(model, device, img_path, num_classes=20, confidence_thresh=0.5):
    img = Image.open(img_path).convert("RGB")
    orig_size = img.size  # (width, height)

    # preprocess (same as training/infer script)
    img_resized = img.resize((512, 512))
    x = np.array(img_resized).transpose(2, 0, 1).astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

        pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)
        confidence = confidence.squeeze(0).cpu().numpy()

    # Resize prediction back to original size
    pred_img = Image.fromarray(pred).resize(orig_size, resample=Image.NEAREST)

    # Resize confidence map correctly (cv2 expects (width, height) dsize)
    confidence_resized = cv2.resize(
        confidence, orig_size, interpolation=cv2.INTER_LINEAR
    )

    # create outputs
    pred_mask_np = np.array(pred_img)
    conf_mask = confidence_resized > confidence_thresh
    masked_pred = pred_mask_np * conf_mask

    colored, colors = colorize_mask(masked_pred, num_classes)
    overlay = Image.fromarray(colored).convert("RGBA")
    base = img.convert("RGBA")
    blended = Image.blend(base, overlay, alpha=0.5)

    # annotated image with boxes/labels
    class_names = get_class_names()
    annotated = draw_detections(
        base.copy(),
        pred_img,
        confidence_resized,
        colors,
        class_names,
        confidence_thresh,
    )

    return pred_mask_np, blended.convert("RGB"), annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Folder with input images")
    parser.add_argument(
        "--out-dir", default="output_tester", help="Folder to save outputs"
    )
    parser.add_argument(
        "--model", default="best_model.pth", help="Path to model weights"
    )
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device(args.device)

    # Load model
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        classes=args.num_classes,
        activation=None,
    )
    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model weights not found: {args.model}\nTrain the model first or point --model to a weights file."
        )
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device).eval()

    # Walk input dir for images
    image_files = [
        f
        for f in os.listdir(args.input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_files.sort()

    if len(image_files) == 0:
        print("No images found in", args.input_dir)
        return

    print(f"Running inference on {len(image_files)} images -> {args.out_dir}")

    for img_name in image_files:
        img_path = os.path.join(args.input_dir, img_name)
        print("Processing", img_name)
        pred_mask, overlay_img, annotated_img = run_on_image(
            model,
            device,
            img_path,
            num_classes=args.num_classes,
            confidence_thresh=args.confidence,
        )

        base_name = os.path.splitext(img_name)[0]
        mask_out = os.path.join(args.out_dir, f"{base_name}_mask.png")
        overlay_out = os.path.join(args.out_dir, f"{base_name}_overlay.png")
        boxes_out = os.path.join(args.out_dir, f"{base_name}_boxes.png")

        # Save mask as uint8 image where pixel value == class id
        Image.fromarray(pred_mask.astype(np.uint8)).save(mask_out)
        overlay_img.save(overlay_out)
        annotated_img.save(boxes_out)

    print("Done.")


if __name__ == "__main__":
    main()
