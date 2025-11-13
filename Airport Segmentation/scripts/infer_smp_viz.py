"""
Run inference and visualize detections with bounding boxes and labels.
Shows only predictions with confidence > 50%.

Usage:
  python scripts/infer_smp_viz.py --image path/to/image.jpg --out result.png --confidence 0.5
"""

import argparse
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
from segmentation_models_pytorch import Unet


def get_class_names():
    """Return mapping of class IDs to readable names."""
    return {
        1: "aircraft",
        2: "belt-loader",
        3: "bus",
        4: "center-line",
        5: "cone",
        6: "dolly",
        7: "equipment-restriction-line",
        8: "follow-me-car",
        9: "ground-power-unit",
        10: "ground-staff",
        11: "jet-bridge",
        12: "lavatory-service-vehicle",
        13: "misc-ground-vehicle",
        14: "passenger",
        15: "passenger-stairs",
        16: "refueler",
        17: "tether",
        18: "tug",
        19: "tug-bar",
    }


def colorize_mask(mask, num_classes=None):
    """Create color map and apply to mask."""
    if num_classes is None:
        num_classes = int(mask.max()) + 1
    np.random.seed(42)
    colors = (np.random.randint(0, 255, size=(num_classes, 3))).astype(np.uint8)
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(num_classes):
        out[mask == c] = colors[c]
    return out, colors


def draw_detections(
    image, pred_mask, confidence_mask, colors, class_names, conf_threshold=0.5
):
    """Draw bounding boxes and labels for each detected object with confidence > threshold.

    pred_mask: PIL Image or numpy 2D array of class ids (H,W)
    confidence_mask: numpy 2D array (H,W) of per-pixel confidence for predicted class
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    mask = np.array(pred_mask)

    # For each class found in the mask
    for class_id in np.unique(mask):
        if class_id == 0:
            continue

        # Binary mask where predicted class == class_id and confidence above threshold
        binary = ((mask == class_id) & (confidence_mask > conf_threshold)).astype(
            np.uint8
        )
        if not binary.any():
            continue

        # Find contours (ensure uint8 0/255)
        contours, _ = cv2.findContours(
            binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter small regions
        min_area = 100
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Region confidence (mean over pixels inside contour)
            region_mask = binary[y : y + h, x : x + w]
            if region_mask.sum() == 0:
                continue
            region_conf = float(
                np.mean(confidence_mask[y : y + h, x : x + w][region_mask > 0])
            )

            color = tuple(int(c) for c in colors[class_id])
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)

            label = (
                f"{class_names.get(class_id, f'class_{class_id}')} {region_conf:.0%}"
            )
            # Compute label size in a way compatible with different Pillow versions
            try:
                # textbbox returns (x0, y0, x1, y1)
                bbox = draw.textbbox((0, 0), label, font=font)
                label_w = int(bbox[2] - bbox[0])
                label_h = int(bbox[3] - bbox[1])
            except Exception:
                try:
                    # Older/fallback: font.getsize
                    label_w, label_h = font.getsize(label)
                    label_w = int(label_w)
                    label_h = int(label_h)
                except Exception:
                    # Last-resort heuristic
                    label_w = int(len(label) * 6)
                    label_h = 12

            lx0, ly0 = int(x), max(0, int(y - label_h - 4))
            lx1, ly1 = int(x + label_w + 6), int(y)
            draw.rectangle([lx0, ly0, lx1, ly1], fill=color)
            draw.text((lx0 + 2, ly0), label, fill="white", font=font)

    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (0-1), default: 0.5",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists("best_model.pth"):
        raise FileNotFoundError("best_model.pth not found. Train first.")

    # Load model
    num_classes = args.num_classes
    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        classes=num_classes,
        activation=None,
    )
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device).eval()

    # Load and preprocess image
    img = Image.open(args.image).convert("RGB")
    img_resized = img.resize((512, 512))
    x = np.array(img_resized).transpose(2, 0, 1).astype(np.float32) / 255.0
    x = torch.from_numpy(x).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        logits = model(x)

        # Get class predictions and confidence scores
        probs = F.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

        pred = pred.squeeze(0).cpu().numpy().astype(np.uint8)
        confidence = confidence.squeeze(0).cpu().numpy()

    # Resize predictions back to original size
    pred_img = Image.fromarray(pred).resize(img.size, resample=Image.NEAREST)
    # img.size is (width, height) and cv2.resize expects dsize=(width, height)
    confidence_resized = cv2.resize(
        confidence, img.size, interpolation=cv2.INTER_LINEAR
    )
    pred_img.save(args.out)

    # Create colored overlay (only for confident predictions)
    conf_mask = confidence_resized > args.confidence
    masked_pred = np.array(pred_img) * conf_mask
    colored, colors = colorize_mask(masked_pred, num_classes)
    overlay = Image.fromarray(colored).convert("RGBA")
    base = img.convert("RGBA")
    blended = Image.blend(base, overlay, alpha=0.5)

    # Draw bounding boxes and labels
    class_names = get_class_names()
    result = draw_detections(
        base.copy(), pred_img, confidence_resized, colors, class_names, args.confidence
    )

    # Save all outputs
    out_overlay = os.path.splitext(args.out)[0] + "_overlay.png"
    out_boxes = os.path.splitext(args.out)[0] + "_boxes.png"
    blended.save(out_overlay)
    result.save(out_boxes)

    print("Saved outputs:")
    print("- Mask:", args.out)
    print("- Overlay:", out_overlay)
    print("- Annotated:", out_boxes)


if __name__ == "__main__":
    main()
