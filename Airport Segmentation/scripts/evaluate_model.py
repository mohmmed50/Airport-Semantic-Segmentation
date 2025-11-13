"""
Evaluate model performance on the validation/test set.
Computes metrics: IoU per class, mean IoU, pixel accuracy, and confusion matrix.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from segmentation_models_pytorch import Unet
import matplotlib.pyplot as plt
import seaborn as sns

# Add the repository root to Python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from scripts.infer_smp_viz import get_class_names


def compute_metrics(pred, target, num_classes):
    """
    Compute IoU for each class and mean IoU
    pred, target: numpy arrays of shape (H,W)
    """
    ious = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls

        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()

        iou = intersection / (union + 1e-10)  # avoid division by zero
        ious.append(iou)

    return np.array(ious)


def compute_confusion_matrix(pred, target, num_classes):
    """
    Compute confusion matrix
    """
    mask = (target >= 0) & (target < num_classes)
    conf_matrix = np.bincount(
        num_classes * target[mask].astype(int) + pred[mask], minlength=num_classes**2
    ).reshape(num_classes, num_classes)

    return conf_matrix


def plot_confusion_matrix(conf_matrix, class_names, output_path):
    """
    Plot and save confusion matrix heatmap
    """
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="valid",
        help="Directory containing validation/test images and masks",
    )
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if not os.path.exists("best_model.pth"):
        raise FileNotFoundError("best_model.pth not found. Train the model first.")

    model = Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        classes=args.num_classes,
        activation=None,
    )
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Get class names
    class_names = list(get_class_names().values())
    class_names.insert(0, "background")  # Add background class

    # Initialize metrics
    total_ious = np.zeros(args.num_classes)
    conf_matrix = np.zeros((args.num_classes, args.num_classes))
    num_images = 0

    # Process all images in the validation/test set
    image_dir = args.data_dir  # Images are directly in the data_dir
    mask_dir = os.path.join(args.data_dir, "masks")  # Converted COCO masks

    if not os.path.exists(mask_dir):
        raise FileNotFoundError(
            f"Mask directory not found: {mask_dir}\n"
            "Run convert_coco_to_masks.py first!"
        )

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.endswith((".jpg", ".png")) and "masks" not in f
    ]

    print(f"\nEvaluating on {len(image_files)} images from {args.data_dir}...")

    for img_file in tqdm(image_files):
        # Load and preprocess image
        img_path = os.path.join(image_dir, img_file)
        mask_path = os.path.join(mask_dir, os.path.splitext(img_file)[0] + ".png")

        if not os.path.exists(mask_path):
            print(f"Warning: Mask not found for {img_file}, skipping...")
            continue

        img = Image.open(img_path).convert("RGB")
        true_mask = np.array(Image.open(mask_path))

        # Resize to model input size
        img_resized = img.resize((512, 512))
        x = np.array(img_resized).transpose(2, 0, 1).astype(np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            pred = pred.squeeze(0).cpu().numpy()

        # Resize prediction to original size for comparison
        pred_resized = Image.fromarray(pred.astype(np.uint8)).resize(
            img.size, resample=Image.NEAREST
        )
        pred_resized = np.array(pred_resized)

        # Update metrics
        total_ious += compute_metrics(pred_resized, true_mask, args.num_classes)
        conf_matrix += compute_confusion_matrix(
            pred_resized, true_mask, args.num_classes
        )
        num_images += 1

    # Calculate final metrics
    mean_ious = total_ious / num_images
    mean_iou = mean_ious.mean()

    # Normalize confusion matrix
    conf_matrix_norm = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-10)

    # Calculate per-class accuracy
    class_acc = np.diag(conf_matrix) / (conf_matrix.sum(axis=1) + 1e-10)

    # Save results
    results_file = os.path.join(args.output_dir, "evaluation_results.txt")
    with open(results_file, "w") as f:
        f.write(f"Evaluation Results on {args.data_dir} set:\n")
        f.write(f"Number of images evaluated: {num_images}\n\n")

        # Sort classes by IoU
        class_metrics = list(zip(class_names, mean_ious, class_acc))
        class_metrics_sorted = sorted(class_metrics, key=lambda x: x[1], reverse=True)

        f.write("Performance by Class (sorted by IoU):\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Class Name':<25} {'IoU':>10} {'Accuracy':>12}\n")
        f.write("-" * 50 + "\n")

        for cls_name, iou, acc in class_metrics_sorted:
            f.write(f"{cls_name:<25} {iou*100:>9.1f}% {acc*100:>11.1f}%\n")

        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Performance:\n")
        f.write(f"Mean IoU: {mean_iou*100:.1f}%\n")
        f.write(f"Mean Accuracy: {class_acc.mean()*100:.1f}%\n")

    # Plot and save confusion matrix
    plot_confusion_matrix(
        conf_matrix_norm,
        class_names,
        os.path.join(args.output_dir, "confusion_matrix.png"),
    )

    print(f"\nResults saved to {args.output_dir}/")
    print(f"Mean IoU: {mean_iou*100:.1f}%")
    print(f"Mean Accuracy: {class_acc.mean()*100:.1f}%")

    # Print top 5 best performing classes
    print("\nTop 5 Best Performing Classes (by IoU):")
    for cls_name, iou, acc in class_metrics_sorted[:5]:
        print(f"{cls_name:<25} IoU: {iou*100:>5.1f}%  Acc: {acc*100:>5.1f}%")


if __name__ == "__main__":
    main()
