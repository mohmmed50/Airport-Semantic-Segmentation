"""
Train a simple Unet model using segmentation_models_pytorch on the produced masks.

Usage (after installing requirements):
  1) Convert COCO annotations to masks:
     python scripts/convert_coco_to_masks.py --coco train/_annotations.coco.json --images-dir train --out-masks train_masks
     python scripts/convert_coco_to_masks.py --coco valid/_annotations.coco.json --images-dir valid --out-masks valid_masks

  2) Train:
     python scripts/train_smp.py --train-images train --train-masks train_masks --val-images valid --val-masks valid_masks --epochs 20

Notes: small demo training loop (CPU/GPU aware). Adjust hyperparams for your GPU.
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from segmentation_models_pytorch import Unet

# When running this file directly (python scripts/train_smp.py) the package
# imports can fail because the project root isn't on sys.path. Ensure the
# repository root is first on sys.path so `from scripts.dataset import ...`
# works both when running as a module and when running the file directly.
if __package__ is None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

from scripts.dataset import SegmentationDataset


def get_transforms():
    # minimal transforms, user can expand
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    train_transform = A.Compose(
        [
            A.Resize(512, 512),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(512, 512),
        ]
    )
    return train_transform, val_transform


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_t, val_t = get_transforms()

    train_ds = SegmentationDataset(
        args.train_images, args.train_masks, transforms=train_t
    )
    val_ds = SegmentationDataset(args.val_images, args.val_masks, transforms=val_t)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # determine number of classes from masks (naive: scan train_masks for max id)
    max_class = 1
    for mask_name in os.listdir(args.train_masks):
        import numpy as np
        from PIL import Image

        m = np.array(Image.open(os.path.join(args.train_masks, mask_name)))
        if m.max() > max_class:
            max_class = int(m.max())

    num_classes = max_class + 1  # assume class ids start at 0
    print("Detected num classes:", num_classes)

    model = Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=num_classes,
        activation=None,
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = 1e9
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running += loss.item()
            pbar.set_postfix(loss=running / (pbar.n + 1))

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch+1} val_loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best_model.pth")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", required=True)
    parser.add_argument("--train-masks", required=True)
    parser.add_argument("--val-images", required=True)
    parser.add_argument("--val-masks", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
