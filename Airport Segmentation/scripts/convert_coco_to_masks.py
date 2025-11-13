"""
Convert COCO segmentation annotations to per-image class masks (PNG).

Usage:
    python scripts/convert_coco_to_masks.py --coco train/_annotations.coco.json --images-dir train --out-masks train_masks

This script creates a folder with mask PNGs where each pixel value is the category_id.
"""

import os
import argparse
import json
import numpy as np
from PIL import Image
from pycocotools import mask as maskutils


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def polygons_to_mask(img_h, img_w, annotations, categories_map):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for ann in annotations:
        cat_id = ann["category_id"]
        seg = ann.get("segmentation", [])
        if not seg:
            continue
        # segmentation can be list of polygons or RLE
        if isinstance(seg, list):
            for poly in seg:
                # poly is list of x,y coords
                poly_np = np.array(poly).reshape(-1, 2)
                # create rle from polygon
                rle = maskutils.frPyObjects([poly], img_h, img_w)
                m = maskutils.decode(rle)
                if m.ndim == 3:
                    m = np.sum(m, axis=2)
                mask[m > 0] = cat_id
        else:
            # RLE
            rle = seg
            m = maskutils.decode(rle)
            if m.ndim == 3:
                m = np.sum(m, axis=2)
            mask[m > 0] = cat_id
    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--out-masks", required=True)
    args = parser.parse_args()

    with open(args.coco, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    ensure_dir(args.out_masks)

    for img_id, img in images.items():
        file_name = img["file_name"]
        h = img["height"]
        w = img["width"]
        anns = anns_by_image.get(img_id, [])
        mask = polygons_to_mask(h, w, anns, None)
        # save mask as PNG (uint8). If many classes >255, need different encoding.
        mask_img = Image.fromarray(mask)
        out_path = os.path.join(args.out_masks, os.path.splitext(file_name)[0] + ".png")
        mask_img.save(out_path)
        print("Saved mask:", out_path)


if __name__ == "__main__":
    main()
