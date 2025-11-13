# Airport Semantic Segmentation — local training & inference

Files added:

- `requirements.txt` – Python packages required.
- `scripts/convert_coco_to_masks.py` – convert COCO segmentation to per-image mask PNGs.
- `scripts/dataset.py` – PyTorch Dataset for images + masks.
- `scripts/train_smp.py` – training script using segmentation_models_pytorch (Unet + resnet34).
- `scripts/infer_smp.py` – inference script to apply `best_model.pth` to a new image.

Quick start (PowerShell):

1. Create and activate venv

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install packages (CPU example). For GPU, install matching torch with CUDA from pytorch.org:

```powershell
pip install -r requirements.txt
```

3. Convert COCO -> masks (run for train and valid):

```powershell
python scripts/convert_coco_to_masks.py --coco train/_annotations.coco.json --images-dir train --out-masks train_masks
python scripts/convert_coco_to_masks.py --coco valid/_annotations.coco.json --images-dir valid --out-masks valid_masks
```

4. Train:

```powershell
python scripts/train_smp.py --train-images train --train-masks train_masks --val-images valid --val-masks valid_masks --epochs 10 --batch-size 2
```

5. Inference on any image:

```powershell
python scripts/infer_smp.py --image path\to\image.jpg --out out_mask.png --num-classes 20
```

Notes & next steps:

- If you run into `pycocotools` install issues on Windows, install via `pip install pycocotools-windows` or use WSL/Colab.
- The scripts are a minimal working pipeline. For better results: use stronger augmentations (albumentations), larger image sizes, tune lr, use GPU.
- I can: run a quick local dry-run (syntax checks), or actually start training here if you want and your machine has the packages and (ideally) a GPU.
