


# 🛫 Airport Semantic Segmentation — Local Training & Inference

A **complete local pipeline** for semantic segmentation of airport surfaces.  
Includes dataset conversion, data loading, model training (U-Net + ResNet34), and inference scripts — all built with **PyTorch** and **Segmentation Models PyTorch (SMP)**.

---

## 📁 Project Structure

| File / Folder | Description |
|----------------|-------------|
| `requirements.txt` | List of required Python packages |
| `scripts/convert_coco_to_masks.py` | Converts COCO-format annotations into per-image binary masks |
| `scripts/dataset.py` | Custom PyTorch Dataset for images and masks |
| `scripts/train_smp.py` | Training script using U-Net (ResNet34 encoder) |
| `scripts/infer_smp.py` | Inference script to generate masks using `best_model.pth` |

---

## ⚙️ Quick Start (Windows PowerShell)

### 1️⃣ Create and activate a virtual environment
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
````

---

### 2️⃣ Install dependencies

> 💡 For GPU training, install the correct CUDA-compatible PyTorch version from [pytorch.org](https://pytorch.org).

```powershell
pip install -r requirements.txt
```

---

### 3️⃣ Convert COCO annotations to per-image masks

Run for both `train` and `valid` sets:

```powershell
python scripts/convert_coco_to_masks.py --coco train/_annotations.coco.json --images-dir train --out-masks train_masks
python scripts/convert_coco_to_masks.py --coco valid/_annotations.coco.json --images-dir valid --out-masks valid_masks
```

---

### 4️⃣ Train the model

```powershell
python scripts/train_smp.py `
  --train-images train `
  --train-masks train_masks `
  --val-images valid `
  --val-masks valid_masks `
  --epochs 10 `
  --batch-size 2
```

---

### 5️⃣ Run inference on a new image

```powershell
python scripts/infer_smp.py --image path\to\image.jpg --out out_mask.png --num-classes 20
```

---

## 🧠 Model Details

| Parameter        | Description                                 |
| ---------------- | ------------------------------------------- |
| **Architecture** | U-Net                                       |
| **Encoder**      | ResNet34                                    |
| **Framework**    | PyTorch + Segmentation Models PyTorch (SMP) |
| **Loss**         | CrossEntropy / Dice (configurable)          |
| **Metrics**      | IoU, F1 Score                               |
| **Input**        | RGB image                                   |
| **Output**       | Multi-class mask (e.g., 20 classes)         |

---

## ⚡ Tips & Recommendations

* 💥 If you face installation errors with `pycocotools` on Windows:

  ```powershell
  pip install pycocotools-windows
  ```

  or use **WSL** / **Google Colab**.

* 🚀 For better performance:

  * Use **Albumentations** for data augmentation.
  * Train with **larger image sizes**.
  * Tune **learning rate** and **batch size**.
  * Use a **GPU** (CUDA) for faster training.

---

## 🧪 Optional

You can:

* Perform a **local dry run** (syntax verification).
* Start **full training** locally if your machine has all dependencies and GPU support.

---

## 🧩 Example Outputs

| Input Image                     | Predicted Mask                |
| ------------------------------- | ----------------------------- |
|![Input](Airport%20Segmentation/test/000044_jpg.rf.97483db41fa321f0e92d5cbc8f5cf460.jpg) | ![Mask](Airport%20Segmentation/output_tester/000044_jpg.rf.97483db41fa321f0e92d5cbc8f5cf460_boxes.png) |

---

## 🧰 Tech Stack

| Tool                           | Usage                      |
| ------------------------------ | -------------------------- |
| 🧠 PyTorch                     | Model training & inference |
| 🧩 segmentation_models_pytorch | U-Net implementation       |
| 🖼️ OpenCV / PIL               | Image processing           |
| 📊 NumPy / Pandas              | Data handling              |
| 🧾 pycocotools                 | COCO format handling       |
| ⚡ Albumentations *(optional)*  | Data augmentation          |


---
## 📜 License

This project is open for educational and research purposes.
© 2025
---
