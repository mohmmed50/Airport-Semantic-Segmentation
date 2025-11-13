# مشروع: فصل المقاطع الدلالية في مطارات (Airport Semantic Segmentation)

ملف README احترافي باللغة العربية يشرح المشروع، طريقة التشغيل، والتركيبة الفنية.

## مقدمة

هذا المشروع عبارة عن أنبوب عمل كامل لتدريب ونشر نموذج فصل مقاطع دلالية (semantic segmentation) مخصص لصور ساحات المطار. يعتمد المشروع على مكتبة segmentation_models_pytorch (UNet + encoder ResNet34) ويتعامل مع بيانات مسجلة بصيغة COCO، مع أدوات لتحويل تعليقات COCO إلى أقنعة (PNG) لكل صورة، تدريب النموذج، وعمل الاستدلال (inference) وتصور النتائج.

المخرجات المتوقعة:

- ملف `best_model.pth` يحتوي أوزان النموذج الأفضل.
- أقنعة ناتجة عن الاستدلال بصيغة PNG حيث قيمة كل بكسل تمثل معرف الفئة (class id).
- صور تراكب ملونة (overlay) للعرض البصري.
- نتائج تقييم (IoU, confusion matrix) محفوظة في `evaluation_results/`.

## هيكل المجلدات (مهم)

- `train/`, `valid/`, `test/` : مجلدات الصور (ملفات الصور). كل مجلد يحتوي على `_annotations.coco.json` بحسب تنسيق COCO.
- `train_masks/`, `valid_masks/` : مجلدات الأقنعة الناتجة (بعد تحويل COCO).
- `scripts/` : سكربتات مساعدة:
  - `convert_coco_to_masks.py` : تحويل تعليقات COCO إلى أقنعة PNG لكل صورة.
  - `dataset.py` : تعريف PyTorch Dataset لقراءة الصور والأقنعة وتحويلها.
  - `train_smp.py` : حلقة تدريب بسيطة باستخدام segmentation_models_pytorch.
  - `infer_smp.py` : استدلال على صورة واحدة وإخراج القناع وOverlay.
  - `infer_smp_viz.py` : استدلال مع رسم حدود ومؤشرات ثقة لكل كائن.
  - `evaluate_model.py` : حساب المقاييس (IoU, mean IoU, دقة البكسل، مصفوفة الالتباس).
- `best_model.pth` : مثال/نموذج محفوظ (أو الناتج بعد التدريب).
- `requirements.txt` : تبعيات المشروع.

## المتطلبات

المتطلبات الأساسية (مذكورة في `requirements.txt`):

- Python 3.8+
- torch >= 2.0.0
- torchvision
- opencv-python
- Pillow
- albumentations
- segmentation-models-pytorch
- pycocotools (على ويندوز قد تحتاج `pycocotools-windows`)
- tqdm
- numpy

ملاحظة: لاختيار نسخة Torch المتوافقة مع الـ CUDA على جهازك، راجع https://pytorch.org.

## إعداد البيئة (PowerShell / Windows)

1. إنشاء وتفعيل بيئة افتراضية:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. تثبيت المتطلبات:

```powershell
pip install -r requirements.txt
# إذا واجهت مشاكل مع pycocotools على ويندوز:
# pip install pycocotools-windows
```

(بدلاً من ذلك، استخدم WSL أو Colab لتفادي المشاكل الخاصة بالويندوز في حالات معينة).

## تحضير البيانات: تحويل COCO إلى أقنعة

يفترض المشروع أن لديك تعليقات COCO لكل من `train` و `valid` (ملف `_annotations.coco.json`). السكربت `convert_coco_to_masks.py` ينشئ قناعًا لكل صورة بقيمة بكسل = `category_id`.

أمثلة (PowerShell):

```powershell
python scripts/convert_coco_to_masks.py --coco train/_annotations.coco.json --images-dir train --out-masks train_masks
python scripts/convert_coco_to_masks.py --coco valid/_annotations.coco.json --images-dir valid --out-masks valid_masks
```

ملاحظة: إذا كانت معرفات الفئات أكبر من 255، فحاجة لتشفير مختلف (حالياً يحفظ كـ uint8).

## التدريب (Train)

السكربت `scripts/train_smp.py` يقوم بالتالي باختصار:

- يقرأ مجموعة البيانات عبر `SegmentationDataset`.
- يعيد تحديد عدد الفئات أوتوماتيكيًا عبر فحص أقنعة التدريب (أكبر قيمة + 1).
- ينشئ نموذج `Unet` مع `resnet34` encoder محملًا بأوزان ImageNet.
- يستخدم `CrossEntropyLoss` وAdam.

مثال لتشغيل التدريب:

```powershell
python scripts/train_smp.py --train-images train --train-masks train_masks --val-images valid --val-masks valid_masks --epochs 20 --batch-size 4
```

ملاحظات:

- الدقة/حجم الدُفعات محدد صغيرًا مبدئيًا (batch-size=4). عدّل حسب الذاكرة/GPU لديك.
- حجم الإدخال ثابت إلى 512x512 داخل السكربت؛ عدّل في `get_transforms()` إن رغبت.

## الاستدلال (Inference)

1. قناع وحفظ overlay (سكربت بسيط):

```powershell
python scripts/infer_smp.py --image path\to\image.jpg --out out_mask.png --num-classes 20
```

ينتج ملف قناع PNG وملف overlay ملون (اسم `out_mask_overlay.png`).

2. استدلال مع رسم حدود ومؤشرات ثقة:

```powershell
python scripts/infer_smp_viz.py --image path\to\image.jpg --out result.png --confidence 0.5
```

ينتج: قناع، overlay، وصورة مع مربعات وتصنيفات محفوظة في الملفات المناسبة.

## التقييم (Evaluation)

لتقييم النموذج على مجموعة `valid` بعد تحويل الأقنعة:

```powershell
python scripts/evaluate_model.py --data-dir valid --num-classes 20 --output-dir evaluation_results
```

المخرجات:

- `evaluation_results/evaluation_results.txt` يحتوي مقاييس IoU لكل فئة والنتائج الإجمالية.
- `evaluation_results/confusion_matrix.png` مصفوفة الالتباس.

## توصيات لتحسين الأداء

- استخدم augmentations أقوى في `get_transforms()` (albumentations) مثل Flip, Rotate, RandomBrightnessContrast.
- زِد الدقة (مثل 768x768) إن كان لديك GPU بذاكرة كافية.
- جرّب مُشفّرات أقوى كـ `resnet50`, `efficientnet`، أو استخدام نماذج مثل `Unet++`, `DeepLabV3` حسب الحاجة.
- استخدم تقنيات تعلم متقدم: LR scheduler, mixed precision (amp), زيادة الدفعة عبر Gradient Accumulation إذا الذاكرة محدودة.

## نقاط شائعة ومشاكل معروفة

- `pycocotools` يسبب مشاكل عند التثبيت على Windows; الحلول: تثبيت `pycocotools-windows` أو تشغيل داخل WSL/Colab.
- حالياً الأقنعة محفوظة بصيغة uint8؛ إذا كان عدد الفئات >=256 فستحتاج لتمثيل 16-bit أو صيغة مختلفة.
- السكربت `mian.py` في الجذر فارغ (قد يكون خطأ طفيف في الاسم `mian` بدل `main`).

## وصف تقني مختصر (للمهندسين)

- نوع النموذج: Unet (encoder=resnet34)
- مدخل النموذج: RGB صورة -> يتم تغيير الحجم داخليًا إلى 512x512
- مخرج النموذج: logits بعدد القنوات = عدد الفئات -> تستخدم `torch.argmax` للحصول على قناع الفئة النهائية
- خسارة: CrossEntropy
- حفظ أفضل نموذج بناءً على أقل قيمة فقد على مجموعة التحقق (validation)

## اقتراحات مستقبلية (تحسينات بسيطة)

- إضافة ملف `setup.py` أو `pyproject.toml` لتثبيت حزمة محلية.
- إضافة سكربت إدارة التجارب (مثلاً باستخدام `hydra` أو `argparse` محسن).
- إضافة اختبارات وحدة بسيطة للتأكد من أن `convert_coco_to_masks.py` ينتج أقنعة متوقعة.

## الترخيص والتواصل

أضف ملف LICENSE إن رغبت. إن أردت مني تحويل README الحالي إلى العربية داخل `README.md` مباشرة أو إضافة تعديلات أكبر (مثلاً أمثلة تشغيلية أو تحقيق تجريبي هنا)، أكتب لي وسأتابع.

---

إذا تريد، أستطيع الآن:

- توليد نسخة إنجليزية محسنة أيضًا.
- إصلاح `mian.py` أو إزالة الملف إن كان تافهًا.
- إنشاء سكربت صغير لتشغيل عملية تحويل+تدريب تجريبية مع تقارير مُصغّرة.

أخبرني أي تعديل إضافي تريده، أو أريدك أن أضع هذا الملف ضمن المشروع الآن؟
