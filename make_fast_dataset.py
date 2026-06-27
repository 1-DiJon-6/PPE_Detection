from pathlib import Path
import shutil
import random

# Откуда берём исходные изображения
SOURCE = Path("vest-helmet.yolov8")

# Куда создаём маленький датасет
TARGET = Path("dataset_fast")

# Сколько изображений оставить
LIMITS = {
    "train": 100,
    "valid": 20,
    "test": 20,
}

random.seed(42)

if TARGET.exists():
    shutil.rmtree(TARGET)

for split, limit in LIMITS.items():
    src_images = SOURCE / split / "images"
    src_labels = SOURCE / split / "labels"

    dst_images = TARGET / split / "images"
    dst_labels = TARGET / split / "labels"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    images = (
        list(src_images.glob("*.jpg"))
        + list(src_images.glob("*.jpeg"))
        + list(src_images.glob("*.png"))
        + list(src_images.glob("*.webp"))
    )

    random.shuffle(images)
    selected = images[:limit]

    copied = 0
    skipped = 0

    for img in selected:
        label = src_labels / f"{img.stem}.txt"

        if label.exists():
            shutil.copy2(img, dst_images / img.name)
            shutil.copy2(label, dst_labels / label.name)
            copied += 1
        else:
            skipped += 1

    print(f"{split}: найдено {len(images)}, скопировано {copied}, пропущено {skipped}")

yaml_text = """train: train/images
val: valid/images
test: test/images

nc: 2
names:
  - reflective_jacket
  - safety_helmet
"""

(TARGET / "data.yaml").write_text(yaml_text, encoding="utf-8")

print("Готово: создан dataset_fast")