import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    ssdlite320_mobilenet_v3_large,
    retinanet_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


CLASSES = ["reflective_jacket", "safety_helmet"]
NUM_CLASSES = len(CLASSES) + 1  # + background


class YoloDetectionDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"

        self.images = [
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue

                    class_id, x_center, y_center, box_width, box_height = map(float, parts)

                    x_center *= width
                    y_center *= height
                    box_width *= width
                    box_height *= height

                    x1 = x_center - box_width / 2
                    y1 = y_center - box_height / 2
                    x2 = x_center + box_width / 2
                    y2 = y_center + box_height / 2

                    if x2 <= x1 or y2 <= y1:
                        continue

                    boxes.append([x1, y1, x2, y2])

                    # В torchvision label 0 — background, поэтому классы сдвигаем на +1
                    labels.append(int(class_id) + 1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        image = torchvision.transforms.functional.to_tensor(image)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_model(model_name):
    if model_name == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        return model

    if model_name == "ssdlite":
        model = ssdlite320_mobilenet_v3_large(
            weights=None,
            weights_backbone=None,
            num_classes=NUM_CLASSES,
        )
        return model

    if model_name == "retinanet":
        model = retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=None,
            num_classes=NUM_CLASSES,
        )
        return model

    raise ValueError(f"Неизвестная модель: {model_name}")


def train(model_name, epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    dataset = YoloDetectionDataset("dataset/train")
    dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    )

    model = get_model(model_name)
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005,
    )

    output_dir = Path("runs") / "torchvision" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0

        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in target.items()}
                for target in targets
            ]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.4f}")

    save_path = output_dir / "model.pt"
    torch.save(model.state_dict(), save_path)

    print(f"Готово. Модель сохранена: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["fasterrcnn", "ssdlite", "retinanet"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)

    args = parser.parse_args()

    train(args.model, args.epochs, args.batch)