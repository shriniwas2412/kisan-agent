"""Fine-tune MobileNetV3-Small on PlantDoc and export it to ONNX.

PlantDoc (Singh et al., 2020, CC BY 4.0) has real field photos with cluttered backgrounds,
which makes it much harder than lab-clean PlantVillage and closer to what farmers send.

    python training/train_leaf.py --data ../datasets/PlantDoc-Dataset-master --epochs 15

Outputs: models/leaf_classifier.onnx, models/labels.json, models/metrics.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True  # a few PlantDoc JPEGs are truncated
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
EXTS = {".jpg", ".jpeg", ".png"}


class Leaves(Dataset):
    def __init__(self, items: list[tuple[Path, int]], tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label


def list_images(root: Path, classes: list[str]) -> list[tuple[Path, int]]:
    """Index images by the TRAIN class list, so a class missing from test can't shift labels."""
    idx = {c: i for i, c in enumerate(classes)}
    items = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name in idx:
            items += [(p, idx[d.name]) for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
    return items


def readable(items):
    """Drop files PIL can't open, and report them instead of crashing mid-epoch."""
    ok = []
    for p, y in items:
        try:
            with Image.open(p) as im:
                im.verify()
            ok.append((p, y))
        except Exception:
            print(f"  skipping unreadable image: {p.name}")
    return ok


def evaluate(model, loader, device, n_classes):
    model.eval()
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)
    top3 = total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).cpu()
            pred = logits.argmax(1)
            for t, p in zip(y, pred):
                confusion[t, p] += 1
            top3 += (logits.topk(3, 1).indices == y[:, None]).any(1).sum().item()
            total += len(y)
    top1 = confusion.diag().sum().item() / total
    return top1, top3 / total, confusion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    classes = sorted(d.name for d in (args.data / "train").iterdir() if d.is_dir())
    train_all = readable(list_images(args.data / "train", classes))
    test = readable(list_images(args.data / "test", classes))

    # 10% of train held out for choosing the best epoch; the official test set is touched once, at the end.
    random.shuffle(train_all)
    n_val = len(train_all) // 10
    val, train = train_all[:n_val], train_all[n_val:]
    print(f"{len(classes)} classes | train {len(train)} | val {len(val)} | test {len(test)} | device {device}")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    dl = lambda items, tf, shuffle: DataLoader(Leaves(items, tf), batch_size=args.batch, shuffle=shuffle, num_workers=2)
    train_dl, val_dl, test_dl = dl(train, train_tf, True), dl(val, eval_tf, False), dl(test, eval_tf, False)

    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))
    model.to(device)

    # Class-weighted loss: PlantDoc is imbalanced (some classes have ~3x the photos of others).
    counts = Counter(y for _, y in train)
    weights = torch.tensor([len(train) / (len(classes) * max(counts[i], 1)) for i in range(len(classes))])
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    best_val, best_state, start = -1.0, None, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
        sched.step()
        val_acc, _, _ = evaluate(model, val_dl, device, len(classes))
        print(f"epoch {epoch:2d} | loss {running / len(train):.3f} | val acc {val_acc:.3f}")
        if val_acc > best_val:
            best_val, best_state = val_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    top1, top3, confusion = evaluate(model, test_dl, device, len(classes))
    support = confusion.sum(1)
    per_class = {
        c: {"support": int(support[i]), "recall": round(confusion[i, i].item() / support[i].item(), 3) if support[i] else None}
        for i, c in enumerate(classes)
    }
    metrics = {
        "dataset": "PlantDoc (official test split)",
        "model": "mobilenet_v3_small (ImageNet-pretrained, fine-tuned)",
        "epochs": args.epochs, "seed": args.seed,
        "best_val_acc": round(best_val, 4), "test_top1": round(top1, 4), "test_top3": round(top3, 4),
        "train_minutes": round((time.time() - start) / 60, 1),
        "per_class": per_class, "confusion": confusion.tolist(),
    }
    print(f"TEST top-1 {top1:.3f} | top-3 {top3:.3f}")

    model.eval().cpu()
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), args.out / "leaf_classifier.onnx",
                      input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, dynamo=False)
    (args.out / "labels.json").write_text(json.dumps(classes, indent=1))
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"saved to {args.out}/")


if __name__ == "__main__":
    main()
