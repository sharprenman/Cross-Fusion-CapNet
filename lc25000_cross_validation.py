import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from hifuse_model import HiFuse_Base, HiFuse_Mini, HiFuse_Small, HiFuse_Tiny


class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        image_path, label = self.dataset.samples[self.indices[item]]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label


class HiFuseClassifier(nn.Module):
    def __init__(self, model_type, num_classes, patch_size, window_size):
        super().__init__()
        if model_type == "mini":
            self.backbone = HiFuse_Mini(num_classes=num_classes, patch_size=patch_size, window_size=window_size)
        elif model_type == "tiny":
            self.backbone = HiFuse_Tiny(num_classes=num_classes, patch_size=patch_size, window_size=window_size)
        elif model_type == "small":
            self.backbone = HiFuse_Small(num_classes=num_classes, patch_size=patch_size, window_size=window_size)
        elif model_type == "base":
            self.backbone = HiFuse_Base(num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        feature_dim = self.backbone.conv_norm.normalized_shape[0]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.pool(features).flatten(1)
        return self.head(pooled)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_transforms(image_size):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def collect_targets(dataset):
    return np.array([label for _, label in dataset.samples], dtype=np.int64)


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            total_loss += loss.item()
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    labels = np.array(all_labels, dtype=np.int64)
    preds = np.array(all_preds, dtype=np.int64)
    probs = np.array(all_probs, dtype=np.float64)

    metrics = {
        "loss": total_loss / max(len(loader), 1),
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except ValueError:
        metrics["auc"] = float("nan")

    metrics["labels"] = labels
    metrics["preds"] = preds
    metrics["probs"] = probs
    metrics["confusion_matrix"] = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    return metrics


def train_one_fold(args, fold, train_dataset, val_dataset, test_dataset, class_names, device):
    model = HiFuseClassifier(
        model_type=args.model_type,
        num_classes=len(class_names),
        patch_size=args.patch_size,
        window_size=args.window_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.lr_patience
    )

    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers) if test_dataset else None

    fold_dir = Path(args.output_dir) / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = -1.0
    best_path = fold_dir / "best_model.pth"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []
        progress = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item()
            all_train_labels.extend(labels.cpu().numpy().tolist())
            all_train_preds.extend(preds.cpu().numpy().tolist())
            progress.set_postfix(loss=running_loss / max(len(all_train_labels) // args.batch_size, 1))

        train_loss = running_loss / max(len(train_loader), 1)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        val_metrics = evaluate(model, val_loader, device, len(class_names))
        scheduler.step(val_metrics["loss"])

        row = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
        }
        history.append(row)
        print(
            f"Fold {fold} Epoch {epoch}: "
            f"train_acc={train_acc:.4f}, val_acc={val_metrics['accuracy']:.4f}, "
            f"val_f1={val_metrics['f1']:.4f}, val_auc={val_metrics['auc']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "args": vars(args),
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                    "val_auc": val_metrics["auc"],
                },
                best_path,
            )

    write_csv(fold_dir / "history.csv", history)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics = evaluate(model, val_loader, device, len(class_names))
    save_confusion_matrix(fold_dir / "val_confusion_matrix.csv", val_metrics["confusion_matrix"], class_names)
    np.save(fold_dir / "val_labels.npy", val_metrics["labels"])
    np.save(fold_dir / "val_preds.npy", val_metrics["preds"])
    np.save(fold_dir / "val_probs.npy", val_metrics["probs"])

    result = {
        "fold": fold,
        "best_epoch": checkpoint["epoch"],
        "val_accuracy": val_metrics["accuracy"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_f1": val_metrics["f1"],
        "val_auc": val_metrics["auc"],
    }

    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, device, len(class_names))
        save_confusion_matrix(fold_dir / "test_confusion_matrix.csv", test_metrics["confusion_matrix"], class_names)
        result.update(
            {
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_auc": test_metrics["auc"],
            }
        )
    else:
        test_metrics = None

    with open(fold_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    test_cm = test_metrics["confusion_matrix"] if test_metrics is not None else None
    return result, val_metrics["confusion_matrix"], test_cm


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(path, matrix, class_names):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred"] + class_names)
        for name, row in zip(class_names, matrix):
            writer.writerow([name] + row.tolist())


def summarize_results(output_dir, rows, aggregate_cm, class_names, aggregate_test_cm=None):
    output_dir = Path(output_dir)
    write_csv(output_dir / "fold_metrics.csv", rows)
    save_confusion_matrix(output_dir / "crossval_confusion_matrix.csv", aggregate_cm, class_names)
    if aggregate_test_cm is not None:
        save_confusion_matrix(output_dir / "test_confusion_matrix_sum.csv", aggregate_test_cm, class_names)

    summary = {}
    numeric_keys = [key for key in rows[0].keys() if key != "fold"]
    for key in numeric_keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.nanmean(values))
        summary[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nCross-validation summary")
    for key in numeric_keys:
        print(f"{key}: {summary[f'{key}_mean']:.4f} +/- {summary[f'{key}_std']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="5-fold cross-validation for LC25000 with HiFuse.")
    parser.add_argument("--train_val_dir", type=str, required=True, help="Path to LC25000 Train and Validation Set.")
    parser.add_argument("--test_dir", type=str, default=None, help="Optional independent LC25000 Test Set path.")
    parser.add_argument("--output_dir", type=str, default="lc25000_cv_results", help="Output directory.")
    parser.add_argument("--model_type", choices=["mini", "tiny", "small", "base"], default="small")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=2)
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    train_transform, eval_transform = build_transforms(args.image_size)
    base_dataset = ImageFolder(args.train_val_dir)
    class_names = base_dataset.classes
    labels = collect_targets(base_dataset)
    num_classes = len(class_names)

    class_counts = {class_names[i]: int((labels == i).sum()) for i in range(num_classes)}
    with open(output_dir / "class_mapping.json", "w") as f:
        json.dump({"class_to_idx": base_dataset.class_to_idx, "class_counts": class_counts}, f, indent=2)
    print(f"Class mapping: {base_dataset.class_to_idx}")
    print(f"Train/validation class counts: {class_counts}")

    test_dataset = None
    if args.test_dir:
        test_base = ImageFolder(args.test_dir)
        if test_base.class_to_idx != base_dataset.class_to_idx:
            raise ValueError(f"Test class mapping differs: {test_base.class_to_idx} vs {base_dataset.class_to_idx}")
        test_dataset = TransformSubset(test_base, range(len(test_base)), eval_transform)

    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows = []
    aggregate_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    aggregate_test_cm = np.zeros((num_classes, num_classes), dtype=np.int64) if test_dataset is not None else None

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        train_dataset = TransformSubset(base_dataset, train_idx, train_transform)
        val_dataset = TransformSubset(base_dataset, val_idx, eval_transform)
        result, fold_cm, test_cm = train_one_fold(
            args, fold, train_dataset, val_dataset, test_dataset, class_names, device
        )
        rows.append(result)
        aggregate_cm += fold_cm
        if aggregate_test_cm is not None and test_cm is not None:
            aggregate_test_cm += test_cm

    summarize_results(output_dir, rows, aggregate_cm, class_names, aggregate_test_cm)


if __name__ == "__main__":
    main()
