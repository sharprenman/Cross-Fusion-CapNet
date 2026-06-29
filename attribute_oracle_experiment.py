import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_ATTRIBUTES = [
    "subtlety",
    "internal_structure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Oracle upper-bound experiment: predict malignancy class from ground-truth attributes only."
    )
    parser.add_argument("--csv_file", type=str, default="nodule_dataset_normalized.csv")
    parser.add_argument("--save_dir", type=str, default="./models/attribute_oracle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--filter_existing_images", action="store_true", default=True)
    parser.add_argument(
        "--use_norm_columns",
        action="store_true",
        help="Use *_norm columns if present. Default uses the same 8 raw attribute columns as the model dataset.",
    )
    parser.add_argument(
        "--include_attribute_malignancy",
        action="store_true",
        help="Also include attribute_malignancy as a feature. Do not use this for strict 8-concept CBM upper bound.",
    )
    return parser.parse_args()


def load_dataframe(csv_file, filter_existing_images):
    df = pd.read_csv(csv_file)
    if filter_existing_images and "image_path" in df.columns:
        exists_mask = df["image_path"].apply(lambda path: os.path.exists(str(path)))
        missing = int((~exists_mask).sum())
        if missing > 0:
            print(f"Filtered {missing} rows with missing image files.")
        df = df.loc[exists_mask].reset_index(drop=True)
    return df


def get_feature_columns(df, use_norm_columns, include_attribute_malignancy):
    attr_cols = DEFAULT_ATTRIBUTES.copy()
    if include_attribute_malignancy:
        attr_cols.append("attribute_malignancy")

    if use_norm_columns:
        norm_cols = [f"{col}_norm" for col in attr_cols]
        if all(col in df.columns for col in norm_cols):
            return norm_cols
        print("Warning: not all *_norm columns exist; falling back to raw attribute columns.")

    missing_cols = [col for col in attr_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required attribute columns: {missing_cols}")
    return attr_cols


def split_like_training_script(n_samples, train_split, val_split, seed):
    train_size = int(train_split * n_samples)
    val_size = int(val_split * n_samples)
    test_size = n_samples - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_samples, generator=generator).numpy()

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:train_size + val_size + test_size]
    return train_idx, val_idx, test_idx


def evaluate_model(name, model, x_train, y_train, x_val, y_val, x_test, y_test):
    model.fit(x_train, y_train)

    rows = []
    for split_name, x, y in [
        ("val", x_val, y_val),
        ("test", x_test, y_test),
    ]:
        pred = model.predict(x)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(x)[:, 1]
        else:
            score = pred.astype(np.float32)

        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp + 1e-8)

        rows.append({
            "model": name,
            "split": split_name,
            "accuracy": accuracy_score(y, pred),
            "balanced_accuracy": balanced_accuracy_score(y, pred),
            "auc": roc_auc_score(y, score),
            "precision_malignant": precision_score(y, pred, pos_label=1, zero_division=0),
            "recall_malignant": recall_score(y, pred, pos_label=1, zero_division=0),
            "specificity_benign": specificity,
            "f1_malignant": f1_score(y, pred, pos_label=1, zero_division=0),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        })
    return rows


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    df = load_dataframe(args.csv_file, args.filter_existing_images)
    feature_cols = get_feature_columns(df, args.use_norm_columns, args.include_attribute_malignancy)

    if "class" not in df.columns:
        raise ValueError("CSV must contain a 'class' label column.")

    x = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["class"].to_numpy(dtype=np.int64)

    train_idx, val_idx, test_idx = split_like_training_script(
        len(df), args.train_split, args.val_split, args.seed
    )

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    print(f"CSV: {args.csv_file}")
    print(f"Samples: {len(df)}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Class counts: {dict(zip(*np.unique(y, return_counts=True)))}")

    models = {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=args.seed)),
        ]),
        "logistic_regression_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
        ]),
        "random_forest_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=args.seed,
                n_jobs=-1,
                min_samples_leaf=3,
            )),
        ]),
    }

    all_rows = []
    for name, model in models.items():
        all_rows.extend(evaluate_model(name, model, x_train, y_train, x_val, y_val, x_test, y_test))

    results = pd.DataFrame(all_rows)
    results_path = os.path.join(args.save_dir, "attribute_oracle_results.csv")
    results.to_csv(results_path, index=False)

    summary_path = os.path.join(args.save_dir, "attribute_oracle_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"CSV: {args.csv_file}\n")
        f.write(f"Samples: {len(df)}\n")
        f.write(f"Features ({len(feature_cols)}): {feature_cols}\n")
        f.write(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}\n")
        f.write(f"Class counts: {dict(zip(*np.unique(y, return_counts=True)))}\n\n")
        f.write(results.to_string(index=False))
        f.write("\n")

    print("\nOracle results:")
    print(results.to_string(index=False))
    print(f"\nSaved CSV: {results_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
