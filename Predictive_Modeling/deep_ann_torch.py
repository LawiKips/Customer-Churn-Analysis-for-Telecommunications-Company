"""Deep ANN churn classifier with logistic output.

Usage (from repo root after `pip install -r requirements.txt`):
    python Predictive_Modeling/deep_ann_torch.py

Behavior:
- Loads pre-split numeric/encoded data from `Data_Preparation`.
- Standardizes all features, preserves 0/1 for boolean-like columns.
- Splits train set into train/val (80/20) for early stopping.
- Trains a multilayer perceptron (PyTorch) with BCEWithLogitsLoss.
- Automatically uses CUDA > MPS > CPU depending on availability.
- Prints metrics table (accuracy, precision, recall, f1, roc_auc, pr_auc).
- Saves best model and scaler stats; exports ROC/PR/confusion-matrix plots.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data_Preparation"
MODEL_DIR = ROOT / "Predictive_Modeling" / "models"
FIG_DIR = ROOT / "Predictive_Modeling" / "figures"
SEED = 42

# device selection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def load_csv(path: Path) -> np.ndarray:
        # Use numpy genfromtxt to avoid pandas dependency at runtime
        return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)

    X_train_raw = load_csv(DATA_DIR / "X_train_raw.csv")
    X_test_raw = load_csv(DATA_DIR / "X_test_raw.csv")
    y_train_raw = load_csv(DATA_DIR / "y_train.csv")
    y_test_raw = load_csv(DATA_DIR / "y_test.csv")

    def struct_to_ndarray(data) -> np.ndarray:
        # Convert structured array to plain float array, mapping True/False strings to 1/0
        arr = []
        for row in data:
            values = []
            for v in row:
                if isinstance(v, (str, bytes)):
                    if v in ("True", b"True"):
                        values.append(1.0)
                    elif v in ("False", b"False"):
                        values.append(0.0)
                    else:
                        # numeric strings
                        values.append(float(v))
                else:
                    values.append(float(v))
            arr.append(values)
        return np.array(arr, dtype=np.float32)

    X_train = struct_to_ndarray(X_train_raw)
    X_test = struct_to_ndarray(X_test_raw)
    y_train = struct_to_ndarray(y_train_raw).reshape(-1)
    y_test = struct_to_ndarray(y_test_raw).reshape(-1)

    return X_train, y_train, X_test, y_test


class DeepANN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),  # logistic head
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def make_loaders(
    X_train: np.ndarray, y_train: np.ndarray, batch_size: int = 256
) -> Tuple[DataLoader, DataLoader, StandardScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # train/val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    train_ds = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val_s, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, scaler, X_val_s, y_val, X_tr_s, y_tr


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "f2": fbeta_score(y_true, preds, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
        "report": classification_report(y_true, preds, digits=3),
    }


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str = "f2") -> Tuple[float, Dict[str, float]]:
    best_thr, best_val, best_metrics = 0.5, -1, {}
    for thr in np.linspace(0.1, 0.9, 41):  # step 0.02
        m = compute_metrics(y_true, probs, threshold=thr)
        val = m[metric]
        if val > best_val:
            best_val = val
            best_thr = thr
            best_metrics = m
    return best_thr, best_metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 120,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    patience: int = 12,
    pos_weight: float | None = None,
) -> Dict[str, float]:
    model.to(DEVICE)
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = None
    best_auc = -np.inf
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # validation
        model.eval()
        with torch.no_grad():
            logits_all = []
            y_all = []
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                logits_all.append(logits.cpu())
                y_all.append(yb)
            logits_all = torch.cat(logits_all)
            y_all = torch.cat(y_all)
            probs = torch.sigmoid(logits_all).numpy()
            auc = roc_auc_score(y_all.numpy(), probs)

        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return {"val_roc_auc": best_auc}


def main() -> None:
    set_seeds(SEED)
    X_train, y_train, X_test, y_test = load_data()

    train_loader, val_loader, scaler, X_val_s, y_val, X_tr_s, y_tr = make_loaders(X_train, y_train)
    X_test_s = scaler.transform(X_test)

    model = DeepANN(input_dim=X_train.shape[1])
    # class imbalance handling via pos_weight (neg/pos)
    pos_weight = float((y_tr == 0).sum() / (y_tr == 1).sum())
    train_model(model, train_loader, val_loader, pos_weight=pos_weight)

    model.eval()
    with torch.no_grad():
        train_probs = torch.sigmoid(model(torch.tensor(X_tr_s, dtype=torch.float32, device=DEVICE))).cpu().numpy()
        val_probs = torch.sigmoid(model(torch.tensor(X_val_s, dtype=torch.float32, device=DEVICE))).cpu().numpy()
        test_probs = torch.sigmoid(model(torch.tensor(X_test_s, dtype=torch.float32, device=DEVICE))).cpu().numpy()

    metrics = {
        "train": compute_metrics(y_tr, train_probs),
        "val": compute_metrics(y_val, val_probs),
        "test": compute_metrics(y_test, test_probs),
        "device": str(DEVICE),
        "pos_weight": pos_weight,
    }

    # Best threshold by maximizing F2 on validation
    best_thr, best_val_metrics = find_best_threshold(y_val, val_probs, metric="f2")
    best_test_metrics = compute_metrics(y_test, test_probs, threshold=best_thr)

    # Threshold sweep to surface higher-recall operating points
    sweep = []
    for thr in np.linspace(0.2, 0.5, 7):  # 0.20,0.25,...,0.50
        m = compute_metrics(y_test, test_probs, threshold=thr)
        sweep.append((thr, m["precision"], m["recall"], m["f1"], m["f2"]))

    # Pretty print table
    def fmt_row(split: str, m: Dict[str, float]) -> str:
        return (
            f"{split:<6} | acc {m['accuracy']:.3f} | prec {m['precision']:.3f} | "
            f"rec {m['recall']:.3f} | f1 {m['f1']:.3f} | f2 {m['f2']:.3f} | "
            f"roc_auc {m['roc_auc']:.3f} | pr_auc {m['pr_auc']:.3f}"
        )

    for split in ["train", "val", "test"]:
        print(fmt_row(split, metrics[split]))
    print("\nTest classification report:")
    print(metrics["test"]["report"])

    print(
        "Metric notes: recall/PR-AUC/F2 matter most to reduce missed churn; precision ties to outreach cost; "
        "ROC-AUC shows separability; PR-AUC is area of precision vs recall; F2 weights recall higher than precision."
    )
    print("\nTest threshold sweep (thr | prec | recall | f1 | f2):")
    for thr, p, r, f1, f2 in sweep:
        print(f"{thr:0.2f} | {p:0.3f} | {r:0.3f} | {f1:0.3f} | {f2:0.3f}")
    print(f"\nAuto-selected threshold (val F2 max): thr={best_thr:0.2f}")
    print(
        f"val*   | acc {best_val_metrics['accuracy']:.3f} | prec {best_val_metrics['precision']:.3f} | "
        f"rec {best_val_metrics['recall']:.3f} | f1 {best_val_metrics['f1']:.3f} | "
        f"f2 {best_val_metrics['f2']:.3f} | roc_auc {best_val_metrics['roc_auc']:.3f} | pr_auc {best_val_metrics['pr_auc']:.3f}"
    )
    print(
        f"test*  | acc {best_test_metrics['accuracy']:.3f} | prec {best_test_metrics['precision']:.3f} | "
        f"rec {best_test_metrics['recall']:.3f} | f1 {best_test_metrics['f1']:.3f} | "
        f"f2 {best_test_metrics['f2']:.3f} | roc_auc {best_test_metrics['roc_auc']:.3f} | pr_auc {best_test_metrics['pr_auc']:.3f}"
    )
    print(f"\npos_weight used: {metrics['pos_weight']:.3f}")
    print(f"device  | {metrics['device']}")

    # Plots
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay

    RocCurveDisplay.from_predictions(y_test, test_probs)
    plt.title("ROC Curve (test)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "deep_ann_roc.png", dpi=300)
    plt.close()

    PrecisionRecallDisplay.from_predictions(y_test, test_probs)
    plt.title("Precision-Recall Curve (test)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "deep_ann_pr.png", dpi=300)
    plt.close()

    best_preds = (test_probs >= best_thr).astype(int)
    ConfusionMatrixDisplay.from_predictions(y_test, best_preds, cmap="Blues")
    plt.title(f"Confusion Matrix (test, thr={best_thr:0.2f})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "deep_ann_confusion.png", dpi=300)
    plt.close()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_},
        MODEL_DIR / "deep_ann_torch.pt",
    )


if __name__ == "__main__":
    main()
