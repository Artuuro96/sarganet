"""
Evaluation module for K-Fold: computes Out-Of-Fold (OOF) predictions
by aggregating validation predictions from all 5 folds.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.dataset import SargazoDataset, get_val_transforms
from src.model import create_model


def load_fold_model(fold, device):
    """Load the best model checkpoint for a specific fold."""
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"best_model_fold_{fold}.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    model = create_model(num_classes=config.NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"[Eval] Loaded fold {fold} checkpoint (epoch {checkpoint['epoch']}, "
          f"Val F1: {checkpoint['val_f1']:.4f})")

    return model


@torch.no_grad()
def get_oof_predictions(device):
    """
    Compute Out-Of-Fold (OOF) predictions for the entire training set.
    """
    df = pd.read_csv(config.LABELS_CSV)
    skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED)
    
    oof_preds = np.zeros(len(df), dtype=int)
    oof_probs = np.zeros((len(df), config.NUM_CLASSES), dtype=np.float64)
    oof_labels = np.zeros(len(df), dtype=int)
    
    for fold, (_, val_idx) in enumerate(skf.split(df, df["label"]), 1):
        print(f"\n  Evaluating Fold {fold}...")
        model = load_fold_model(fold, device)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        val_dataset = SargazoDataset(val_df, config.IMAGES_DIR, transform=get_val_transforms())
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
        )
        
        fold_probs = []
        for images, _ in val_loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda"):
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
            fold_probs.extend(probs.cpu().numpy())
            
        fold_probs = np.array(fold_probs)
        fold_preds = np.argmax(fold_probs, axis=1)
        
        oof_probs[val_idx] = fold_probs
        oof_preds[val_idx] = fold_preds
        oof_labels[val_idx] = val_dataset.get_labels()
        
    return oof_preds, oof_labels, oof_probs


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], cbar_kws={"label": "Count"},
    )
    axes[0].set_title("OOF Confusion Matrix (Counts)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("True", fontsize=12)

    sns.heatmap(
        cm_pct, annot=True, fmt=".1f", cmap="Oranges",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], cbar_kws={"label": "Percentage (%)"},
    )
    axes[1].set_title("OOF Confusion Matrix (% per class)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Predicted", fontsize=12)
    axes[1].set_ylabel("True", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Eval] Confusion matrix saved to {save_path}")


def evaluate():
    """Full K-Fold OOF evaluation pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SargaNet — K-Fold OOF Evaluation")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    preds, labels, probs = get_oof_predictions(device)

    # ─── Metrics ─────────────────────────────────────────────────────────
    acc = accuracy_score(labels, preds) * 100
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")

    print(f"\n{'='*60}")
    print(f"[OOF Eval] Overall Accuracy: {acc:.2f}%")
    print(f"[OOF Eval] Macro F1:        {f1_macro:.4f}")
    print(f"[OOF Eval] Weighted F1:     {f1_weighted:.4f}")
    print(f"{'='*60}")

    print(f"\n[OOF Eval] Classification Report:")
    report = classification_report(
        labels, preds, target_names=config.CLASS_NAMES, digits=4
    )
    print(report)

    # Save report
    report_path = os.path.join(config.OUTPUT_DIR, "oof_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"OOF Overall Accuracy: {acc:.2f}%\n")
        f.write(f"OOF Macro F1: {f1_macro:.4f}\n")
        f.write(f"OOF Weighted F1: {f1_weighted:.4f}\n\n")
        f.write(report)

    # Confusion matrix
    cm_path = os.path.join(config.OUTPUT_DIR, "oof_confusion_matrix.png")
    plot_confusion_matrix(labels, preds, config.CLASS_NAMES, cm_path)


if __name__ == "__main__":
    evaluate()
