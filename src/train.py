"""
SargaNet 5-Fold Cross Validation Training Pipeline with EfficientNetV2-S.
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.dataset import SargazoDataset, get_train_transforms, get_val_transforms
from src.model import (
    create_model,
    freeze_backbone,
    unfreeze_backbone,
    get_parameter_groups,
)


def set_seed(seed=42):
    """Make training reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(labels):
    """Compute normalized inverse class frequencies."""
    counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    weights = 1.0 / counts
    weights = weights / weights.sum() * config.NUM_CLASSES
    return torch.FloatTensor(weights)


def rand_bbox(size, lam):
    """Generate a random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_one_epoch(
    model, loader, criterion, optimizer, scaler, scheduler, device, epoch
):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"  Train Epoch {epoch}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Apply MixUp (33%), CutMix (33%), or Normal (33%) for strong regularization
        r = np.random.rand()
        if r < 0.33: # MixUp
            lam = np.random.beta(0.2, 0.2)
            index = torch.randperm(images.size(0)).to(device)
            images = lam * images + (1 - lam) * images[index, :]
            labels_a, labels_b = labels, labels[index]

            with torch.autocast(device_type="cuda"):
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                
            track_labels = labels_a if lam > 0.5 else labels_b
            
        elif r < 0.66: # CutMix
            lam = np.random.beta(1.0, 1.0)
            index = torch.randperm(images.size(0)).to(device)
            labels_a, labels_b = labels, labels[index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            
            # Apply CutMix
            images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
            
            # Adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
            
            with torch.autocast(device_type="cuda"):
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                
            track_labels = labels_a if lam > 0.5 else labels_b
            
        else: # Normal
            with torch.autocast(device_type="cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            track_labels = labels

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=config.GRADIENT_CLIP_MAX_NORM
        )

        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(track_labels).sum().item()
        total += track_labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(track_labels.cpu().numpy())

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / total:.1f}%",
        )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    """Validate the model on the validation set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"  Val   Epoch {epoch}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")

    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance, with optional label smoothing.
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction='none'
        )
        pt = torch.exp(-F.cross_entropy(inputs, targets, reduction='none'))
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_fold(fold, train_df, val_df, device):
    """Train a single fold."""
    print(f"\n{'='*60}")
    print(f"  FOLD {fold}/{config.NUM_FOLDS}")
    print(f"{'='*60}\n")
    
    # ─── Create datasets ─────────────────────────────────────────────────
    train_dataset = SargazoDataset(train_df, config.IMAGES_DIR, transform=get_train_transforms())
    val_dataset = SargazoDataset(val_df, config.IMAGES_DIR, transform=get_val_transforms())

    # ─── Class weights and loader ────────────────────────────────────────
    train_labels = train_dataset.get_labels()
    class_weights = compute_class_weights(train_labels).to(device)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )

    # ─── Model ───────────────────────────────────────────────────────────
    model = create_model(num_classes=config.NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # ─── Loss function ───────────────────────────────────────────────────
    # Using class_weights to handle class imbalance
    criterion = FocalLoss(weight=class_weights, gamma=2.0, label_smoothing=0.1)
    scaler = GradScaler()
    
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    global_epoch = 0
    history = {
        "train_loss": [], "train_acc": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_f1": [],
        "lr": [], "phase": [],
    }
    
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"best_model_fold_{fold}.pth")

    # ───────────────────────────────────────────────────────────────────────
    # Phase 1: Warmup (Head only)
    # ───────────────────────────────────────────────────────────────────────
    print(f"  PHASE 1: Head-only training ({config.WARMUP_EPOCHS} epochs)")
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.WARMUP_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.WARMUP_EPOCHS * len(train_loader)
    )

    for epoch in range(1, config.WARMUP_EPOCHS + 1):
        global_epoch += 1
        curr_lr = optimizer.param_groups[0]["lr"]

        t_loss, t_acc, t_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, device, global_epoch
        )
        v_loss, v_acc, v_f1, _, _ = validate(
            model, val_loader, criterion, device, global_epoch
        )

        history["train_loss"].append(t_loss); history["train_acc"].append(t_acc); history["train_f1"].append(t_f1)
        history["val_loss"].append(v_loss); history["val_acc"].append(v_acc); history["val_f1"].append(v_f1)
        history["lr"].append(curr_lr); history["phase"].append(1)

        print(f"  Epoch {global_epoch:2d} | Train Loss: {t_loss:.4f} Acc: {t_acc:.1f}% F1: {t_f1:.3f} | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.1f}% F1: {v_f1:.3f} | LR: {curr_lr:.2e}")

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_epoch = global_epoch
            torch.save({"epoch": global_epoch, "model_state_dict": model.state_dict(),
                        "val_f1": v_f1, "val_acc": v_acc}, ckpt_path)
            print(f"  ★ New best model saved! (F1: {v_f1:.4f})")

    # ───────────────────────────────────────────────────────────────────────
    # Phase 2: Full Fine-tuning
    # ───────────────────────────────────────────────────────────────────────
    print(f"\n  PHASE 2: Full fine-tuning ({config.FINETUNE_EPOCHS} epochs)")
    unfreeze_backbone(model)
    param_groups = get_parameter_groups(model, config.FINETUNE_LR, config.WARMUP_LR)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5 * len(train_loader), T_mult=2
    )

    for epoch in range(1, config.FINETUNE_EPOCHS + 1):
        global_epoch += 1
        curr_lr = optimizer.param_groups[0]["lr"]

        t_loss, t_acc, t_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, device, global_epoch
        )
        v_loss, v_acc, v_f1, _, _ = validate(
            model, val_loader, criterion, device, global_epoch
        )

        history["train_loss"].append(t_loss); history["train_acc"].append(t_acc); history["train_f1"].append(t_f1)
        history["val_loss"].append(v_loss); history["val_acc"].append(v_acc); history["val_f1"].append(v_f1)
        history["lr"].append(curr_lr); history["phase"].append(2)

        print(f"  Epoch {global_epoch:2d} | Train Loss: {t_loss:.4f} Acc: {t_acc:.1f}% F1: {t_f1:.3f} | "
              f"Val Loss: {v_loss:.4f} Acc: {v_acc:.1f}% F1: {v_f1:.3f} | LR: {curr_lr:.2e}")

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_epoch = global_epoch
            patience_counter = 0
            torch.save({"epoch": global_epoch, "model_state_dict": model.state_dict(),
                        "val_f1": v_f1, "val_acc": v_acc}, ckpt_path)
            print(f"  ★ New best model saved! (F1: {v_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping triggered at epoch {global_epoch}")
                break

    print(f"\n  Fold {fold} Complete! Best Val F1: {best_val_f1:.4f} at epoch {best_epoch}")
    return history


def train():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SargaNet — EfficientNetV2-S {config.NUM_FOLDS}-Fold CV")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    df = pd.read_csv(config.LABELS_CSV)
    
    skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED)
    all_histories = {}
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["label"]), 1):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        history = train_fold(fold, train_df, val_df, device)
        all_histories[f"fold_{fold}"] = history
        
    hist_path = os.path.join(config.OUTPUT_DIR, "kfold_history.json")
    with open(hist_path, "w") as f:
        json.dump(all_histories, f, indent=2)
        
    print(f"\n[Train] 5-Fold Training Complete! Models saved in {config.CHECKPOINT_DIR}")


if __name__ == "__main__":
    train()
