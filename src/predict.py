"""
Inference module: generates Kaggle submission CSV via Ensembling (5-Fold) + TTA.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.dataset import SargazoTestDataset, get_tta_transforms
from src.model import create_model


def load_fold_models(device):
    """Load all 5 fold models."""
    models = []
    for fold in range(1, config.NUM_FOLDS + 1):
        ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"best_model_fold_{fold}.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing fold {fold} checkpoint: {ckpt_path}")

        model = create_model(num_classes=config.NUM_CLASSES, pretrained=False)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()
        models.append(model)
        print(f"[Predict] Loaded fold {fold} model (Val F1: {checkpoint['val_f1']:.4f})")
    
    return models


@torch.no_grad()
def predict_ensemble_tta(models, device):
    """
    Run inference using Ensemble of 5 Folds + Test-Time Augmentation.
    """
    tta_transforms = get_tta_transforms()
    n_tta = len(tta_transforms)
    n_models = len(models)
    
    test_df = pd.read_csv(config.TEST_CSV)
    image_names = test_df["image_name"].tolist()
    n_images = len(image_names)

    # Accumulate probabilities across (Models x TTA)
    all_probs = np.zeros((n_images, config.NUM_CLASSES), dtype=np.float64)

    for fold, model in enumerate(models, 1):
        print(f"\n[Predict] Processing Fold {fold} / {n_models}")
        for tta_idx, transform in enumerate(tta_transforms):
            test_dataset = SargazoTestDataset(config.TEST_CSV, config.IMAGES_DIR, transform=transform)
            test_loader = DataLoader(
                test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
            )

            name_to_idx = {name: idx for idx, name in enumerate(image_names)}
            
            pbar = tqdm(test_loader, desc=f"  TTA pass {tta_idx + 1}/{n_tta}", leave=False)
            for images, names in pbar:
                images = images.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda"):
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()

                for prob, name in zip(probs, names):
                    idx = name_to_idx[name]
                    all_probs[idx] += prob

    # Average across all models and TTA passes
    all_probs /= (n_models * n_tta)
    predictions = np.argmax(all_probs, axis=1)

    return image_names, predictions, all_probs


def predict():
    """Generate Kaggle submission CSV with 5-Fold Ensemble + TTA predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  SargaNet — Ensemble (5-Fold) + TTA Prediction")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    models = load_fold_models(device)

    image_names, predictions, probs = predict_ensemble_tta(models, device)

    # Map back to string labels
    predicted_labels = [config.IDX_TO_LABEL[p] for p in predictions]

    submission = pd.DataFrame({"image_name": image_names, "label": predicted_labels})
    submission_path = os.path.join(config.OUTPUT_DIR, "submission_ensemble.csv")
    submission.to_csv(submission_path, index=False)

    print(f"\n[Predict] Submission saved to {submission_path}")
    print(f"[Predict] Total test images: {len(submission)}")
    
    print(f"\n[Predict] Predicted class distribution:")
    for cls_name in config.CLASS_NAMES:
        count = (submission["label"] == cls_name).sum()
        pct = 100.0 * count / len(submission)
        bar = "█" * int(pct / 2)
        print(f"  {cls_name:12s}: {count:4d} ({pct:5.1f}%) {bar}")

    prob_df = pd.DataFrame(probs, columns=config.CLASS_NAMES)
    prob_df.insert(0, "image_name", image_names)
    prob_df.to_csv(os.path.join(config.OUTPUT_DIR, "test_probabilities_ensemble.csv"), index=False)


if __name__ == "__main__":
    predict()
