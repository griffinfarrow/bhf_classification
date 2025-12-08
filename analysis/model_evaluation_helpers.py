import os
import sys
import random
import warnings
import math

# Third-party numerical and data handling
import numpy as np
import pandas as pd
import h5py
import cv2
from PIL import Image

# Visualization
import matplotlib.pyplot as plt
from tqdm import tqdm

# Machine learning utilities
from sklearn.metrics import (
    average_precision_score,
    label_ranking_average_precision_score,
    roc_auc_score,
    brier_score_loss,
    f1_score, 
    precision_score, 
    recall_score
)

# PyTorch core
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler

# PyTorch vision
from torchvision import models

# Albumentations
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2



def check_device():
    """
    Check available compute devices and return the best one.
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ CUDA available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ MPS (Apple Silicon GPU) available")
    else:
        device = torch.device("cpu")
        print("✗ Using CPU (no GPU acceleration available)")
    
    print(f"\nSelected device: {device}")
    return device


class Head(nn.Module):
    def __init__(self, in_features, hidden_layer, dropout_rate=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_layer),
            nn.BatchNorm1d(hidden_layer),  
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_layer, hidden_layer // 2), 
            nn.BatchNorm1d(hidden_layer // 2),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_layer // 2, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

class MultiHeadEfficientNet(nn.Module):
    def __init__(self, num_conditions=5, hidden_dim=512, dropout_rate=0.3, model="convnext"):
        super().__init__()
        
        if model=="convnext":
            backbone = models.convnext_base(weights="IMAGENET1K_V1", progress=True)
            in_features = backbone.classifier[2].in_features
        elif model=="efficientnet":
            backbone = models.efficientnet_v2_s(weights="DEFAULT")
            in_features = backbone.classifier[1].in_features
        else: 
            raise Exception(f"Model {model} not recognised ") 
        assert isinstance(in_features, int), f"in_features should be int, got {type(in_features)}"
        
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        
        self.shared_feature_processor = nn.Sequential(
            nn.Linear(in_features, in_features), 
            nn.BatchNorm1d(in_features),
            nn.GELU(), 
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate)
        )
        
        self.heads = nn.ModuleList([
            Head(hidden_dim, hidden_dim // 2, dropout_rate) 
            for _ in range(num_conditions)
        ])
        
    def forward(self, x):
        backbone_feats = self.backbone(x).flatten(1) # squeeze non-batch dimension
        processed_feats = self.shared_feature_processor(backbone_feats)
        outputs = [head(processed_feats) for head in self.heads]
        return torch.cat(outputs, dim=1)  # [batch, num_conditions]

def read_preprocessed_images(filepath):

    with h5py.File(filepath, 'r') as f:
        raw_names = f["img_names"][:]
        images = f["images"][:]
    image_names = [n.decode("utf-8") for n in raw_names]
    image_set = dict(zip(image_names, images))

    return image_set

class ECGDataset(Dataset):
    def __init__(
        self,
        image_names,  # list of filenames
        image_set,    # dict: {filename: np.array(H,W)}
        labels_df,    # dataframe indexed by image_id or image_name
        transforms=None
    ):
        assert len(image_names) == len(labels_df), \
        "Mismatch between number of labels and number of images"
        self.image_names = list(image_names)
        self.images = image_set
        self.labels = torch.tensor(labels_df.values, dtype=torch.float32)
        self.transforms = transforms

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = self.images[image_name]
        # Duplicate grayscale to 3 channels: (H, W, 3)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        if self.transforms is not None:
            image_tens = self.transforms(image=image)["image"]
        else:
            image_tens = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        label = self.labels[idx]
        return image_tens, label

val_transforms = A.Compose([
    A.CLAHE(
        clip_limit=4,
        tile_grid_size=(8, 8),
        p=1.0
    ),
    # Normalization (ImageNet)
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    # Convert to tensor
    ToTensorV2()
])

def compute_ece(y_true, y_pred, n_bins=15):
    """
    Expected Calibration Error for binary predictions.
    y_true, y_pred are 1D arrays for one label.
    """
    if len(np.unique(y_true)) < 2:
        return float('nan')  # no ECE if only one class
    
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_pred >= lo) & (y_pred < hi)
        if np.sum(mask) == 0:
            continue
        
        bin_accuracy = np.mean(y_true[mask])
        bin_confidence = np.mean(y_pred[mask])
        ece += (np.sum(mask) / N) * abs(bin_accuracy - bin_confidence)
    
    return float(ece)

def compute_ranking_metrics(all_y_true, all_y_pred):
    """
    Compute AP/AUROC/LRAP ranking metrics and calibration metrics.
    Inputs are numpy arrays.
    """
    L = all_y_true.shape[1]

    per_label_ap = []
    per_label_auroc = []
    per_label_brier = []
    per_label_brier_baseline = []
    thresholds = []
    per_label_f1s = []
    per_label_ece = []
    per_label_precision = []
    per_label_recall = []
    
    for j in range(L):
        
        y_true = all_y_true[:, j]
        y_pred = all_y_pred[:, j]

        unique_classes = np.unique(y_true)
        has_both_classes = (len(unique_classes) == 2)
        has_any_positive = (np.sum(y_true) > 0)

        # ----- Average Precision -----
        if has_any_positive:
            ap = average_precision_score(y_true, y_pred)
        else:
            ap = float('nan') 
        per_label_ap.append(float(ap))

        # ----- AUROC -----
        if has_both_classes:
            auroc = roc_auc_score(y_true, y_pred)
        else:
            auroc = float('nan')
        per_label_auroc.append(float(auroc))

        # ----- Brier -----
        br = brier_score_loss(y_true, y_pred)
        per_label_brier.append(float(br))
        p = np.mean(y_true)
        brier_baseline = p * (1 - p)
        per_label_brier_baseline.append(float(brier_baseline))
        
        # ----- F1 (threshold sweep) -----
        best_f1 = 0
        best_t = 0.5
        best_p = 0
        best_r = 0

        for t in np.linspace(0.01, 0.99, 99):
            preds_bin = (y_pred >= t).astype(int)

            f1 = f1_score(y_true, preds_bin, zero_division=0)
            p = precision_score(y_true, preds_bin, zero_division=0)
            r = recall_score(y_true, preds_bin, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                best_p = p
                best_r = r

        thresholds.append(round(float(best_t), 2))
        per_label_f1s.append(float(best_f1))
        per_label_precision.append(float(best_p))
        per_label_recall.append(float(best_r))
        # ----- ECE -----
        ece = compute_ece(y_true, y_pred)
        per_label_ece.append(float(ece))

    # relative brier improvement
    relative_brier_improvement = [
        (per_label_brier_baseline[ind] - per_label_brier[ind]) / per_label_brier_baseline[ind]
        if per_label_brier_baseline[ind] > 0 else float('nan')
        for ind in range(L)
    ]

    # Micro-averaged metrics
    try:
        micro_ap = average_precision_score(all_y_true.reshape(-1), all_y_pred.reshape(-1))
    except ValueError:
        micro_ap = float('nan')
    try:
        micro_auroc = roc_auc_score(all_y_true.reshape(-1), all_y_pred.reshape(-1))
    except ValueError:
        micro_auroc = float('nan')

    # ----- MICRO ECE -----
    micro_ece = compute_ece(all_y_true.reshape(-1), all_y_pred.reshape(-1))

    # Macro-averaged metrics
    macro_ap = float(np.nanmean(per_label_ap))
    macro_auroc = float(np.nanmean(per_label_auroc))
    macro_brier = float(np.mean(per_label_brier))
    macro_f1 = float(np.mean(per_label_f1s))
    macro_ece = float(np.nanmean(per_label_ece))
    macro_precision = float(np.nanmean(per_label_precision))
    macro_recall = float(np.nanmean(per_label_recall))
    
    # LRAP
    lrap = label_ranking_average_precision_score(all_y_true, all_y_pred)

    return {
        "per_label_ap": per_label_ap,
        "per_label_auroc": per_label_auroc,
        "per_label_brier": per_label_brier,
        "per_label_brier_baseline": per_label_brier_baseline,
        "relative_brier_improvement": relative_brier_improvement,
        "per_label_f1": per_label_f1s,
        "per_label_precision": per_label_precision, 
        "per_label_recall": per_label_recall, 
        "per_label_ece": per_label_ece,
        "thresholds": thresholds,
        "macro_ap": float(macro_ap),
        "macro_auroc": float(macro_auroc),
        "macro_brier": float(macro_brier),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision), 
        "macro_recall": float(macro_recall),
        "macro_ece": float(macro_ece),
        "micro_ap": float(micro_ap),
        "micro_auroc": float(micro_auroc),
        "micro_ece": float(micro_ece),
        "lrap": float(lrap),
    }

def get_probs_and_labels(loader, model, device=torch.device("cpu")):
    model.eval()
    pbar = tqdm(total=len(loader), desc="Collecting predictions", unit="batch")
    all_probs = []
    all_labels = []
    with torch.inference_mode():
        for i, (x, y) in enumerate(loader):
            if (i + 1) % 5 == 0 or (i + 1) == len(loader):
                pbar.n = i + 1
                pbar.refresh()
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)  # multi-label output
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)