import torch 
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score, 
    label_ranking_average_precision_score
)
import numpy as np 

def batch_training_metrics(y_true, y_pred):
    """Compute sums that can be averaged later."""
    y_true = y_true.float()
    y_pred = y_pred.float()
    
    # Sum predicted probs per label
    sum_pred_prob_per_label = torch.sum(y_pred, dim=0)  # (L,)
    
    # Sum ground truth per label
    sum_true_per_label = torch.sum(y_true, dim=0)       # (L,)

    # Soft cardinality
    soft_cardinality_sum = torch.sum(torch.sum(y_pred, dim=1))  # scalar

    # Probability mass
    prob_mass_sum = torch.sum(y_pred)  # scalar
    
    # Calibration metrics - accumulate per label
    sum_pred_given_positive = torch.sum(y_pred * y_true, dim=0)  # (L,)
    sum_pred_given_negative = torch.sum(y_pred * (1 - y_true), dim=0)  # (L,)
    count_positive = torch.sum(y_true, dim=0)  # (L,)
    count_negative = torch.sum(1 - y_true, dim=0)  # (L,)

    return {
        "sum_pred_prob_per_label": sum_pred_prob_per_label,
        "sum_true_per_label": sum_true_per_label,
        "soft_cardinality_sum": soft_cardinality_sum,
        "prob_mass_sum": prob_mass_sum,
        "sum_pred_given_positive": sum_pred_given_positive,
        "sum_pred_given_negative": sum_pred_given_negative,
        "count_positive": count_positive,
        "count_negative": count_negative,
    }


def aggregate_training_epoch(batch_stats, total_samples):
    L = batch_stats[0]["sum_pred_prob_per_label"].shape[0]

    total_pred_prob = torch.zeros(L)
    total_true = torch.zeros(L)
    total_prob_mass = 0.0
    total_cardinality = 0.0
    total_pred_pos = torch.zeros(L)
    total_pred_neg = torch.zeros(L)
    total_count_pos = torch.zeros(L)
    total_count_neg = torch.zeros(L)

    for s in batch_stats:
        total_pred_prob += s["sum_pred_prob_per_label"].cpu()
        total_true += s["sum_true_per_label"].cpu()
        total_prob_mass += s["prob_mass_sum"].item()
        total_cardinality += s["soft_cardinality_sum"].item()
        total_pred_pos += s["sum_pred_given_positive"].cpu()
        total_pred_neg += s["sum_pred_given_negative"].cpu()
        total_count_pos += s["count_positive"].cpu()
        total_count_neg += s["count_negative"].cpu()
    
    mean_pred_when_positive = (total_pred_pos / (total_count_pos + 1e-8)).tolist()
    mean_pred_when_negative = (total_pred_neg / (total_count_neg + 1e-8)).tolist()

    return {
        "mean_pred_prob_per_label": (total_pred_prob / total_samples).tolist(),
        "mean_true_prob_per_label": (total_true / total_samples).tolist(),
        "mean_prob_mass": total_prob_mass / total_samples,
        "mean_cardinality": total_cardinality / total_samples,
        "mean_pred_when_positive": mean_pred_when_positive,
        "mean_pred_when_negative": mean_pred_when_negative,
        "calibration_gap": [(p - n) for p, n in zip(mean_pred_when_positive, mean_pred_when_negative)],
    }

def compute_ranking_metrics(all_y_true, all_y_pred):
    """Compute AP/AUROC/LRAP ranking metrics. Inputs are numpy arrays."""
    L = all_y_true.shape[1]

    per_label_ap = []
    per_label_auroc = []
    
    for j in range(L):
        # Average Precision
        ap = average_precision_score(all_y_true[:, j], all_y_pred[:, j])
        per_label_ap.append(float(ap))
        
        # AUROC
        try:
            auroc = roc_auc_score(all_y_true[:, j], all_y_pred[:, j])
            per_label_auroc.append(float(auroc))
        except ValueError:
            # Handle case where only one class is present in y_true
            per_label_auroc.append(float('nan'))

    # Micro-averaged metrics
    micro_ap = average_precision_score(all_y_true.reshape(-1), all_y_pred.reshape(-1))
    try:
        micro_auroc = roc_auc_score(all_y_true.reshape(-1), all_y_pred.reshape(-1))
    except ValueError:
        micro_auroc = float('nan')
    
    # Macro-averaged metrics
    macro_ap = sum(per_label_ap) / L
    valid_aurocs = [x for x in per_label_auroc if not np.isnan(x)]
    macro_auroc = sum(valid_aurocs) / len(valid_aurocs) if valid_aurocs else float('nan')
    
    # LRAP
    lrap = label_ranking_average_precision_score(all_y_true, all_y_pred)

    return {
        "per_label_ap": per_label_ap,
        "per_label_auroc": per_label_auroc,
        "macro_ap": float(macro_ap),
        "macro_auroc": float(macro_auroc),
        "micro_ap": float(micro_ap),
        "micro_auroc": float(micro_auroc),
        "lrap": float(lrap),
    }
