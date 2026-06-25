import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from utils.logger import get_logger
from utils.settings import MODEL_DIR

logger = get_logger("roc_pr_plots")

PLOT_DIR = os.path.join(MODEL_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def plot_roc_curve(y_true, y_pred, model_name="model"):
    """Generate and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {model_name}")
    plt.legend(loc="lower right")

    save_path = os.path.join(PLOT_DIR, f"{model_name}_roc.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"ROC curve saved: {save_path}")
    return roc_auc


def plot_pr_curve(y_true, y_pred, model_name="model"):
    """Generate and save Precision‑Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="green", lw=2, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision‑Recall Curve — {model_name}")
    plt.legend(loc="lower left")

    save_path = os.path.join(PLOT_DIR, f"{model_name}_pr.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"PR curve saved: {save_path}")
    return ap


def plot_all(y_true, y_pred, model_name="model"):
    """Generate both ROC and PR curves."""
    logger.info(f"Generating ROC + PR plots for {model_name}...")
    roc_auc = plot_roc_curve(y_true, y_pred, model_name)
    ap = plot_pr_curve(y_true, y_pred, model_name)
    return {"roc_auc": roc_auc, "average_precision": ap}
