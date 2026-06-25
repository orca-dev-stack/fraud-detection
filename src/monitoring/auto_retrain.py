from monitoring.drift_adwin import check_drift
from models.train_tabular import train_tabular
from data.unified_loader import load_all
from features.build_features import build_features
from utils.logger import get_logger

logger = get_logger("auto_retrain")


def maybe_retrain(current_auc: float, threshold: float = 0.90):
    """If AUC drops below threshold or drift detected, retrain."""
    loss = 1 - current_auc
    drift = check_drift(loss)

    if current_auc < threshold or drift:
        logger.warning(f"Triggering retrain — AUC={current_auc:.3f}, drift={drift}")
        df = load_all()
        df = build_features(df)
        X = df.drop(columns=["label"])
        y = df["label"]
        train_tabular(X, y)
        logger.info("Retrain complete.")
