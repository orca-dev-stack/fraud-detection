import numpy as np
from datetime import datetime
from utils.logger import get_logger

logger = get_logger("thresholds")

# ---------------------------------------------------------
# 0. Base static thresholds
# ---------------------------------------------------------

DEFAULT_THRESHOLDS = {
    "high": 0.85,
    "medium": 0.65,
    "low": 0.40,
}

DEFAULT_TYPE_THRESHOLDS = {
    "card": 0.80,
    "crypto": 0.75,
    "ach": 0.70,
    "wallet": 0.78,
    "merchant": 0.72,
}


# ---------------------------------------------------------
# 1. Static threshold logic
# ---------------------------------------------------------

def get_risk_level(score: float) -> str:
    if score >= DEFAULT_THRESHOLDS["high"]:
        return "high"
    elif score >= DEFAULT_THRESHOLDS["medium"]:
        return "medium"
    elif score >= DEFAULT_THRESHOLDS["low"]:
        return "low"
    return "none"


def is_fraud_static(score: float) -> bool:
    return score >= DEFAULT_THRESHOLDS["medium"]


# ---------------------------------------------------------
# 2. Drift‑adaptive thresholds
# ---------------------------------------------------------

def update_thresholds_from_drift(drift_value: float):
    global DEFAULT_THRESHOLDS

    if drift_value > 0.05:
        DEFAULT_THRESHOLDS["high"] = min(0.95, DEFAULT_THRESHOLDS["high"] + 0.02)
        DEFAULT_THRESHOLDS["medium"] = min(0.80, DEFAULT_THRESHOLDS["medium"] + 0.02)
        logger.warning(f"Thresholds tightened due to drift: {DEFAULT_THRESHOLDS}")

    elif drift_value < -0.05:
        DEFAULT_THRESHOLDS["high"] = max(0.80, DEFAULT_THRESHOLDS["high"] - 0.02)
        DEFAULT_THRESHOLDS["medium"] = max(0.60, DEFAULT_THRESHOLDS["medium"] - 0.02)
        logger.info(f"Thresholds relaxed: {DEFAULT_THRESHOLDS}")

    return DEFAULT_THRESHOLDS


# ---------------------------------------------------------
# 3. Percentile‑based thresholds
# ---------------------------------------------------------

def percentile_threshold(scores: np.ndarray, pct: float = 0.98):
    if len(scores) == 0:
        return DEFAULT_THRESHOLDS["high"]
    t = float(np.percentile(scores, pct * 100))
    logger.info(f"Percentile threshold ({pct*100:.1f}%): {t:.4f}")
    return t


# ---------------------------------------------------------
# 4. Transaction‑type thresholds
# ---------------------------------------------------------

def type_based_threshold(tx_type: str):
    return DEFAULT_TYPE_THRESHOLDS.get(tx_type.lower(), 0.75)


# ---------------------------------------------------------
# 5. Time‑of‑day thresholds
# ---------------------------------------------------------

def time_of_day_threshold():
    hour = datetime.utcnow().hour
    if 0 <= hour < 6:
        return 0.90
    elif 6 <= hour < 18:
        return 0.80
    else:
        return 0.85


# ---------------------------------------------------------
# 6. Meta‑model learned threshold
# ---------------------------------------------------------

def learned_threshold(meta_model, features: dict):
    try:
        t = meta_model.predict([list(features.values())])[0]
        t = float(np.clip(t, 0.5, 0.99))
        logger.info(f"Meta-model threshold: {t:.4f}")
        return t
    except Exception as e:
        logger.error(f"Meta-model threshold failed: {e}")
        return 0.80


# ---------------------------------------------------------
# 7. Unified threshold engine
# ---------------------------------------------------------

def compute_final_threshold(
    scores: np.ndarray,
    tx_type: str,
    meta_model=None,
    context_features: dict = None
):
    t_pct = percentile_threshold(scores)
    t_type = type_based_threshold(tx_type)
    t_time = time_of_day_threshold()
    t_meta = learned_threshold(meta_model, context_features) if meta_model else 0.80

    final = (
        0.35 * t_pct +
        0.25 * t_type +
        0.20 * t_time +
        0.20 * t_meta
    )

    final = float(np.clip(final, 0.5, 0.99))
    logger.info(f"Final threshold: {final:.4f}")
    return final


# ---------------------------------------------------------
# 8. Final fraud decision
# ---------------------------------------------------------

def is_fraud(score: float, threshold: float):
    return score >= threshold
