from river.drift import ADWIN
from utils.logger import get_logger

logger = get_logger("drift_adwin")

adwin = ADWIN()


def check_drift(loss_value: float) -> bool:
    """Feed loss/score into ADWIN; returns True if drift detected."""
    in_drift = adwin.update(loss_value)
    if in_drift:
        logger.warning(f"Drift detected! Loss={loss_value:.4f}")
    return in_drift
