import os
import shutil
from utils.settings import MODEL_DIR
from utils.logger import get_logger

logger = get_logger("versioning")


def version_model(model_name: str):
    """Copy current model to a versioned backup."""
    src = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(src):
        logger.error(f"Model not found: {src}")
        return

    versions_dir = os.path.join(MODEL_DIR, "versions")
    os.makedirs(versions_dir, exist_ok=True)

    dst = os.path.join(versions_dir, f"{model_name}.bak")
    shutil.copy2(src, dst)
    logger.info(f"Versioned {model_name} -> {dst}")


def rollback_model(model_name: str):
    """Restore last backup for a model."""
    versions_dir = os.path.join(MODEL_DIR, "versions")
    backup = os.path.join(versions_dir, f"{model_name}.bak")
    if not os.path.exists(backup):
        logger.error(f"No backup found for {model_name}")
        return

    dst = os.path.join(MODEL_DIR, model_name)
    shutil.copy2(backup, dst)
    logger.warning(f"Rolled back {model_name} from backup.")
