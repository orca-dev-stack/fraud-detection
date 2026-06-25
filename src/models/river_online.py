from river import linear_model, preprocessing, metrics
from utils.logger import get_logger

logger = get_logger("river_online")

model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
metric = metrics.ROCAUC()


def river_update(x: dict, y: int):
    """Update online model with one sample."""
    y_pred = model.predict_proba_one(x).get(1, 0.0)
    metric.update(y, y_pred)
    model.learn_one(x, y)
    logger.info(f"River online update — y={y}, y_pred={y_pred:.3f}, AUC={metric.get():.3f}")
    return y_pred
