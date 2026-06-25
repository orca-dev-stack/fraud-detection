import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from rules.thresholds import get_risk_level, is_fraud

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from data.unified_loader import load_all
from features.build_features import build_features
from utils.settings import MODEL_DIR
from utils.logger import get_logger

logger = get_logger("incremental_train")

mlflow.set_experiment("osprey_incremental")


# ---------------------------------------------------------
# 1. Load existing models (if available)
# ---------------------------------------------------------
def load_existing_models():
    models = {}

    # SGD (partial fit)
    sgd_path = os.path.join(MODEL_DIR, "sgd_incremental.pkl")
    if os.path.exists(sgd_path):
        models["sgd"] = mlflow.sklearn.load_model(sgd_path)
        logger.info("Loaded existing SGD model.")
    else:
        models["sgd"] = SGDClassifier(loss="log_loss", max_iter=5)
        logger.info("Created new SGD model.")

    # LightGBM
    lgb_path = os.path.join(MODEL_DIR, "lgbm_fraud.txt")
    if os.path.exists(lgb_path):
        models["lgbm"] = lgb.Booster(model_file=lgb_path)
        logger.info("Loaded existing LightGBM model.")
    else:
        models["lgbm"] = None

    # XGBoost
    xgb_path = os.path.join(MODEL_DIR, "xgb_fraud.json")
    if os.path.exists(xgb_path):
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_path)
        models["xgb"] = xgb_model
        logger.info("Loaded existing XGBoost model.")
    else:
        models["xgb"] = None

    # CatBoost
    cat_path = os.path.join(MODEL_DIR, "catboost_fraud.cbm")
    if os.path.exists(cat_path):
        cat = CatBoostClassifier()
        cat.load_model(cat_path)
        models["cat"] = cat
        logger.info("Loaded existing CatBoost model.")
    else:
        models["cat"] = None

    return models


# ---------------------------------------------------------
# 2. Incremental training logic
# ---------------------------------------------------------
def incremental_train(df: pd.DataFrame, batch_size: int = 5000):
    logger.info("Starting incremental training...")

    df = df.sample(frac=1).reset_index(drop=True)  # shuffle
    X = df.drop(columns=["label"])
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = load_existing_models()

    # SGD supports true incremental learning
    sgd = models["sgd"]

    # LightGBM incremental training requires Dataset + keep_training_booster
    lgbm = models["lgbm"]

    # XGBoost incremental training via update()
    xgb_model = models["xgb"]

    # CatBoost incremental training via continue_training
    cat = models["cat"]

    # ---------------------------------------------------------
    # Loop through batches
    # ---------------------------------------------------------
    for i in range(0, len(df), batch_size):
        X_batch = X_scaled[i:i+batch_size]
        y_batch = y.iloc[i:i+batch_size]

        logger.info(f"Training batch {i} → {i+batch_size}")

        # SGD partial fit
        with mlflow.start_run(run_name="SGD_incremental", nested=True):
            sgd.partial_fit(X_batch, y_batch, classes=np.array([0, 1]))
            mlflow.sklearn.log_model(sgd, "sgd_incremental")

        # LightGBM incremental
        if lgbm is not None:
            with mlflow.start_run(run_name="LGBM_incremental", nested=True):
                train_data = lgb.Dataset(X_batch, label=y_batch)
                lgbm = lgb.train(
                    params={"objective": "binary", "learning_rate": 0.01},
                    train_set=train_data,
                    init_model=lgbm,
                    keep_training_booster=True,
                    num_boost_round=50
                )
                lgbm.save_model(os.path.join(MODEL_DIR, "lgbm_fraud.txt"))

        # XGBoost incremental
        if xgb_model is not None:
            with mlflow.start_run(run_name="XGB_incremental", nested=True):
                xgb_model.fit(
                    X_batch, y_batch,
                    xgb_model=xgb_model.get_booster(),
                    verbose=False
                )
                xgb_model.save_model(os.path.join(MODEL_DIR, "xgb_fraud.json"))

        # CatBoost incremental
        if cat is not None:
            with mlflow.start_run(run_name="CatBoost_incremental", nested=True):
                cat.fit(
                    X_batch, y_batch,
                    init_model=cat,
                    verbose=False
                )
                cat.save_model(os.path.join(MODEL_DIR, "catboost_fraud.cbm"))

    # Save SGD model
    mlflow.sklearn.save_model(sgd, os.path.join(MODEL_DIR, "sgd_incremental.pkl"))

    logger.info("Incremental training complete.")


# ---------------------------------------------------------
# 3. Main entry point
# ---------------------------------------------------------
def main():
    logger.info("Loading unified dataset...")
    df = load_all()
    df = build_features(df)

    logger.info(f"Dataset loaded: {df.shape}")
    incremental_train(df, batch_size=5000)



score = unified_score

risk = get_risk_level(score)
fraud_flag = is_fraud(score)

return {
    "score": score,
    "risk": risk,
    "fraud": fraud_flag
}



if __name__ == "__main__":
    main()
