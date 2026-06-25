import os
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import mlflow
import mlflow.sklearn
from models.roc_pr_plots import plot_all



from config.train_config import TRAIN_CONFIG

MODEL_DIR = "models_artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)

mlflow.set_experiment("osprey_tabular")


def train_tabular(X, y):
    cfg = TRAIN_CONFIG["tabular"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["random_state"]
    )

    # LightGBM
    with mlflow.start_run(run_name="LightGBM_fraud"):
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val)
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": cfg["lgbm"]["learning_rate"],
            "num_leaves": cfg["lgbm"]["num_leaves"],
            "feature_fraction": cfg["lgbm"]["feature_fraction"],
            "bagging_fraction": cfg["lgbm"]["bagging_fraction"],
            "bagging_freq": cfg["lgbm"]["bagging_freq"],
        }
        mlflow.log_params(params)
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_val],
            num_boost_round=cfg["lgbm"]["num_boost_round"],
            early_stopping_rounds=cfg["lgbm"]["early_stopping_rounds"],
            verbose_eval=False,
        )
        preds = model.predict(X_val)
        auc = roc_auc_score(y_val, preds)
        mlflow.log_metric("auc", auc)
        mlflow.lightgbm.log_model(model, "model")
        model.save_model(os.path.join(MODEL_DIR, "lgbm_fraud.txt"))
        print("[LightGBM] AUC:", auc)

    # XGBoost
    with mlflow.start_run(run_name="XGBoost_fraud"):
        model = xgb.XGBClassifier(
            n_estimators=cfg["xgb"]["n_estimators"],
            max_depth=cfg["xgb"]["max_depth"],
            learning_rate=cfg["xgb"]["learning_rate"],
            subsample=cfg["xgb"]["subsample"],
            colsample_bytree=cfg["xgb"]["colsample_bytree"],
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=-1,
        )
        mlflow.log_params(model.get_params())
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        mlflow.log_metric("auc", auc)
        mlflow.sklearn.log_model(model, "model")
        model.save_model(os.path.join(MODEL_DIR, "xgb_fraud.json"))
        print("[XGBoost] AUC:", auc)

    # CatBoost
    with mlflow.start_run(run_name="CatBoost_fraud"):
        model = CatBoostClassifier(
            iterations=cfg["catboost"]["iterations"],
            depth=cfg["catboost"]["depth"],
            learning_rate=cfg["catboost"]["learning_rate"],
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
        )
        mlflow.log_params(model.get_params())
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        mlflow.log_metric("auc", auc)
        mlflow.catboost.log_model(model, "model")
        model.save_model(os.path.join(MODEL_DIR, "catboost_fraud.cbm"))
        print("[CatBoost] AUC:", auc)

    print("[Tabular] All models trained and logged to MLflow.")
    
    metrics = plot_all(y_val, preds, model_name="LightGBM_fraud")
    print(metrics)
