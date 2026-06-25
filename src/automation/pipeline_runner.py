import time
from utils.logger import get_logger
from data.unified_loader import load_all
from features.build_features import build_features

from models.incremental_train import incremental_train
from models.train_tabular import train_tabular

from monitoring.auto_retrain import maybe_retrain
from monitoring.drift_adwin import check_drift

from ingestion.kafka_producer import publish_transaction
from ingestion.rabbit_producer import queue_alert

logger = get_logger("pipeline_runner")


class PipelineRunner:
    """
    Main orchestrator for Osprey automation.
    Handles:
    - Data loading
    - Feature building
    - Incremental training
    - Full retraining
    - Drift detection
    - Alerting
    - Kafka/RabbitMQ integration
    """

    def __init__(self):
        logger.info("PipelineRunner initialized.")

    # ---------------------------------------------------------
    # 1. Load + preprocess data
    # ---------------------------------------------------------
    def load_and_prepare(self):
        logger.info("Loading unified dataset...")
        df = load_all()
        logger.info(f"Loaded dataset: {df.shape}")

        logger.info("Building features...")
        df = build_features(df)
        logger.info(f"Feature matrix ready: {df.shape}")

        return df

    # ---------------------------------------------------------
    # 2. Incremental training
    # ---------------------------------------------------------
    def run_incremental_training(self, df):
        logger.info("Running incremental training...")
        incremental_train(df, batch_size=5000)
        logger.info("Incremental training complete.")

    # ---------------------------------------------------------
    # 3. Full retraining
    # ---------------------------------------------------------
    def run_full_training(self, df):
        logger.info("Running full retraining...")
        X = df.drop(columns=["label"])
        y = df["label"]
        train_tabular(X, y)
        logger.info("Full retraining complete.")

    # ---------------------------------------------------------
    # 4. Drift detection + auto retrain
    # ---------------------------------------------------------
    def check_for_drift(self, current_auc):
        logger.info(f"Checking drift for AUC={current_auc:.4f}")
        maybe_retrain(current_auc)

    # ---------------------------------------------------------
    # 5. Push events to Kafka
    # ---------------------------------------------------------
    def push_to_kafka(self, df):
        logger.info("Publishing sample transactions to Kafka...")
        for _, row in df.head(5).iterrows():
            publish_transaction(row.to_dict())
        logger.info("Kafka publish complete.")

    # ---------------------------------------------------------
    # 6. Push alerts to RabbitMQ
    # ---------------------------------------------------------
    def push_alert(self, message, severity="medium"):
        logger.info(f"Queueing alert: {message}")
        queue_alert(message, severity)

    # ---------------------------------------------------------
    # 7. Full pipeline execution
    # ---------------------------------------------------------
    def run(self):
        logger.info("Starting Osprey automation pipeline...")

        df = self.load_and_prepare()

        # Step 1: Incremental training
        self.run_incremental_training(df)

        # Step 2: Drift check (dummy AUC for now)
        dummy_auc = 0.88
        self.check_for_drift(dummy_auc)

        # Step 3: Push sample events to Kafka
        self.push_to_kafka(df)

        # Step 4: Send alert to RabbitMQ
        self.push_alert("Pipeline run completed successfully", "low")

        logger.info("Pipeline execution finished.")


if __name__ == "__main__":
    runner = PipelineRunner()
    runner.run()
