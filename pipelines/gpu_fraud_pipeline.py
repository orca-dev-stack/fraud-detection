import json
from kafka import KafkaConsumer
import cudf

from utils.logger import get_logger
from features.gpu_vector_builder import build_gpu_vector
from inference.triton_client import triton_predict
from rules.thresholds import compute_final_threshold, is_fraud
from alerts.alert_manager import send_alert

logger = get_logger("gpu_fraud_pipeline")

consumer = KafkaConsumer(
    "osprey.transactions.raw",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
)


def run_gpu_fraud_pipeline(meta_model=None, recent_scores_buffer=None):
    """
    End-to-end GPU pipeline:
    Kafka → cuDF → GPU features → Triton → thresholds → alerts.
    """
    logger.info("Starting GPU fraud pipeline...")
    batch = []

    if recent_scores_buffer is None:
        recent_scores_buffer = []

    for msg in consumer:
        batch.append(msg.value)

        if len(batch) >= 1024:  # micro-batch
            # 1. Raw → cuDF (GPU)
            df = cudf.DataFrame(batch)

            # 2. Build GPU feature vectors
            gpu_features = build_gpu_vector(df)

            # 3. Triton inference (GPU model)
            preds = triton_predict("osprey_fraud_gpu", gpu_features)

            # 4. Thresholds per transaction
            for i in range(len(df)):
                score = float(preds[i])
                recent_scores_buffer.append(score)
                if len(recent_scores_buffer) > 5000:
                    recent_scores_buffer.pop(0)

                tx = df.iloc[i].to_pandas().to_dict()
                tx_type = tx.get("tx_type", "card")

                threshold = compute_final_threshold(
                    scores=np.array(recent_scores_buffer),
                    tx_type=tx_type,
                    meta_model=meta_model,
                    context_features=tx,
                )

                if is_fraud(score, threshold):
                    send_alert(
                        message=f" GPU FRAUD ALERT — score={score:.3f}, threshold={threshold:.3f}",
                        severity="high",
                    )

            logger.info(f"Processed GPU batch: {len(df)} rows")
            batch = []
