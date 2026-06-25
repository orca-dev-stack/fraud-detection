import json
from kafka import KafkaConsumer
from utils.logger import get_logger
from inference.unified_score import combine_scores
from alerts.alert_manager import send_alert

logger = get_logger("kafka_consumer")

consumer = KafkaConsumer(
    "osprey.transactions.raw",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
)


def score_transaction(tx: dict) -> float:
    # TODO: build features + call your models
    # For now, just a dummy score:
    return 0.9


def run_consumer():
    logger.info("Kafka consumer started.")
    for msg in consumer:
        tx = msg.value
        score = score_transaction(tx)

        if score > 0.85:
            send_alert(
                message=f" High-risk transaction (Kafka) — score={score:.3f}",
                severity="high",
            )
        elif score > 0.65:
            send_alert(
                message=f" Medium-risk transaction (Kafka) — score={score:.3f}",
                severity="medium",
            )

counter = 0
counter += 1
if counter % 10000 == 0:
    producer.send("osprey.incremental.trigger", {"reason": "volume_threshold"})

