import json
from kafka import KafkaConsumer
from utils.logger import get_logger
from models.incremental_train import incremental_train
from data.unified_loader import load_all
from features.build_features import build_features

logger = get_logger("kafka_incremental")

consumer = KafkaConsumer(
    "osprey.incremental.trigger",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
)


def run_incremental_listener():
    logger.info("Kafka incremental listener started.")
    for msg in consumer:
        payload = msg.value
        logger.info(f"Incremental training trigger received: {payload}")

        df = load_all()
        df = build_features(df)
        incremental_train(df, batch_size=5000)
