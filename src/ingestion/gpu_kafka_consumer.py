import json
from kafka import KafkaConsumer
import cudf
from utils.logger import get_logger
from features.gpu_vector_builder import build_gpu_vector

logger = get_logger("gpu_kafka_consumer")

consumer = KafkaConsumer(
    "osprey.transactions.raw",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
)


def run_gpu_consumer():
    logger.info("GPU Kafka consumer started.")
    batch = []

    for msg in consumer:
        batch.append(msg.value)

        if len(batch) >= 1024:  # micro-batch
            df = cudf.DataFrame(batch)        # goes straight to GPU
            gpu_features = build_gpu_vector(df)
            # TODO: call GPU model here
            logger.info(f"Processed batch on GPU: {len(df)} rows")
            batch = []
