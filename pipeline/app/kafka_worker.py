import json
import logging
from concurrent.futures import ThreadPoolExecutor
from kafka import KafkaConsumer
from .config import settings
from .pipeline import run_pipeline_for_study

# Используем единый логгер из main
logger = logging.getLogger("dicom-pipeline")

executor = ThreadPoolExecutor(max_workers=settings.WORKERS)


def process_message(study_id: str):
    try:
        result = run_pipeline_for_study(study_id)
        logger.info(f"Processed study {study_id}: {result}")
    except Exception:
        logger.exception(f"Failed to process study {study_id}")

def kafka_worker():
    consumer = KafkaConsumer(
        settings.KAFKA_TOPIC,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS.split(","),
        group_id=settings.KAFKA_GROUP_ID,
        security_protocol=settings.KAFKA_SECURITY_PROTOCOL,
        ssl_cafile=settings.KAFKA_SSL_CAFILE,
        ssl_certfile=settings.KAFKA_SSL_CERTFILE,
        ssl_keyfile=settings.KAFKA_SSL_KEYFILE,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest"
    )
    logger.info(f"Kafka consumer started on topic {settings.KAFKA_TOPIC}")
    for msg in consumer:
        study_id = msg.value.get("study_id")
        if study_id:
            executor.submit(process_message, study_id)