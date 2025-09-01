import logging
import threading
from fastapi import FastAPI, Form, HTTPException
from .config import settings
from .kafka_worker import kafka_worker
from .pipeline import run_pipeline_for_study

# Настройка логирования один раз
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dicom-pipeline")
app = FastAPI(title="DICOM Pipeline")

@app.on_event("startup")
def startup_event():
    if settings.KAFKA_BOOTSTRAP_SERVERS:
        # Запускаем Kafka воркер в отдельном потоке
        t = threading.Thread(target=kafka_worker, daemon=True)
        t.start()
        logger.info("Running in Kafka mode")
    else:
        logger.info("Running in REST mode")

@app.post("/process-study/")
async def process_study(study_id: str = Form(...)):
    try:
        result = run_pipeline_for_study(study_id)

        # TODO: ДОБАВИТЬ ЖЕЛАЕМЫЙ ВОЗВРАТ РЕЗУЛЬТАТОВ

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))