import logging
import threading
from fastapi import FastAPI, Form, HTTPException
from .config import settings
from .kafka_worker import kafka_worker
from .pipeline import run_pipeline_for_study

# Configure logging based on settings from .env
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
log_format = settings.LOG_FORMAT or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Configure root logger
logging.basicConfig(
    level=log_level,
    format=log_format
)

# Create unified logger for the application
logger = logging.getLogger("dicom-pipeline")
logger.setLevel(log_level)

# Log startup configuration
logger.info(f"Starting DICOM Pipeline with log level: {settings.LOG_LEVEL}")
if settings.DEBUG_MODE:
    logger.info("Debug mode enabled")
    logger.debug(f"Debug output directory: {settings.DEBUG_OUTPUT_DIR}")
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

        # Remove non-serializable fields from the response
        response = {
            "status": result.get("status", "unknown"),
            "study_id": result.get("study_id"),
            "report_upload": result.get("report_upload", {}),
        }
        
        # Add pipeline results if available, but exclude non-serializable fields
        if "pipeline_result" in result and result["pipeline_result"]:
            pipeline_result = result["pipeline_result"]
            response["pipeline_result"] = {
                "input_type": pipeline_result.get("input_type"),
                "archive_extracted_to": pipeline_result.get("archive_extracted_to"),
                "size_bytes": pipeline_result.get("size_bytes"),
                "series_detected": pipeline_result.get("series_detected"),
                "correspondence_pairs": pipeline_result.get("correspondence_pairs"),
                "results": pipeline_result.get("results"),  # This is already JSON string
                # Exclude 'segmentations' as it contains numpy arrays
                # Exclude 'disk_results' and 'pathology_measurements' as they may contain complex objects
            }
            
            # Add summary of disk results if available
            if "disk_results" in pipeline_result:
                response["pipeline_result"]["disk_count"] = len(pipeline_result["disk_results"])
            
            # Add summary of pathology measurements if available  
            if "pathology_measurements" in pipeline_result:
                response["pipeline_result"]["pathology_count"] = len(pipeline_result["pathology_measurements"])

        return response
    except Exception as e:
        logger.error(f"Error processing study {study_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))