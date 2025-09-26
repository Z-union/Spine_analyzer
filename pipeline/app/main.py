import logging
import threading
import uuid
import os
from fastapi import FastAPI, Form, HTTPException, File, UploadFile, BackgroundTasks, Depends
from fastapi.responses import FileResponse
from typing import List
import json

from .config import settings
from .kafka_worker import kafka_worker
from .pipeline import run_pipeline_for_study
from .database import init_db, save_processing_result, get_result_by_study_id, get_result_by_processing_id
from .report_generator import create_disk_report
from .orthanc_client import get_dicomweb_client
from dicomweb_client.api import DICOMwebClient


# Configure logging
log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
log_format = settings.LOG_FORMAT or "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=log_level, format=log_format)
logger = logging.getLogger("dicom-pipeline")
logger.setLevel(log_level)

logger.info(f"Starting DICOM Pipeline with log level: {settings.LOG_LEVEL}")
if settings.DEBUG_MODE:
    logger.info("Debug mode enabled")
    logger.debug(f"Debug output directory: {settings.DEBUG_OUTPUT_DIR}")

app = FastAPI(title="DICOM Pipeline")


@app.on_event("startup")
def startup_event():
    init_db()  # Initialize SQLite DB
    if settings.KAFKA_BOOTSTRAP_SERVERS:
        t = threading.Thread(target=kafka_worker, daemon=True)
        t.start()
        logger.info("Running in Kafka mode")
    else:
        logger.info("Running in REST mode")


def extract_disk_data_from_results(pipeline_result):
    """
    Extract structured disk data for the report from pipeline results.
    Handles the case where 'results' is a JSON string and values are lists.
    """
    disk_data = []
    
    results_str = pipeline_result.get("results")
    if not results_str:
        logger.warning("No 'results' field found in pipeline_result.")
        return disk_data

    try:
        # Parse the results string into a list of dicts
        results = json.loads(results_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse 'results' JSON: {e}")
        return disk_data

    if not isinstance(results, list):
        logger.warning("Parsed 'results' is not a list.")
        return disk_data

    # Map and flatten data for report
    for item in results:
        if not isinstance(item, dict):
            continue

        def safe_get_value(key, default=None, index=0):
            """Safely get value from dict, handling list values."""
            value = item.get(key, default)
            if isinstance(value, list):
                return value[index] if len(value) > index else default
            return value

        def safe_get_nested_list_value(key, outer_index=0, inner_index=0, default=None):
            """For fields like 'UP endplate' which are list of lists."""
            value = item.get(key, default)
            if isinstance(value, list) and len(value) > outer_index:
                inner_list = value[outer_index]
                if isinstance(inner_list, list) and len(inner_list) > inner_index:
                    return inner_list[inner_index]
                return inner_list if not isinstance(inner_list, list) else default
            return default

        row = {
            "disk_label": safe_get_value("disk_label", "N/A"),
            "Modic": safe_get_value("Modic", "N/A"),
            "UP endplate": safe_get_nested_list_value("UP endplate", default="N/A"),
            "LOW endplate": safe_get_nested_list_value("LOW endplate", default="N/A"),
            "Spondylolisthesis": safe_get_nested_list_value("Spondylolisthesis", default="N/A"),
            "Disc herniation": safe_get_nested_list_value("Disc herniation", default="N/A"),
            "Disc narrowing": safe_get_nested_list_value("Disc narrowing", default="N/A"),
            "Disc bulging": safe_get_nested_list_value("Disc bulging", default="N/A"),
            "Pfirrmann grade": safe_get_value("Pfirrmann grade", "N/A"),
            "hernia_detected": safe_get_value("hernia_detected", False),
            "hernia_volume_mm3": safe_get_value("hernia_volume_mm3", "N/A"),
            "hernia_max_protrusion_mm": safe_get_value("hernia_max_protrusion_mm", "N/A"),
            "spondy_detected": safe_get_value("spondy_detected", False),
            "spondy_displacement_mm": safe_get_value("spondy_displacement_mm", "N/A"),
            "spondy_displacement_percentage": safe_get_value("spondy_displacement_percentage", "N/A"),
            "spondy_grade": safe_get_value("spondy_grade", "N/A"),
        }

        # Replace None/Null with "N/A" or empty
        for k, v in row.items():
            if v is None or v == 'null' or v == 'N/A':
                row[k] = ""

        disk_data.append(row)

    return disk_data

@app.post("/process-study/")
async def process_study(study_id: str = Form(...), client: DICOMwebClient = Depends(get_dicomweb_client)):
    try:
        processing_id = str(uuid.uuid4())
        logger.info(f"Processing study {study_id} with processing_id={processing_id}")

        result, segmentation_images = run_pipeline_for_study(study_id)

        # Extract pipeline result
        pipeline_result = result.get("pipeline_result", {})
        results_json = json.dumps(pipeline_result, default=str)  # Serialize complex objects

        study  = client.retrieve_study(study_id)
        ds = study[0]
        
        patient_info = {
            "patient_name": str(ds.get("PatientName", "Unknown")),
            "patient_age": str(ds.get("PatientAge", "N/A")),
            "study_date": ds.get("StudyDate", "N/A"),
            "modality": str(ds.get("Modality", "MRI")),
            "series_description": str(ds.get("SeriesDescription", "Lumbar Spine MRI"))
        }

        if patient_info["study_date"] != "N/A":
            sdate = patient_info["study_date"]
            patient_info["study_date"] = f"{sdate[:4]}-{sdate[4:6]}-{sdate[6:8]}"

        # Extract disk data for report
        disk_data = extract_disk_data_from_results(pipeline_result)
        report_filename = f"{processing_id}.docx"
        # Handle both legacy list and new variants dict for segmentation_images
        if isinstance(segmentation_images, dict):
            imgs = segmentation_images.get('variant_a') or next((v for k, v in segmentation_images.items() if isinstance(v, list) and v), [])
        else:
            imgs = segmentation_images or []
        idx1, idx2 = 27, 28
        img1 = imgs[idx1] if isinstance(imgs, list) and len(imgs) > idx1 else (imgs[-2] if isinstance(imgs, list) and len(imgs) >= 2 else None)
        img2 = imgs[idx2] if isinstance(imgs, list) and len(imgs) > idx2 else (imgs[-1] if isinstance(imgs, list) and len(imgs) >= 1 else None)
        report_path = create_disk_report(disk_data, patient_info, img1, img2, output_filename=report_filename)

        # Save to database
        save_processing_result(
            processing_id=processing_id,
            study_id=study_id,
            results_json=results_json,
            report_path=report_path
        )

        # Prepare response
        response = {
            "status": result.get("status", "unknown"),
            "study_id": study_id,
            "processing_id": processing_id,
            "report_available": True,
            "report_download_url": f"/report/{processing_id}",
            "pipeline_result": {
                "input_type": pipeline_result.get("input_type"),
                "archive_extracted_to": pipeline_result.get("archive_extracted_to"),
                "size_bytes": pipeline_result.get("size_bytes"),
                "series_detected": pipeline_result.get("series_detected"),
                "correspondence_pairs": pipeline_result.get("correspondence_pairs"),
                "results": pipeline_result.get("results"),
                "disk_count": len(pipeline_result.get("disk_results", [])),
                "pathology_count": len(pipeline_result.get("pathology_measurements", [])),
            }
        }

        return response

    except Exception as e:
        logger.error(f"Error processing study {study_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{study_id}")
async def get_status(study_id: str):
    """
    Get processing result and processing_id by study_id.
    """
    result = get_result_by_study_id(study_id)
    if not result:
        raise HTTPException(status_code=404, detail="Study not found or not processed yet.")
    return {
        "study_id": result["study_id"],
        "processing_id": result["id"],
        "processed_at": result["processed_at"],
        "report_available": bool(result["report_path"]),
        "report_download_url": f"/report/{result['id']}" if result["report_path"] else None
    }


@app.get("/report/{processing_id}")
async def download_report(processing_id: str):
    """
    Download the generated DOCX report by processing_id.
    """
    result = get_result_by_processing_id(processing_id)
    if not result:
        raise HTTPException(status_code=404, detail="Processing ID not found.")

    report_path = result["report_path"]
    if not report_path or not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found.")

    filename = os.path.basename(report_path)
    return FileResponse(
        path=report_path,
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        filename=f"report_{processing_id}.docx"
    )