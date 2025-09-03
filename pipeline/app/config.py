from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root and .env absolute path to ensure it loads regardless of CWD
_BASE_DIR = Path(__file__).resolve().parents[2]
_ENV_PATH = _BASE_DIR / ".env"

class Settings(BaseSettings):
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Optional[str] = None
    
    # Orthanc PACS Server
    ORTHANC_URL: str = "http://orthanc:8042"
    ORTHANC_USER: str = "orthanc"
    ORTHANC_PASSWORD: str = "orthanc"

    # Triton Inference Server
    TRITON_URL: str = "triton:8001"
    TRITON_TIMEOUT: Optional[int] = None
    
    # Kafka Configuration (optional - for async processing)
    KAFKA_BOOTSTRAP_SERVERS: Optional[str] = None
    KAFKA_TOPIC: Optional[str] = None
    KAFKA_GROUP_ID: Optional[str] = None
    KAFKA_SECURITY_PROTOCOL: Optional[str] = None
    KAFKA_SSL_CAFILE: Optional[str] = None
    KAFKA_SSL_CERTFILE: Optional[str] = None
    KAFKA_SSL_KEYFILE: Optional[str] = None
    KAFKA_SASL_MECHANISM: Optional[str] = None
    KAFKA_SASL_USERNAME: Optional[str] = None
    KAFKA_SASL_PASSWORD: Optional[str] = None

    # Processing Configuration
    WORKERS: int = 2
    SAG_PATCH_SIZE: Optional[list[int]] = [128, 96, 96]
    AX_PATCH_SIZE: Optional[list[int]] = [224, 224]
    DILATE_SIZE: int = 5
    CANAL_LABEL: int = 2
    
    # Batch sizes for different models
    SEG_SAG_STAGE_1_BATCH_SIZE: int = 1
    SEG_SAG_STAGE_2_BATCH_SIZE: int = 1
    GRADING_BATCH_SIZE: int = 4
    INFERENCE_BATCH_SIZE: int = 1  # General fallback
    
    USE_GPU: bool = True
    
    # Performance Tuning
    MAX_CONCURRENT_STUDIES: int = 2
    MEMORY_LIMIT_MB: int = 4096
    STUDY_TIMEOUT: int = 1800
    
    # Output Configuration
    GENERATE_SR_REPORT: bool = True
    GENERATE_SC_IMAGES: bool = True
    MAX_SC_IMAGES: int = 50
    SC_IMAGE_QUALITY: int = 95
    
    # Debug Options
    DEBUG_MODE: bool = False
    DEBUG_OUTPUT_DIR: str = "/tmp/spine_analyzer_debug"
    SAVE_INTERMEDIATE_RESULTS: bool = False
    
    # Disk Labels and Descriptions
    EXTRACT_LABELS_RANGE: list[int] = list(range(63, 101))
    VERTEBRA_DESCRIPTIONS: dict[int, str] = {
        63: 'C2-C3', 64: 'C3-C4', 65: 'C4-C5', 66: 'C5-C6', 67: 'C6-C7',
        71: 'C7-T1', 72: 'T1-T2', 73: 'T2-T3', 74: 'T3-T4', 75: 'T4-T5',
        76: 'T5-T6', 77: 'T6-T7', 78: 'T7-T8', 79: 'T8-T9', 80: 'T9-T10',
        81: 'T10-T11', 82: 'T11-T12', 91: 'T12-L1', 92: 'L1-L2', 93: 'L2-L3',
        94: 'L3-L4', 95: 'L4-L5', 96: 'L5-S1', 100: 'S1-S2'
    }
    LANDMARK_LABELS: list[int] = list(VERTEBRA_DESCRIPTIONS.keys())
    CANAL_LABEL: int = 2

    model_config = SettingsConfigDict(env_file=str(_ENV_PATH), env_file_encoding="utf-8")


settings = Settings()
