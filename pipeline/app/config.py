from typing import Optional
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root and .env absolute path to ensure it loads regardless of CWD
_BASE_DIR = Path(__file__).resolve().parents[2]
_ENV_PATH = _BASE_DIR / ".env"

class Settings(BaseSettings):
    ORTHANC_URL: str = "http://localhost:8042"
    ORTHANC_USER: str = "orthanc"
    ORTHANC_PASSWORD: str = "orthanc"

    KAFKA_BOOTSTRAP_SERVERS: Optional[str] = None
    KAFKA_TOPIC: Optional[str] = None
    KAFKA_GROUP_ID: Optional[str] = None
    KAFKA_SECURITY_PROTOCOL: Optional[str] = None
    KAFKA_SSL_CAFILE: Optional[str] = None
    KAFKA_SSL_CERTFILE: Optional[str] = None
    KAFKA_SSL_KEYFILE: Optional[str] = None

    TRITON_URL: str = "http://orthanc:8001"

    WORKERS: int = 2

    SAG_PATCH_SIZE: Optional[list[int]] = [128, 96, 96]
    AX_PATCH_SIZE: Optional[list[int]] = [224, 224]

    DILATE_SIZE: int = 5
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
