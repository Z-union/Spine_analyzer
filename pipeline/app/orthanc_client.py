import os
import tempfile
import requests
from .config import settings

def fetch_study_temp(study_id: str) -> str:
    """Скачивает исследование из Orthanc во временный файл"""
    url = f"{settings.ORTHANC_URL}/studies/{study_id}/archive"
    resp = requests.get(url, auth=(settings.ORTHANC_USER, settings.ORTHANC_PASSWORD))
    resp.raise_for_status()
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    with os.fdopen(tmp_fd, "wb") as tmp_file:
        tmp_file.write(resp.content)
    return tmp_path
