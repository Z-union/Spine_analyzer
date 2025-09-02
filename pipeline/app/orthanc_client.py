import os
import tempfile
import requests
import logging
from .config import settings

logger = logging.getLogger("dicom-pipeline")

def find_orthanc_id_by_uid(uid: str, level: str = "Study") -> str:
    """
    Находит Orthanc ID по DICOM UID
    
    Args:
        uid: DICOM UID (StudyInstanceUID, SeriesInstanceUID или SOPInstanceUID)
        level: Уровень поиска - "Study", "Series" или "Instance"
        
    Returns:
        Orthanc internal ID
        
    Raises:
        ValueError: если объект не найден
    """
    url = f"{settings.ORTHANC_URL}/tools/find"
    
    # Определяем поле для поиска в зависимости от уровня
    query_field = {
        "Study": "StudyInstanceUID",
        "Series": "SeriesInstanceUID", 
        "Instance": "SOPInstanceUID"
    }.get(level)
    
    if not query_field:
        raise ValueError(f"Invalid level: {level}. Must be Study, Series or Instance")
    
    query = {
        "Level": level,
        "Query": {
            query_field: uid
        }
    }
    
    resp = requests.post(
        url,
        json=query,
        auth=(settings.ORTHANC_USER, settings.ORTHANC_PASSWORD)
    )
    resp.raise_for_status()
    
    results = resp.json()
    if not results:
        raise ValueError(f"{level} with UID {uid} not found in Orthanc")
    
    orthanc_id = results[0]
    logger.info(f"Found Orthanc {level} ID {orthanc_id} for UID {uid}")
    return orthanc_id

def fetch_instance_temp(instance_uid: str) -> str:
    """
    Скачивает DICOM Instance из Orthanc во временный файл.
    Работает напрямую с SOPInstanceUID без необходимости знать Study ID.
    
    Args:
        instance_uid: SOPInstanceUID или Orthanc Instance ID
        
    Returns:
        Путь к временному DICOM файлу
    """
    # Проверяем, является ли это SOPInstanceUID (содержит точки)
    if '.' in instance_uid:
        logger.info(f"Detected SOPInstanceUID format, converting to Orthanc ID...")
        try:
            orthanc_id = find_orthanc_id_by_uid(instance_uid, level="Instance")
        except Exception as e:
            logger.error(f"Failed to find Orthanc ID for SOPInstanceUID {instance_uid}: {e}")
            raise
    else:
        orthanc_id = instance_uid
    
    # Скачиваем DICOM файл
    url = f"{settings.ORTHANC_URL}/instances/{orthanc_id}/file"
    logger.info(f"Downloading instance from {url}")
    
    resp = requests.get(url, auth=(settings.ORTHANC_USER, settings.ORTHANC_PASSWORD))
    resp.raise_for_status()
    
    # Сохраняем во временный файл
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".dcm")
    with os.fdopen(tmp_fd, "wb") as tmp_file:
        tmp_file.write(resp.content)
    
    logger.info(f"Instance downloaded to {tmp_path}, size: {len(resp.content)} bytes")
    return tmp_path

def fetch_series_temp(series_uid: str) -> str:
    """
    Скачивает все DICOM файлы Series из Orthanc в виде архива.
    Работает напрямую с SeriesInstanceUID без необходимости знать Study ID.
    
    Args:
        series_uid: SeriesInstanceUID или Orthanc Series ID
        
    Returns:
        Путь к временному архиву с DICOM файлами серии
    """
    # Проверяем, является ли это SeriesInstanceUID (содержит точки)
    if '.' in series_uid:
        logger.info(f"Detected SeriesInstanceUID format, converting to Orthanc ID...")
        try:
            orthanc_id = find_orthanc_id_by_uid(series_uid, level="Series")
        except Exception as e:
            logger.error(f"Failed to find Orthanc ID for SeriesInstanceUID {series_uid}: {e}")
            raise
    else:
        orthanc_id = series_uid
    
    # Скачиваем архив серии
    url = f"{settings.ORTHANC_URL}/series/{orthanc_id}/archive"
    logger.info(f"Downloading series from {url}")
    
    resp = requests.get(url, auth=(settings.ORTHANC_USER, settings.ORTHANC_PASSWORD))
    resp.raise_for_status()
    
    # Сохраняем во временный файл
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    with os.fdopen(tmp_fd, "wb") as tmp_file:
        tmp_file.write(resp.content)
    
    logger.info(f"Series downloaded to {tmp_path}, size: {len(resp.content)} bytes")
    return tmp_path

def fetch_study_temp(study_id: str) -> str:
    """
    Скачивает исследование из Orthanc во временный файл.
    Поддерживает как Orthanc ID, так и StudyInstanceUID.
    
    Args:
        study_id: Orthanc ID или StudyInstanceUID
        
    Returns:
        Путь к временному файлу с архивом исследования
    """
    # Проверяем, является ли это StudyInstanceUID (содержит точки)
    if '.' in study_id:
        logger.info(f"Detected StudyInstanceUID format, converting to Orthanc ID...")
        try:
            orthanc_id = find_orthanc_id_by_uid(study_id, level="Study")
        except Exception as e:
            logger.error(f"Failed to find Orthanc ID for StudyInstanceUID {study_id}: {e}")
            raise
    else:
        orthanc_id = study_id
    
    # Скачиваем архив исследования
    url = f"{settings.ORTHANC_URL}/studies/{orthanc_id}/archive"
    logger.info(f"Downloading study from {url}")
    
    resp = requests.get(url, auth=(settings.ORTHANC_USER, settings.ORTHANC_PASSWORD))
    resp.raise_for_status()
    
    # Сохраняем во временный файл
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    with os.fdopen(tmp_fd, "wb") as tmp_file:
        tmp_file.write(resp.content)
    
    logger.info(f"Study downloaded to {tmp_path}, size: {len(resp.content)} bytes")
    return tmp_path

def get_instance_info(instance_uid: str) -> dict:
    """
    Получает информацию об Instance из Orthanc
    
    Args:
        instance_uid: SOPInstanceUID или Orthanc Instance ID
        
    Returns:
        Словарь с информацией об instance, включая parent Study и Series
    """
    # Конвертируем в Orthanc ID если нужно
    if '.' in instance_uid:
        orthanc_id = find_orthanc_id_by_uid(instance_uid, level="Instance")
    else:
        orthanc_id = instance_uid
    
    url = f"{settings.ORTHANC_URL}/instances/{orthanc_id}"
    resp = requests.get(url, auth=(settings.ORTHANC_USER, settings.ORTHANC_PASSWORD))
    resp.raise_for_status()
    
    instance_info = resp.json()
    
    # Добавляем информацию о parent структурах
    result = {
        'orthanc_id': orthanc_id,
        'sop_instance_uid': instance_info.get('MainDicomTags', {}).get('SOPInstanceUID'),
        'parent_series': instance_info.get('ParentSeries'),
        'parent_study': instance_info.get('ParentStudy'),
        'instance_number': instance_info.get('MainDicomTags', {}).get('InstanceNumber'),
        'main_dicom_tags': instance_info.get('MainDicomTags', {})
    }
    
    return result

def list_all_studies() -> list:
    """
    Получает список всех исследований в Orthanc
    
    Returns:
        Список словарей с информацией об исследованиях
    """
    url = f"{settings.ORTHANC_URL}/studies"
    resp = requests.get(url, auth=(settings.ORTHANC_USER, settings.ORTHANC_PASSWORD))
    resp.raise_for_status()
    
    study_ids = resp.json()
    studies = []
    
    for study_id in study_ids:
        url = f"{settings.ORTHANC_URL}/studies/{study_id}"
        resp = requests.get(url, auth=(settings.ORTHANC_USER, settings.ORTHANC_PASSWORD))
        study_info = resp.json()
        
        studies.append({
            'orthanc_id': study_id,
            'study_instance_uid': study_info.get('MainDicomTags', {}).get('StudyInstanceUID'),
            'study_date': study_info.get('MainDicomTags', {}).get('StudyDate'),
            'study_description': study_info.get('MainDicomTags', {}).get('StudyDescription'),
            'patient_name': study_info.get('PatientMainDicomTags', {}).get('PatientName'),
            'patient_id': study_info.get('PatientMainDicomTags', {}).get('PatientID')
        })
    
    return studies
