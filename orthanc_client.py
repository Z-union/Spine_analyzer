"""
Модуль для работы с Orthanc через DICOMweb API.
"""

from typing import List, Union
from dicomweb_client.api import DICOMwebClient
import pydicom.dataset
from requests.auth import HTTPBasicAuth
from dicomweb_client.session_utils import create_session_from_auth
import glob
from pydicom import dcmread
import os
import pydicom


def download_study_from_orthanc(
    study_instance_uid: str, 
    client: DICOMwebClient
) -> List[pydicom.dataset.FileDataset]:
    """
    Загружает исследование из Orthanc по Study Instance UID.
    
    Args:
        study_instance_uid: UID исследования
        client: DICOMwebClient для подключения к Orthanc
        
    Returns:
        Список DICOM datasets
        
    Raises:
        Exception: Если не удалось загрузить исследование
    """
    try:
        dicom_files = client.retrieve_study(study_instance_uid)
        
        if dicom_files:
            print(f"DICOMweb success: Retrieved {len(dicom_files)} instances")
            return dicom_files
        else:
            raise Exception("No instances retrieved via DICOMweb")
            
    except Exception as e:
        print(f"DICOMweb failed: {e}")
        raise


def is_dicom_file(path: Union[os.PathLike, str, bytes]) -> bool:
    """
    Проверяет, является ли файл DICOM-файлом.

    Args:
        path: Путь к файлу

    Returns:
        True, если файл DICOM, иначе False
    """
    try:
        dcmread(path)
        return True
    except:
        return False


def read_dicoms_in_folder(study_folder: str) -> List[pydicom.dataset.FileDataset]:
    """
    Читает все DICOM файлы из папки.
    
    Args:
        study_folder: Путь к папке с DICOM файлами
        
    Returns:
        Список DICOM datasets
        
    Raises:
        ValueError: Если в папке не найдено DICOM файлов
    """
    dicom_paths = [
        f for f in glob.glob(os.path.join(study_folder, '**', '*'), recursive=True) 
        if os.path.isfile(f) and is_dicom_file(f)
    ]
    
    if not dicom_paths:
        raise ValueError(f"В папке {study_folder} не найдено DICOM-файлов")

    dicoms = [dcmread(f, stop_before_pixels=True) for f in dicom_paths]
    return dicoms


def create_orthanc_client(
    orthanc_url: str,
    username: str,
    password: str
) -> DICOMwebClient:
    """
    Создает DICOMwebClient для подключения к Orthanc.
    
    Args:
        orthanc_url: URL Orthanc сервера
        username: Имя пользователя
        password: Пароль
        
    Returns:
        Настроенный DICOMwebClient
    """
    auth = HTTPBasicAuth(username, password)
    session = create_session_from_auth(auth)
    client = DICOMwebClient(f"{orthanc_url}/dicom-web", session=session)
    return client


# Пример использования
if __name__ == "__main__":
    # Настройки Orthanc
    ORTHANC_URL = "http://158.160.109.200:8042" 
    ORTHANC_USERNAME = "admin"  
    ORTHANC_PASSWORD = "mypassword"  
    
    # UID исследования для тестирования
    failing_study_uid = '1.2.840.113619.6.44.320724410489713965087995388011340888881'
    
    # Создание клиента
    client = create_orthanc_client(ORTHANC_URL, ORTHANC_USERNAME, ORTHANC_PASSWORD)
    
    # Загрузка из Orthanc
    try:
        dicom_orthanc = download_study_from_orthanc(
            study_instance_uid=failing_study_uid,
            client=client
        )
        print(f"Загружено из Orthanc: {len(dicom_orthanc)} файлов")
    except Exception as e:
        print(f"Ошибка загрузки из Orthanc: {e}")
    
    # Загрузка из папки
    try:
        dicoms_folder = read_dicoms_in_folder("ST000000")
        print(f"Загружено из папки: {len(dicoms_folder)} файлов")
    except Exception as e:
        print(f"Ошибка загрузки из папки: {e}")