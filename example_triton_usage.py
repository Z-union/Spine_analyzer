"""
Пример использования рефакторенной архитектуры с Triton Inference Server.
"""

import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_triton_vs_local():
    """
    Демонстрирует разницу между локальными моделями и Triton.
    """
    print("=== Сравнение: Локальные модели vs Triton ===")
    
    # Получение данных (одинаково для обоих подходов)
    from orthanc_client import read_dicoms_in_folder
    
    study_folder = "ST000000"
    logger.info(f"Загрузка исследования из папки: {study_folder}")
    
    try:
        studies = read_dicoms_in_folder(study_folder)
        logger.info(f"Загружено {len(studies)} DICOM файлов")
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        return
    
    # === ВАРИАНТ 1: Локальные модели ===
    print("\n--- Локальные модели ---")
    try:
        from spine_processor import initialize_models, process_study
        
        # Инициализация моделей (один раз)
        logger.info("Инициализация локальных моделей...")
        models = initialize_models()
        ax_model, sag_step_1_model, sag_step_2_model, grading_model = models
        logger.info("✓ Локальные модели загружены")
        
        # Обработка
        logger.info("Обработка с локальными моделями...")
        results_local = process_study(studies, ax_model, sag_step_1_model, sag_step_2_model, grading_model, logger)
        
        if 'error' in results_local:
            logger.error(f"Ошибка локальной обработки: {results_local['error']}")
        else:
            logger.info(f"✓ Локальная обработка завершена за {results_local.get('processing_time', 'N/A')} сек")
            logger.info(f"✓ Обработано дисков: {results_local.get('processed_disks', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Ошибка с локальными моделями: {e}")
        results_local = None
    
    # === ВАРИАНТ 2: Triton Inference Server ===
    print("\n--- Triton Inference Server ---")
    try:
        from triton_client import initialize_triton_models
        from spine_processor_triton import process_study_triton
        
        # Инициализация Triton клиента (один раз)
        logger.info("Подключение к Triton Inference Server...")
        triton_client, model_names = initialize_triton_models("localhost:8000")
        logger.info("✓ Triton клиент инициализирован")
        logger.info(f"✓ Доступные модели: {list(model_names.keys())}")
        
        # Обработка
        logger.info("Обработка через Triton...")
        results_triton = process_study_triton(studies, triton_client, model_names, logger)
        
        if 'error' in results_triton:
            logger.error(f"Ошибка Triton обработки: {results_triton['error']}")
        else:
            logger.info(f"✓ Triton обработка завершена за {results_triton.get('processing_time', 'N/A')} сек")
            logger.info(f"✓ Обработано дисков: {results_triton.get('processed_disks', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Ошибка с Triton (нормально если сервер не запущен): {e}")
        results_triton = None
    
    # === СРАВНЕНИЕ РЕЗУЛЬТАТОВ ===
    print("\n--- Сравнение результатов ---")
    
    if results_local and results_triton:
        if 'error' not in results_local and 'error' not in results_triton:
            local_time = results_local.get('processing_time', 0)
            triton_time = results_triton.get('processing_time', 0)
            
            logger.info(f"Время локальной обработки: {local_time:.2f} сек")
            logger.info(f"Время Triton обработки: {triton_time:.2f} сек")
            
            if triton_time > 0 and local_time > 0:
                speedup = local_time / triton_time
                logger.info(f"Ускорение: {speedup:.2f}x")
    
    return results_local, results_triton


def example_api_ready_code():
    """
    Показывает, как код готов для интеграции с FastAPI.
    """
    print("\n=== Готовность к FastAPI ===")
    
    # Псевдокод FastAPI приложения
    fastapi_code = '''
from fastapi import FastAPI, Depends
from spine_processor import initialize_models, process_study
from spine_processor_triton import process_study_triton
from triton_client import initialize_triton_models
from orthanc_client import create_orthanc_client, download_study_from_orthanc

app = FastAPI()

# Глобальные переменные для кэширования
local_models = None
triton_client = None
triton_model_names = None
orthanc_client = None

@app.on_event("startup")
async def startup_event():
    global local_models, triton_client, triton_model_names, orthanc_client
    
    # Инициализация моделей (один раз при старте)
    local_models = initialize_models()
    
    # Инициализация Triton (один раз при старте)
    triton_client, triton_model_names = initialize_triton_models("localhost:8000")
    
    # Инициализация Orthanc клиента (один раз при старте)
    orthanc_client = create_orthanc_client("http://orthanc:8042", "admin", "password")

@app.post("/analyze_study_local")
async def analyze_study_local(study_uid: str):
    """Анализ с локальными моделями"""
    studies = download_study_from_orthanc(study_uid, orthanc_client)
    results = process_study(studies, *local_models)
    return results

@app.post("/analyze_study_triton")
async def analyze_study_triton(study_uid: str):
    """Анализ через Triton"""
    studies = download_study_from_orthanc(study_uid, orthanc_client)
    results = process_study_triton(studies, triton_client, triton_model_names)
    return results

@app.post("/analyze_study_folder")
async def analyze_study_folder(folder_path: str, use_triton: bool = False):
    """Анализ из папки"""
    from orthanc_client import read_dicoms_in_folder
    studies = read_dicoms_in_folder(folder_path)
    
    if use_triton:
        results = process_study_triton(studies, triton_client, triton_model_names)
    else:
        results = process_study(studies, *local_models)
    
    return results
'''
    
    print("Пример FastAPI интеграции:")
    print(fastapi_code)
    
    logger.info("✓ Код готов для FastAPI интеграции")
    logger.info("✓ Модели инициализируются один раз при старте")
    logger.info("✓ Клиенты кэшируются и переиспользуются")
    logger.info("✓ Легко переключаться между локальными моделями и Triton")


def main():
    """
    Запуск всех примеров.
    """
    print("Демонстрация рефакторенной архитектуры")
    print("=" * 60)
    
    # Пример 1: Сравнение локальных моделей и Triton
    try:
        results_local, results_triton = example_triton_vs_local()
    except Exception as e:
        logger.error(f"Ошибка в примере сравнения: {e}")
    
    # Пример 2: Готовность к FastAPI
    try:
        example_api_ready_code()
    except Exception as e:
        logger.error(f"Ошибка в примере FastAPI: {e}")
    
    print("\n" + "=" * 60)
    print("Ключевые преимущества рефакторенной архитектуры:")
    print("1. ✓ Модели загружаются один раз, а не на каждый запрос")
    print("2. ✓ Легко заменить источник данных (папка ↔ Orthanc)")
    print("3. ✓ Простое переключение между локальными моделями и Triton")
    print("4. ✓ Готовность к FastAPI и горизонтальному масштабированию")
    print("5. ✓ Четкое разделение ответственности")


if __name__ == "__main__":
    main()