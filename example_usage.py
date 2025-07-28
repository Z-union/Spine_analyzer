"""
Примеры использования рефакторенной архитектуры анализатора позвоночника.
"""

import logging
from pathlib import Path

from spine_processor import initialize_models, process_study
from orthanc_client import create_orthanc_client, download_study_from_orthanc, read_dicoms_in_folder
from triton_client import initialize_triton_models


def example_local_models_folder():
    """
    Пример использования с локальными моделями и папкой DICOM файлов.
    """
    print("=== Пример 1: Локальные модели + папка ===")
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Инициализация моделей (один раз)
        logger.info("Инициализация локальных моделей...")
        models = initialize_models()
        ax_model, sag_step_1_model, sag_step_2_model, grading_model = models
        logger.info("Модели загружены")
        
        # 2. Получение исследования из папки
        study_folder = "ST000000"
        logger.info(f"Загрузка исследования из папки: {study_folder}")
        studies = read_dicoms_in_folder(study_folder)
        logger.info(f"Загружено {len(studies)} DICOM файлов")
        
        # 3. Обработка исследования
        logger.info("Обработка исследования...")
        results = process_study(studies, ax_model, sag_step_1_model, sag_step_2_model, grading_model, logger)
        
        # 4. Вывод результатов
        if 'error' in results:
            logger.error(f"Ошибка: {results['error']}")
        else:
            logger.info(f"Время обработки: {results.get('processing_time', 'N/A')} сек")
            logger.info(f"Обработано дисков: {results.get('processed_disks', 'N/A')}")
            
            if 'disk_results' in results:
                for disk_label, disk_result in results['disk_results'].items():
                    if 'error' not in disk_result:
                        logger.info(f"Диск {disk_label}: Pfirrman={disk_result.get('pfirrman', 'N/A')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Ошибка в примере 1: {e}")
        return None


def example_local_models_orthanc():
    """
    Пример использования с локальными моделями и Orthanc.
    """
    print("=== Пример 2: Локальные модели + Orthanc ===")
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Инициализация моделей (один раз)
        logger.info("Инициализация локальных моделей...")
        models = initialize_models()
        ax_model, sag_step_1_model, sag_step_2_model, grading_model = models
        logger.info("Модели загружены")
        
        # 2. Создание клиента Orthanc
        ORTHANC_URL = "http://158.160.109.200:8042"
        ORTHANC_USERNAME = "admin"
        ORTHANC_PASSWORD = "mypassword"
        
        logger.info(f"Подключение к Orthanc: {ORTHANC_URL}")
        orthanc_client = create_orthanc_client(ORTHANC_URL, ORTHANC_USERNAME, ORTHANC_PASSWORD)
        
        # 3. Получение исследования из Orthanc
        study_uid = '1.2.840.113619.6.44.320724410489713965087995388011340888881'
        logger.info(f"Загрузка исследования: {study_uid}")
        studies = download_study_from_orthanc(study_uid, orthanc_client)
        logger.info(f"Загружено {len(studies)} DICOM файлов")
        
        # 4. Обработка исследования
        logger.info("Обработка исследования...")
        results = process_study(studies, ax_model, sag_step_1_model, sag_step_2_model, grading_model, logger)
        
        # 5. Вывод результатов
        if 'error' in results:
            logger.error(f"Ошибка: {results['error']}")
        else:
            logger.info(f"Время обработки: {results.get('processing_time', 'N/A')} сек")
            logger.info(f"Обработано дисков: {results.get('processed_disks', 'N/A')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Ошибка в примере 2: {e}")
        return None


def example_triton_models():
    """
    Пример использования с Triton Inference Server.
    """
    print("=== Пример 3: Triton Inference Server ===")
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Инициализация Triton клиента
        logger.info("Подключение к Triton Inference Server...")
        triton_client, model_names = initialize_triton_models("localhost:8000")
        logger.info("Triton клиент инициализирован")
        logger.info(f"Доступные модели: {list(model_names.keys())}")
        
        # 2. Получение исследования
        study_folder = "ST000000"
        logger.info(f"Загрузка исследования из папки: {study_folder}")
        studies = read_dicoms_in_folder(study_folder)
        logger.info(f"Загружено {len(studies)} DICOM файлов")
        
        # 3. Обработка с Triton (пока заглушка)
        logger.warning("Полная интеграция с Triton пока не реализована")
        logger.info("Для полной реализации нужно адаптировать process_study для работы с Triton")
        
        # Пример тестового вызова
        import numpy as np
        test_data = np.random.rand(1, 1, 64, 64, 64).astype(np.float32)
        
        # Тест сегментации
        seg_result = triton_client.infer_segmentation(
            model_names["ax_model"], 
            test_data
        )
        if seg_result is not None:
            logger.info(f"Тест сегментации успешен, форма: {seg_result.shape}")
        
        # Тест grading
        grading_result = triton_client.infer_grading(
            model_names["grading_model"],
            test_data
        )
        if grading_result is not None:
            logger.info(f"Тест grading успешен, результаты: {len(grading_result)} выходов")
        
        return {"status": "triton_test_completed"}
        
    except Exception as e:
        logger.error(f"Ошибка в примере 3: {e}")
        return None


def example_batch_processing():
    """
    Пример пакетной обработки нескольких исследований.
    """
    print("=== Пример 4: Пакетная обработка ===")
    
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Инициализация моделей (один раз для всех исследований)
        logger.info("Инициализация моделей для пакетной обр��ботки...")
        models = initialize_models()
        ax_model, sag_step_1_model, sag_step_2_model, grading_model = models
        logger.info("Модели загружены")
        
        # 2. Список исследований для обработки
        study_folders = ["ST000000"]  # Можно добавить больше папок
        
        results_batch = {}
        
        # 3. Обработка каждого исследования
        for i, study_folder in enumerate(study_folders):
            logger.info(f"Обработка исследования {i+1}/{len(study_folders)}: {study_folder}")
            
            try:
                # Получение исследования
                studies = read_dicoms_in_folder(study_folder)
                logger.info(f"Загружено {len(studies)} DICOM файлов")
                
                # Обработка
                results = process_study(studies, ax_model, sag_step_1_model, sag_step_2_model, grading_model, logger)
                results_batch[study_folder] = results
                
                if 'error' in results:
                    logger.error(f"Ошибка в исследовании {study_folder}: {results['error']}")
                else:
                    logger.info(f"Иссл��дование {study_folder} обработано успешно")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке {study_folder}: {e}")
                results_batch[study_folder] = {"error": str(e)}
        
        # 4. Сводная статистика
        successful = sum(1 for r in results_batch.values() if 'error' not in r)
        failed = len(results_batch) - successful
        
        logger.info(f"Пакетная обработка завершена:")
        logger.info(f"Успешно: {successful}, Ошибок: {failed}")
        
        return results_batch
        
    except Exception as e:
        logger.error(f"Ошибка в пакетной обработке: {e}")
        return None


def main():
    """
    Запуск всех примеров.
    """
    print("Запуск примеров использования рефакторенной архитектуры")
    print("=" * 60)
    
    # Пример 1: Локальные модели + папка
    result1 = example_local_models_folder()
    print()
    
    # Пример 2: Локальные модели + Orthanc (закомментирован, так как требует доступ к серверу)
    # result2 = example_local_models_orthanc()
    # print()
    
    # Пример 3: Triton (закомментирован, так как требует запущенный Triton сервер)
    # result3 = example_triton_models()
    # print()
    
    # Пример 4: Пакетная обработка
    result4 = example_batch_processing()
    print()
    
    print("Примеры завершены")


if __name__ == "__main__":
    main()