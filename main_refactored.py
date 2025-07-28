"""
Рефакторенная версия основного скрипта для анализа позвоночника.
Разделяет инициализацию моделей, получение данных и обработку исследований.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

import pydicom.dataset

from spine_processor import initialize_models, process_study
from orthanc_client import (
    create_orthanc_client, 
    download_study_from_orthanc, 
    read_dicoms_in_folder
)
try:
    from triton_client import initialize_triton_models, TritonModelClient
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    initialize_triton_models = None
    TritonModelClient = None


def setup_logger(output_dir: Path, log_filename: str = "spine_analysis.log") -> logging.Logger:
    """
    Настраивает logger для записи в файл и консоль.
    
    Args:
        output_dir: Папка для сохранения лог-файла
        log_filename: Имя лог-файла
        
    Returns:
        Настроенный logger
    """
    log_path = output_dir / log_filename

    # Создаем logger
    logger = logging.getLogger('spine_analyzer')
    logger.setLevel(logging.INFO)

    # Очищаем существующие handlers
    logger.handlers.clear()

    # Создаем formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler для файла
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Handler для консоли
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Добавляем handlers к logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_study_from_folder(study_folder: str) -> List[pydicom.dataset.FileDataset]:
    """
    Получает исследование из папки с DICOM файлами.
    
    Args:
        study_folder: Путь к папке с DICOM файлами
        
    Returns:
        Список DICOM datasets
    """
    return read_dicoms_in_folder(study_folder)


def get_study_from_orthanc(study_instance_uid: str, client) -> List[pydicom.dataset.FileDataset]:
    """
    Получает исследование из Orthanc по Study Instance UID.
    
    Args:
        study_instance_uid: UID исследования
        client: DICOMwebClient для подключения к Orthanc
        
    Returns:
        Список DICOM datasets
    """
    return download_study_from_orthanc(study_instance_uid, client)


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Рефакторенный анализатор позвоночника с поддержкой Triton Inference Server"
    )
    
    # Источник данных
    parser.add_argument(
        '--source', 
        choices=['folder', 'orthanc'], 
        default='folder',
        help='Источник DICOM данных'
    )
    
    parser.add_argument(
        '--study_folder', 
        type=str, 
        default='ST000000',
        help='Папка с DICOM файлами (для source=folder)'
    )
    
    parser.add_argument(
        '--study_uid', 
        type=str,
        help='Study Instance UID (для source=orthanc)'
    )
    
    # Настройки Orthanc
    parser.add_argument(
        '--orthanc_url', 
        type=str, 
        default='http://158.160.109.200:8042',
        help='URL Orthanc сервера'
    )
    
    parser.add_argument(
        '--orthanc_username', 
        type=str, 
        default='admin',
        help='Имя пользователя Orthanc'
    )
    
    parser.add_argument(
        '--orthanc_password', 
        type=str, 
        default='mypassword',
        help='Пароль Orthanc'
    )
    
    # Настройки моделей
    parser.add_argument(
        '--use_triton', 
        action='store_true',
        help='Использовать Triton Inference Server вместо локальных моделей'
    )
    
    parser.add_argument(
        '--triton_url', 
        type=str, 
        default='localhost:8000',
        help='URL Triton Inference Server'
    )
    
    # Выходная папка
    parser.add_argument(
        '--output', 
        type=Path, 
        default=Path('./results'),
        help='Папка для сохранения результатов'
    )
    
    return parser.parse_args()


def main():
    """
    Основная функция с рефакторенной архитектурой.
    """
    start_time = time.time()
    
    try:
        # Парсинг аргументов
        args = parse_args()
        
        # Создание выходной папки
        args.output.mkdir(exist_ok=True, parents=True)
        
        # Настройка логирования
        logger = setup_logger(args.output)
        logger.info("=== Запуск рефакторенного анализатора позвоночника ===")
        logger.info(f"Источник данных: {args.source}")
        logger.info(f"Использование Triton: {args.use_triton}")
        logger.info(f"Выходная папка: {args.output}")
        
        # 1. Инициализация моделей (один раз)
        logger.info("Инициализация моделей...")
        
        if args.use_triton:
            # Использование Triton Inference Server
            logger.info(f"Подключение к Triton Inference Server: {args.triton_url}")
            triton_client, model_names = initialize_triton_models(args.triton_url)
            models = (triton_client, model_names)
            logger.info("Triton модели инициализированы")
        else:
            # Использование локальных моделей
            logger.info("Загрузка локальных моделей...")
            models = initialize_models()
            logger.info("Локальные модели загружены")
        
        # 2. Получение исследования
        logger.info("Получение исследования...")
        
        if args.source == 'folder':
            logger.info(f"Загрузка из папки: {args.study_folder}")
            studies = get_study_from_folder(args.study_folder)
        elif args.source == 'orthanc':
            if not args.study_uid:
                raise ValueError("Для source=orthanc необходимо указать --study_uid")
            
            logger.info(f"Подключение к Orthanc: {args.orthanc_url}")
            orthanc_client = create_orthanc_client(
                args.orthanc_url, 
                args.orthanc_username, 
                args.orthanc_password
            )
            
            logger.info(f"Загрузка исследования: {args.study_uid}")
            studies = get_study_from_orthanc(args.study_uid, orthanc_client)
        else:
            raise ValueError(f"Неподдерживаемый источник: {args.source}")
        
        logger.info(f"Загружено {len(studies)} DICOM файлов")
        
        # 3. Обработка исследования
        logger.info("Начинаем обработку исследования...")
        
        if args.use_triton:
            # Для Triton нужна адаптированная версия process_study
            # Пока используем заглушку
            logger.warning("Обработка с Triton пока не реализована полностью")
            results = {"error": "Triton processing not implemented yet"}
        else:
            # Обработка с локальными моделями
            ax_model, sag_step_1_model, sag_step_2_model, grading_model = models
            results = process_study(
                studies, 
                ax_model, 
                sag_step_1_model, 
                sag_step_2_model, 
                grading_model, 
                logger
            )
        
        # 4. Сохранение результатов
        logger.info("Сохранение результатов...")
        
        if 'error' in results:
            logger.error(f"Ошибка обработки: {results['error']}")
        else:
            # Сохранение результатов в файл
            import json
            results_file = args.output / "analysis_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Результаты сохранены в: {results_file}")
            
            # Краткая статистика
            if 'disk_results' in results:
                logger.info(f"Обработано дисков: {len(results['disk_results'])}")
                for disk_label, disk_result in results['disk_results'].items():
                    if 'error' not in disk_result:
                        logger.info(f"Диск {disk_label}: Pfirrman={disk_result.get('pfirrman', 'N/A')}")
        
        # Общее время выполнения
        total_time = time.time() - start_time
        logger.info(f"Общее время выполнения: {total_time:.2f} секунд")
        logger.info("=== Анализ завершен ===")
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())