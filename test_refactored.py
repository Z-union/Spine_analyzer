"""
Тестовый скрипт для проверки рефакторенной архитектуры.
"""

import sys
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Тестирует импорт всех модулей."""
    logger.info("Тестирование импортов...")
    
    try:
        from spine_processor import initialize_models, process_study
        logger.info("✓ spine_processor импортирован успешно")
    except ImportError as e:
        logger.error(f"✗ Ошибка импорта spine_processor: {e}")
        return False
    
    try:
        from orthanc_client import (
            create_orthanc_client, 
            download_study_from_orthanc, 
            read_dicoms_in_folder
        )
        logger.info("✓ orthanc_client импортирован успешно")
    except ImportError as e:
        logger.error(f"✗ Ошибка импорта orthanc_client: {e}")
        return False
    
    try:
        from triton_client import initialize_triton_models, TritonModelClient
        logger.info("✓ triton_client импортирован успешно")
    except ImportError as e:
        logger.warning(f"⚠ triton_client не импортирован (нормально если Triton не установлен): {e}")
    
    return True


def test_folder_reading():
    """Тестирует чтение DICOM файлов из папки."""
    logger.info("Тестирование чтения из папки...")
    
    try:
        from orthanc_client import read_dicoms_in_folder
        
        # Проверяем наличие тестовой папки
        test_folder = "ST000000"
        if not Path(test_folder).exists():
            logger.warning(f"⚠ Тестовая папка {test_folder} не найдена, пропускаем тест")
            return True
        
        studies = read_dicoms_in_folder(test_folder)
        logger.info(f"✓ Успешно прочитано {len(studies)} DICOM файлов из {test_folder}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Ошибка чтения из папки: {e}")
        return False


def test_model_initialization():
    """Тестирует инициализацию моделей."""
    logger.info("Тестирование инициализации моделей...")
    
    try:
        from spine_processor import initialize_models
        
        # Проверяем наличие файлов моделей
        model_paths = [
            'model/weights/ax.pth',
            'model/weights/sag_step_1.pth', 
            'model/weights/sag_step_2.pth',
            'model/weights/grading.pth'
        ]
        
        missing_models = []
        for path in model_paths:
            if not Path(path).exists():
                missing_models.append(path)
        
        if missing_models:
            logger.warning(f"⚠ Отсутствуют файлы моделей: {missing_models}")
            logger.warning("⚠ Пропускаем тест инициализации моделей")
            return True
        
        models = initialize_models()
        logger.info("✓ Модели инициализированы успешно")
        
        # Проверяем, что все модели загружены
        ax_model, sag_step_1_model, sag_step_2_model, grading_model = models
        assert ax_model is not None
        assert sag_step_1_model is not None
        assert sag_step_2_model is not None
        assert grading_model is not None
        
        logger.info("✓ Все модели загружены корректно")
        return True
        
    except Exception as e:
        logger.error(f"✗ Ошибка инициализации моделей: {e}")
        return False


def test_triton_connection():
    """Тестирует подключение к Triton (если доступен)."""
    logger.info("Тестирование подключения к Triton...")
    
    try:
        from triton_client import initialize_triton_models
        
        # Пытаемся подключиться к локальному Triton
        triton_client, model_names = initialize_triton_models("localhost:8000")
        
        if triton_client.is_server_ready():
            logger.info("✓ Triton сервер доступен")
            logger.info(f"✓ Модели: {list(model_names.keys())}")
            return True
        else:
            logger.warning("⚠ Triton сервер недоступен (нормально если не запуще��)")
            return True
            
    except Exception as e:
        logger.warning(f"⚠ Triton недоступен (нормально если не установлен): {e}")
        return True


def test_main_script():
    """Тестирует основной скрипт."""
    logger.info("Тестирование основного скрипта...")
    
    try:
        import main_refactored
        logger.info("✓ main_refactored импортирован успешно")
        
        # Тестируем парсинг аргументов
        args = main_refactored.parse_args()
        logger.info("✓ Парсинг аргументов работает")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Ошибка в основном скрипте: {e}")
        return False


def run_all_tests():
    """Запускает все тесты."""
    logger.info("=" * 50)
    logger.info("Запуск тестов рефакторенной архитектуры")
    logger.info("=" * 50)
    
    tests = [
        ("Импорты", test_imports),
        ("Чтение из папки", test_folder_reading),
        ("Инициализация моделей", test_model_initialization),
        ("Подклю��ение к Triton", test_triton_connection),
        ("Основной скрипт", test_main_script),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ Критическая ошибка в тесте '{test_name}': {e}")
            results.append((test_name, False))
    
    # Сводка результатов
    logger.info("\n" + "=" * 50)
    logger.info("СВОДКА РЕЗУЛЬТАТОВ")
    logger.info("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, result in results:
        status = "✓ ПРОЙДЕН" if result else "✗ ПРОВАЛЕН"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    logger.info(f"\nИтого: {passed} пройдено, {failed} провалено")
    
    if failed == 0:
        logger.info("🎉 Все тесты пройдены успешно!")
        return True
    else:
        logger.warning(f"⚠ {failed} тестов провалено")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)