#!/usr/bin/env python3
"""
Простой тест для проверки импорта функции создания BMP.
"""

import sys
from pathlib import Path

# Добавляем текущую директорию в путь для импорта
sys.path.append('.')

try:
    print("Пытаемся импортировать функцию...")
    from run import create_sagittal_bmp_images
    print("✓ Функция create_sagittal_bmp_images успешно импортирована")
except ImportError as e:
    print(f"✗ Ошибка импорта: {e}")
    
try:
    print("Пытаемся импортировать setup_logger...")
    from run import setup_logger
    print("✓ Функция setup_logger успешно импортирована")
except ImportError as e:
    print(f"✗ Ошибка импорта: {e}")

# Проверяем, какие функции доступны в модуле run
try:
    import run
    print(f"Доступные функции в модуле run:")
    for name in dir(run):
        if name.startswith('create_') or name.startswith('setup_'):
            print(f"  - {name}")
except Exception as e:
    print(f"Ошибка при проверке модуля run: {e}") 