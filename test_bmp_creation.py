#!/usr/bin/env python3
"""
Тестовый скрипт для проверки создания BMP изображений с подписанными позвонками.
"""

import sys
from pathlib import Path
import logging
import nibabel
from nibabel.nifti1 import Nifti1Image
from nibabel.loadsave import save
import numpy as np

# Добавляем текущую директорию в путь для импорта
sys.path.append('.')

from run import create_sagittal_bmp_images, setup_logger

def create_test_data():
    """
    Создает тестовые NIfTI данные для проверки функции создания BMP.
    """
    # Создаем тестовые данные
    shape = (256, 256, 64)  # Размер тестового изображения
    
    # Создаем тестовое МРТ изображение
    mri_data = np.random.rand(*shape) * 1000 + 500  # Симуляция МРТ данных
    
    # Создаем тестовую сегментацию
    seg_data = np.zeros(shape, dtype=np.uint8)
    
    # Добавляем несколько позвонков в разных срезах
    vertebra_labels = [63, 64, 65, 66, 67, 71, 72, 73, 74, 75]  # C2-C3 до T4-T5
    vertebra_descriptions = {
        63: 'C2-C3', 64: 'C3-C4', 65: 'C4-C5', 66: 'C5-C6', 67: 'C6-C7',
        71: 'C7-T1', 72: 'T1-T2', 73: 'T2-T3', 74: 'T3-T4', 75: 'T4-T5'
    }
    
    # Создаем эллиптические области для позвонков
    for i, label in enumerate(vertebra_labels):
        # Размещаем позвонки в разных срезах
        slice_idx = 20 + i * 4  # Каждый позвонок в своем срезе
        
        if slice_idx < shape[2]:
            # Создаем эллиптическую область для позвонка
            center_y, center_x = shape[0] // 2, shape[1] // 2
            radius_y, radius_x = 30, 20
            
            y, x = np.ogrid[:shape[0], :shape[1]]
            mask = ((x - center_x) ** 2 / radius_x ** 2 + 
                   (y - center_y) ** 2 / radius_y ** 2 <= 1)
            
            seg_data[mask, slice_idx] = label
    
    # Добавляем канал и спинной мозг
    for slice_idx in range(shape[2]):
        center_y, center_x = shape[0] // 2, shape[1] // 2
        radius_y, radius_x = 15, 10
        
        y, x = np.ogrid[:shape[0], :shape[1]]
        canal_mask = ((x - center_x) ** 2 / radius_x ** 2 + 
                     (y - center_y) ** 2 / radius_y ** 2 <= 1)
        
        seg_data[canal_mask, slice_idx] = 2  # Канал
        
        # Спинной мозг внутри канала
        radius_y, radius_x = 8, 5
        cord_mask = ((x - center_x) ** 2 / radius_x ** 2 + 
                    (y - center_y) ** 2 / radius_y ** 2 <= 1)
        
        seg_data[cord_mask, slice_idx] = 1  # Спинной мозг
    
    # Создаем NIfTI изображения
    affine = np.eye(4)
    header = nibabel.nifti1.Nifti1Header()
    
    nifti_img = Nifti1Image(mri_data, affine, header)
    nifti_seg = Nifti1Image(seg_data, affine, header)
    
    return nifti_img, nifti_seg

def main():
    """
    Основная функция для тестирования создания BMP изображений.
    """
    try:
        # Создаем выходную папку
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        # Настраиваем logger
        logger = setup_logger(output_dir, "test_bmp.log")
        logger.info("Начинаем тестирование создания BMP изображений")
        
        # Создаем тестовые данные
        logger.info("Создаем тестовые NIfTI данные")
        nifti_img, nifti_seg = create_test_data()
        
        # Сохраняем тестовые данные
        save(nifti_img, str(output_dir / "test_mri.nii.gz"))
        save(nifti_seg, str(output_dir / "test_seg.nii.gz"))
        logger.info("Тестовые данные сохранены")
        
        # Создаем BMP изображения
        logger.info("Создаем BMP изображения с подписанными позвонками")
        create_sagittal_bmp_images(nifti_img, nifti_seg, output_dir, logger)
        
        logger.info("Тестирование завершено успешно!")
        logger.info(f"Результаты сохранены в папке: {output_dir}")
        
        # Проверяем, что файлы созданы
        bmp_dir = output_dir / "segments" / "sag"
        if bmp_dir.exists():
            bmp_files = list(bmp_dir.glob("*.bmp"))
            logger.info(f"Создано {len(bmp_files)} BMP файлов")
            for bmp_file in bmp_files[:5]:  # Показываем первые 5 файлов
                logger.info(f"  - {bmp_file.name}")
        else:
            logger.error("Папка с BMP файлами не создана!")
            
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 