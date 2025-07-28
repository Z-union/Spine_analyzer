#!/usr/bin/env python3
"""
Прямой тест создания BMP изображений без импорта из run.py
"""

import sys
from pathlib import Path
import logging
import nibabel
from nibabel.nifti1 import Nifti1Image
from nibabel.loadsave import save
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import traceback

def create_sagittal_bmp_images(nifti_img: Nifti1Image, nifti_seg: Nifti1Image, output_dir: Path, logger=None):
    """
    Создает BMP изображения с подписанными позвонками для каждого среза сагиттальной проекции.
    """
    try:
        # Создаем папку для BMP изображений
        bmp_dir = output_dir / "segments" / "sag"
        bmp_dir.mkdir(parents=True, exist_ok=True)
        
        if logger:
            logger.info(f"Создаем BMP изображения в папке: {bmp_dir}")
        
        # Получаем данные
        img_data = nifti_img.get_fdata()
        seg_data = nifti_seg.get_fdata()
        
        # Словарь для описания позвонков
        vertebra_descriptions = {
            63: 'C2-C3', 64: 'C3-C4', 65: 'C4-C5', 66: 'C5-C6', 67: 'C6-C7',
            71: 'C7-T1', 72: 'T1-T2', 73: 'T2-T3', 74: 'T3-T4', 75: 'T4-T5',
            76: 'T5-T6', 77: 'T6-T7', 78: 'T7-T8', 79: 'T8-T9', 80: 'T9-T10',
            81: 'T10-T11', 82: 'T11-T12', 91: 'T12-L1', 92: 'L1-L2', 93: 'L2-L3',
            94: 'L3-L4', 95: 'L4-L5', 96: 'L5-S1', 100: 'S1-S2'
        }
        
        # Цвета для разных типов структур
        colors = {
            'vertebra': (255, 255, 0),    # Желтый для позвонков
            'disk': (0, 255, 0),          # Зеленый для дисков
            'canal': (0, 0, 255),         # Синий для канала
            'cord': (255, 0, 0),          # Красный для спинного мозга
            'sacrum': (128, 0, 128),      # Фиолетовый для крестца
            'hernia': (255, 165, 0),      # Оранжевый для грыж
            'bulging': (255, 192, 203),   # Розовый для выбуханий
            'background': (0, 0, 0)       # Черный для фона
        }
        
        # Проходим по всем срезам по минимальной оси (обычно это Z-ось)
        num_slices = img_data.shape[2]
        
        if logger:
            logger.info(f"Создаем BMP для {num_slices} срезов сагиттальной проекции")
        
        for slice_idx in range(num_slices):
            try:
                # Получаем срез МРТ и сегментацию
                mri_slice = img_data[:, :, slice_idx]
                seg_slice = seg_data[:, :, slice_idx]
                
                # Нормализуем МРТ срез для отображения
                mri_normalized = ((mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min()) * 255).astype(np.uint8)
                
                # Создаем RGB изображение
                rgb_image = np.stack([mri_normalized, mri_normalized, mri_normalized], axis=2)
                
                # Накладываем сегментацию
                for label_value in np.unique(seg_slice):
                    if label_value == 0:  # Пропускаем фон
                        continue
                    
                    # Определяем цвет в зависимости от метки
                    if label_value in vertebra_descriptions:
                        color = colors['vertebra']
                    elif label_value in [1, 2]:  # Канал и спинной мозг
                        color = colors['canal'] if label_value == 2 else colors['cord']
                    elif label_value == 50:  # Крестец
                        color = colors['sacrum']
                    elif label_value == 200:  # Грыжа
                        color = colors['hernia']
                    elif label_value == 201:  # Выбухание
                        color = colors['bulging']
                    else:
                        color = colors['disk']  # По умолчанию диск
                    
                    # Создаем маску для этой метки
                    mask = (seg_slice == label_value)
                    if np.any(mask):
                        # Накладываем цвет на маску
                        for i in range(3):
                            rgb_image[:, :, i][mask] = color[i]
                
                # Конвертируем в PIL Image для добавления текста
                pil_image = Image.fromarray(rgb_image)
                draw = ImageDraw.Draw(pil_image)
                
                # Пытаемся загрузить шрифт, если не получится - используем стандартный
                try:
                    font = ImageFont.truetype("arial.ttf", 12)
                except:
                    font = ImageFont.load_default()
                
                # Добавляем подписи для найденных позвонков
                found_vertebrae = []
                for label_value in np.unique(seg_slice):
                    if label_value in vertebra_descriptions:
                        # Находим центр позвонка
                        mask = (seg_slice == label_value)
                        if np.any(mask):
                            coords = np.argwhere(mask)
                            center_y, center_x = coords.mean(axis=0).astype(int)
                            
                            # Добавляем подпись
                            description = vertebra_descriptions[label_value]
                            text_bbox = draw.textbbox((0, 0), description, font=font)
                            text_width = text_bbox[2] - text_bbox[0]
                            text_height = text_bbox[3] - text_bbox[1]
                            
                            # Позиция текста (смещение от центра)
                            text_x = max(0, center_x - text_width // 2)
                            text_y = max(0, center_y - text_height // 2)
                            
                            # Рисуем фон для текста
                            draw.rectangle([text_x-2, text_y-2, text_x+text_width+2, text_y+text_height+2], 
                                         fill=(0, 0, 0), outline=(255, 255, 255))
                            
                            # Рисуем текст
                            draw.text((text_x, text_y), description, fill=(255, 255, 255), font=font)
                            found_vertebrae.append(description)
                
                # Добавляем информацию о срезе
                slice_info = f"Slice {slice_idx+1}/{num_slices}"
                draw.text((10, 10), slice_info, fill=(255, 255, 255), font=font)
                
                if found_vertebrae:
                    vertebrae_info = f"Found: {', '.join(found_vertebrae)}"
                    draw.text((10, 30), vertebrae_info, fill=(255, 255, 255), font=font)
                
                # Сохраняем BMP
                bmp_filename = f"sagittal_slice_{slice_idx+1:03d}.bmp"
                bmp_path = bmp_dir / bmp_filename
                pil_image.save(bmp_path, 'BMP')
                
                if logger and slice_idx % 10 == 0:  # Логируем каждые 10 срезов
                    logger.info(f"Сохранен срез {slice_idx+1}/{num_slices}: {bmp_filename}")
                
            except Exception as e:
                if logger:
                    logger.error(f"Ошибка при создании BMP для среза {slice_idx}: {e}")
                continue
        
        if logger:
            logger.info(f"Создано {num_slices} BMP изображений в папке {bmp_dir}")
            
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при создании BMP изображений: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

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
        
        print("Начинаем тестирование создания BMP изображений")
        
        # Создаем тестовые данные
        print("Создаем тестовые NIfTI данные")
        nifti_img, nifti_seg = create_test_data()
        
        # Сохраняем тестовые данные
        save(nifti_img, str(output_dir / "test_mri.nii.gz"))
        save(nifti_seg, str(output_dir / "test_seg.nii.gz"))
        print("Тестовые данные сохранены")
        
        # Создаем BMP изображения
        print("Создаем BMP изображения с подписанными позвонками")
        create_sagittal_bmp_images(nifti_img, nifti_seg, output_dir, None)
        
        print("Тестирование завершено успешно!")
        print(f"Результаты сохранены в папке: {output_dir}")
        
        # Проверяем, что файлы созданы
        bmp_dir = output_dir / "segments" / "sag"
        if bmp_dir.exists():
            bmp_files = list(bmp_dir.glob("*.bmp"))
            print(f"Создано {len(bmp_files)} BMP файлов")
            for bmp_file in bmp_files[:5]:  # Показываем первые 5 файлов
                print(f"  - {bmp_file.name}")
        else:
            print("Папка с BMP файлами не создана!")
            
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 