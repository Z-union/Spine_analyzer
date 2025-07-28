#!/usr/bin/env python3
"""
Тестовый скрипт для проверки разных вариаций BMP изображений.
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

def create_sagittal_bmp_images(nifti_img: Nifti1Image, nifti_seg: Nifti1Image, output_dir: Path, logger=None, slice_axis=0, variation=0):
    """
    Создает BMP изображения с подписанными позвонками для каждого среза.
    
    :param nifti_img: NIfTI изображение МРТ
    :param nifti_seg: NIfTI изображение сегментации
    :param output_dir: Папка для сохранения результатов
    :param logger: Logger для записи сообщений
    :param slice_axis: Ось для нарезки (0=X, 1=Y, 2=Z), по умолчанию 0
    :param variation: Вариация отображения (0=контуры, 1=заливка, 2=контуры+заливка)
    """
    try:
        # Создаем папку для BMP изображений
        bmp_dir = output_dir / "segments" / "sag"
        bmp_dir.mkdir(parents=True, exist_ok=True)
        
        if logger:
            logger.info(f"Создаем BMP изображения в папке: {bmp_dir}")
            logger.info(f"Ось нарезки: {slice_axis}, вариация: {variation}")
        
        # Получаем данные
        img_data = nifti_img.get_fdata()
        seg_data = nifti_seg.get_fdata()
        
        # Словарь для описания позвонков (не дисков)
        vertebra_descriptions = {
            13: 'C2', 21: 'C3', 41: 'C4', 50: 'C5',  # Шейные позвонки
            # Добавляем другие позвонки по мере необходимости
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
        
        # Проходим по всем срезам по указанной оси
        num_slices = img_data.shape[slice_axis]
        
        if logger:
            logger.info(f"Создаем BMP для {num_slices} срезов по оси {slice_axis}")
        
        for slice_idx in range(num_slices):
            try:
                # Получаем срез МРТ и сегментации по указанной оси
                if slice_axis == 0:
                    mri_slice = img_data[slice_idx, :, :]
                    seg_slice = seg_data[slice_idx, :, :]
                elif slice_axis == 1:
                    mri_slice = img_data[:, slice_idx, :]
                    seg_slice = seg_data[:, slice_idx, :]
                else:  # slice_axis == 2
                    mri_slice = img_data[:, :, slice_idx]
                    seg_slice = seg_data[:, :, slice_idx]
                
                # Нормализуем МРТ срез для отображения
                mri_normalized = ((mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min()) * 255).astype(np.uint8)
                
                # Поворачиваем изображение: только 90° против часовой стрелки
                # (убираем разворот на 180°, чтобы не было вверх ногами)
                mri_normalized = np.rot90(mri_normalized, k=1)
                seg_slice = np.rot90(seg_slice, k=1)
                
                # Создаем RGB изображение
                rgb_image = np.stack([mri_normalized, mri_normalized, mri_normalized], axis=2)
                
                # Накладываем сегментацию в зависимости от вариации
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
                        if variation == 0:  # Только контуры
                            from scipy.ndimage import binary_erosion
                            eroded_mask = binary_erosion(mask, iterations=1)
                            contour_mask = mask & ~eroded_mask
                            for i in range(3):
                                rgb_image[:, :, i][contour_mask] = color[i]
                        elif variation == 1:  # Только заливка
                            for i in range(3):
                                rgb_image[:, :, i][mask] = color[i]
                        else:  # variation == 2: Контуры + заливка
                            from scipy.ndimage import binary_erosion
                            eroded_mask = binary_erosion(mask, iterations=1)
                            contour_mask = mask & ~eroded_mask
                            # Заливка с прозрачностью
                            for i in range(3):
                                rgb_image[:, :, i][mask] = (rgb_image[:, :, i][mask] * 0.7 + color[i] * 0.3).astype(np.uint8)
                            # Контуры поверх
                            for i in range(3):
                                rgb_image[:, :, i][contour_mask] = color[i]
                
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
                axis_names = ['X', 'Y', 'Z']
                slice_info = f"Slice {slice_idx+1}/{num_slices} ({axis_names[slice_axis]})"
                draw.text((10, 10), slice_info, fill=(255, 255, 255), font=font)
                
                # Добавляем информацию о вариации
                variation_names = ['Контуры', 'Заливка', 'Контуры+Заливка']
                variation_info = f"Вариация: {variation_names[variation]}"
                draw.text((10, 30), variation_info, fill=(255, 255, 255), font=font)
                
                if found_vertebrae:
                    vertebrae_info = f"Found: {', '.join(found_vertebrae)}"
                    draw.text((10, 50), vertebrae_info, fill=(255, 255, 255), font=font)
                
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
    Основная функция для тестирования разных вариаций BMP изображений.
    """
    try:
        # Создаем выходную папку
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        print("Начинаем тестирование разных вариаций BMP изображений")
        
        # Создаем тестовые данные
        print("Создаем тестовые NIfTI данные")
        nifti_img, nifti_seg = create_test_data()
        
        # Сохраняем тестовые данные
        save(nifti_img, str(output_dir / "test_mri.nii.gz"))
        save(nifti_seg, str(output_dir / "test_seg.nii.gz"))
        print("Тестовые данные сохранены")
        
        # Тестируем разные вариации и оси
        variations = [0, 1, 2]  # Контуры, Заливка, Контуры+Заливка
        axes = [0, 1, 2]  # X, Y, Z
        
        for axis in axes:
            for variation in variations:
                print(f"Создаем BMP для оси {axis}, вариация {variation}")
                create_sagittal_bmp_images(nifti_img, nifti_seg, output_dir, None, slice_axis=axis, variation=variation)
        
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