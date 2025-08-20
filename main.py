"""
Версия основного скрипта для анализа позвоночника с измерениями патологий.
Включает grading анализ и последующие измерения грыж и листезов.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

import pydicom.dataset

from spine_processor import get_device, run_segmentation, process_study
from grading_processor import SpineGradingProcessor
from pathology_measurements import measure_all_pathologies
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

import torch
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import cv2

from utils.constant import LANDMARK_LABELS, VERTEBRA_DESCRIPTIONS, COLORS


def initialize_models_fixed(
    ax_model_path: str = 'model/weights/ax.pth',
    sag_step_1_model_path: str = 'model/weights/sag_step_1.pth',
    sag_step_2_model_path: str = 'model/weights/sag_step_2.pth',
    grading_model_path: str = 'model/weights/grading.pth',
    use_dual_channel_grading: bool = True
):
    """
    Инициализирует все модели для обработки позвоночника.
    """
    device = get_device()
    
    # Загружаем модели сегментации
    ax_model = torch.load(ax_model_path, map_location=device, weights_only=False)
    sag_step_1_model = torch.load(sag_step_1_model_path, map_location=device, weights_only=False)
    sag_step_2_model = torch.load(sag_step_2_model_path, map_location=device, weights_only=False)
    
    # Создаем grading процессор
    try:
        grading_processor = SpineGradingProcessor(
            model_path=grading_model_path,
            device=str(device),
            use_dual_channel=use_dual_channel_grading
        )
        logging.info(f"Инициализирован grading процессор (двухканальный: {use_dual_channel_grading})")
    except Exception as e:
        logging.warning(f"Ошибка при инициализации grading процессора: {e}")
        logging.warning("Используется заглушка")
        grading_processor = None
    
    # Переводим модели сегментации в режим оценки
    ax_model.eval()
    sag_step_1_model.eval()
    sag_step_2_model.eval()
    
    return ax_model, sag_step_1_model, sag_step_2_model, grading_processor


def process_study_with_measurements(
    studies: List[pydicom.dataset.FileDataset],
    ax_model: torch.nn.Module,
    sag_step_1_model: torch.nn.Module,
    sag_step_2_model: torch.nn.Module,
    grading_processor: object,
    voxel_spacing: tuple = (1.0, 1.0, 1.0),
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Обрабатывает исследование позвоночника с измерениями патологий.
    """
    start_time = time.time()
    
    try:
        if logger:
            logger.info("Начинаем обработку исследования с измерениями патологий")
            logger.info(f"Количество DICOM файлов: {len(studies)}")
            logger.info(f"Размер вокселя: {voxel_spacing} мм")
        
        # Выполняем сегментацию
        nifti_img, nifti_seg = run_segmentation(studies, ax_model, sag_step_1_model, sag_step_2_model, logger)
        
        if nifti_img is None or nifti_seg is None:
            if logger:
                logger.error("Сегментация не удалась")
            return {"error": "Segmentation failed"}
        
        # Анализируем присутствующие диски
        seg_data = nifti_seg.get_fdata()
        unique_labels = np.unique(seg_data)
        present_disks = [label for label in LANDMARK_LABELS if label in unique_labels]
        
        if logger:
            logger.info(f"Найдены диски: {present_disks}")
        
        # Получаем данные изображения
        mri_data = [img.get_fdata() if hasattr(img, 'get_fdata') else None for img in nifti_img]
        mri_data = mri_data[:-1]
        example = next(arr for arr in mri_data if arr is not None)
        mri_data = [arr if arr is not None else np.zeros_like(example) for arr in mri_data]
        
        mask_data = nifti_seg.get_fdata()
        
        if grading_processor is not None:
            if logger:
                logger.info("Запускаем grading анализ дисков...")
            
            try:
                disk_results = grading_processor.process_disks(mri_data, mask_data, present_disks)
                grading_summary = grading_processor.create_summary(disk_results)
                
                if logger:
                    logger.info(f"Grading анализ завершен. Обработано дисков: {len(disk_results)}")
                    
            except Exception as e:
                if logger:
                    logger.error(f"Ошибка при grading анализе: {e}")
                disk_results = {disk_label: {"error": str(e)} for disk_label in present_disks}
                grading_summary = {"error": "Grading analysis failed"}
        else:
            if logger:
                logger.warning("Grading процессор не инициализирован, используется заглушка")
            disk_results = {disk_label: {"error": "Grading processor not available"} for disk_label in present_disks}
            grading_summary = {"error": "Grading processor not available"}
        
        # Измеряем патологии для дисков с обнаруженными патологиями
        pathology_measurements = {}
        if disk_results and 'error' not in grading_summary:
            if logger:
                logger.info("Запускаем измерения патологий...")
            
            try:
                pathology_measurements = measure_all_pathologies(
                    mri_data, mask_data, disk_results, voxel_spacing
                )
                
                if logger:
                    logger.info(f"Измерения патологий завершены. Измерено дисков: {len(pathology_measurements)}")
                    
                    # Выводим краткую статистику по измерениям
                    herniation_count = 0
                    spondylolisthesis_count = 0
                    
                    for disk_label, measurements in pathology_measurements.items():
                        if measurements.get('Disc herniation', {}).get('detected', False):
                            herniation_count += 1
                            volume = measurements['Disc herniation']['volume_mm3']
                            protrusion = measurements['Disc herniation']['max_protrusion_mm']
                            level_name = measurements.get('level_name', f'Disk_{disk_label}')
                            logger.info(f"Грыжа {level_name}: объем={volume:.1f}мм³, выпячивание={protrusion:.1f}мм")
                        
                        if measurements.get('Spondylolisthesis', {}).get('detected', False):
                            spondylolisthesis_count += 1
                            displacement = measurements['Spondylolisthesis']['displacement_mm']
                            percentage = measurements['Spondylolisthesis']['displacement_percentage']
                            grade = measurements['Spondylolisthesis']['grade']
                            level_name = measurements.get('level_name', f'Disk_{disk_label}')
                            logger.info(f"Листез {level_name}: смещение={displacement:.1f}мм ({percentage:.1f}%), степень={grade}")
                    
                    logger.info(f"Итого: грыж={herniation_count}, листезов={spondylolisthesis_count}")
                    
            except Exception as e:
                if logger:
                    logger.error(f"Ошибка при измерении патологий: {e}")
                pathology_measurements = {"error": str(e)}
        
        elapsed_time = time.time() - start_time
        
        if logger:
            logger.info(f"Обработка завершена за {elapsed_time:.2f} секунд")
        
        return {
            "processing_time": elapsed_time,
            "processed_disks": len(disk_results),
            "disk_results": disk_results,
            "grading_summary": grading_summary,
            "pathology_measurements": pathology_measurements,
            "segmentation_shape": seg_data.shape,
            "unique_labels": unique_labels.tolist(),
            "present_disks": present_disks,
            "voxel_spacing": voxel_spacing
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Критическая ошибка при обработке исследования: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}


def setup_logger(output_dir: Path, log_filename: str = "spine_analysis.log") -> logging.Logger:
    """
    Настраивает logger для записи в файл и консоль.
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
    """
    return read_dicoms_in_folder(study_folder)


def get_study_from_orthanc(study_instance_uid: str, client) -> List[pydicom.dataset.FileDataset]:
    """
    Получает исследование из Orthanc по Study Instance UID.
    """
    return download_study_from_orthanc(study_instance_uid, client)


def save_results_to_csv(results: Dict, output_dir: Path, logger: Optional[logging.Logger] = None) -> None:
    """
    Сохраняет результаты анализа в CSV файл.
    """
    try:
        if 'error' in results:
            if logger:
                logger.warning("Результаты содержат ошибки, CSV не будет создан")
            return
        
        csv_data = []
        
        # Обрабатываем результаты grading
        disk_results = results.get('disk_results', {})
        pathology_measurements = results.get('pathology_measurements', {})
        
        for disk_label, disk_result in disk_results.items():
            if 'error' in disk_result:
                continue
                
            row = {
                'disk_label': disk_label,
                'level_name': disk_result.get('level_name', f'Disk_{disk_label}'),
                'processing_time_ms': disk_result.get('processing_time', 0) * 1000,
            }
            
            # Добавляем результаты grading
            predictions = disk_result.get('predictions', {})
            row.update({
                'pfirrmann_grade': predictions.get('Pfirrmann grade', 0),
                'modic_changes': predictions.get('Modic', 0),
                'herniation_grade': predictions.get('Disc herniation', 0),
                'bulging_grade': predictions.get('Disc bulging', 0),
                'narrowing_grade': predictions.get('Disc narrowing', 0),
                'spondylolisthesis_grade': predictions.get('Spondylolisthesis', 0),
            })
            
            # Добавляем измерения патологий
            disk_measurements = pathology_measurements.get(str(disk_label), {})
            
            # Грыжа
            herniation = disk_measurements.get('Disc herniation', {})
            row.update({
                'herniation_detected': herniation.get('detected', False),
                'herniation_volume_mm3': herniation.get('volume_mm3', 0.0),
                'herniation_max_protrusion_mm': herniation.get('max_protrusion_mm', 0.0),
                'herniation_area_mm2': herniation.get('area_mm2', 0.0),
            })
            
            # Листез
            spondylolisthesis = disk_measurements.get('Spondylolisthesis', {})
            row.update({
                'spondylolisthesis_detected': spondylolisthesis.get('detected', False),
                'spondylolisthesis_displacement_mm': spondylolisthesis.get('displacement_mm', 0.0),
                'spondylolisthesis_displacement_percentage': spondylolisthesis.get('displacement_percentage', 0.0),
                'spondylolisthesis_grade': spondylolisthesis.get('grade', 'Normal'),
            })
            
            csv_data.append(row)
        
        # Создаем DataFrame и сохраняем в CSV
        if csv_data:
            df = pd.DataFrame(csv_data)
            csv_path = output_dir / "spine_analysis_results.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            if logger:
                logger.info(f"Результаты сохранены в CSV: {csv_path}")
                logger.info(f"Обработано дисков: {len(csv_data)}")
        else:
            if logger:
                logger.warning("Нет данных для сохранения в CSV")
                
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при сохранении CSV: {e}")


def create_sagittal_images_with_segmentation(nifti_img, nifti_seg, output_dir: Path, 
                                           logger: Optional[logging.Logger] = None) -> None:
    """
    Создает сагиттальные изображения с сегментацией в виде контуров и подписями позвонков.
    """
    try:
        # Создаем папку для изображений
        images_dir = output_dir / "sagittal_images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        if logger:
            logger.info(f"Создание сагиттальных изображений в папке: {images_dir}")
        
        # Получаем данные
        if hasattr(nifti_img, 'get_fdata'):
            img_data = nifti_img.get_fdata()
        else:
            img_data = np.asarray(nifti_img)
            
        if hasattr(nifti_seg, 'get_fdata'):
            seg_data = nifti_seg.get_fdata()
        else:
            seg_data = np.asarray(nifti_seg)
        
        # Нормализуем изображение
        img_normalized = (img_data - img_data.min()) / (img_data.max() - img_data.min())
        
        # Определяем цвета для разных структур
        colors = COLORS if 'COLORS' in globals() else {
            'vertebra': [255, 255, 0],    # Желтый
            'disk': [0, 255, 0],          # Зеленый
            'canal': [0, 0, 255],         # Синий
            'cord': [255, 0, 255],        # Пурпурный
            'sacrum': [255, 165, 0],      # Оранжевый
            'hernia': [255, 0, 0],        # Красный
            'bulging': [255, 192, 203],   # Розовый
        }
        
        # Описания позвонков
        vertebra_descriptions = VERTEBRA_DESCRIPTIONS if 'VERTEBRA_DESCRIPTIONS' in globals() else {
            11: "C1", 12: "C2", 13: "C3", 14: "C4", 15: "C5", 16: "C6", 17: "C7",
            21: "Th1", 22: "Th2", 23: "Th3", 24: "Th4", 25: "Th5", 26: "Th6", 
            27: "Th7", 28: "Th8", 29: "Th9", 30: "Th10", 31: "Th11", 32: "Th12",
            41: "L1", 42: "L2", 43: "L3", 44: "L4", 45: "L5", 50: "S1",
        }
        
        # Создаем изображения для каждого сагиттального среза
        num_slices = img_data.shape[0]  # Предполагаем, что сагиттальные срезы по оси 0
        
        for slice_idx in range(num_slices):
            try:
                # Получаем срез
                img_slice = img_normalized[slice_idx, :, :]
                seg_slice = seg_data[slice_idx, :, :]
                
                # Поворачиваем для правильной ориентации
                img_slice = np.rot90(img_slice, k=1)
                seg_slice = np.rot90(seg_slice, k=1)
                img_slice = np.fliplr(img_slice)
                seg_slice = np.fliplr(seg_slice)
                
                # Создаем RGB изображение
                img_rgb = np.stack([img_slice, img_slice, img_slice], axis=2)
                img_rgb = (img_rgb * 255).astype(np.uint8)
                
                # Убеждаемся, что массив имеет правильный порядок осей и тип для OpenCV
                img_rgb = np.ascontiguousarray(img_rgb, dtype=np.uint8)
                
                # Добавляем контуры сегментации
                unique_labels = np.unique(seg_slice)
                for label_value in unique_labels:
                    if label_value == 0:
                        continue
                    
                    # Определяем цвет для метки
                    if label_value in vertebra_descriptions:
                        color = colors.get('vertebra', [255, 255, 0])
                    elif label_value in [1, 2]:  # Канал и спинной мозг
                        color = colors.get('canal' if label_value == 2 else 'cord', [0, 0, 255])
                    elif label_value == 50:  # Крестец
                        color = colors.get('sacrum', [255, 165, 0])
                    elif label_value == 200:  # Грыжа
                        color = colors.get('hernia', [255, 0, 0])
                    elif label_value == 201:  # Выбухание
                        color = colors.get('bulging', [255, 192, 203])
                    else:  # Диски
                        color = colors.get('disk', [0, 255, 0])
                    
                    # Создаем маску для текущей метки
                    mask = (seg_slice == label_value).astype(np.uint8)
                    
                    # Убеждаемся, что маска имеет правильный тип для OpenCV
                    mask = np.ascontiguousarray(mask, dtype=np.uint8)
                    
                    # Находим контуры
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Рисуем контуры, убеждаемся что цвет в правильном формате (BGR для OpenCV)
                    if len(contours) > 0:
                        # Конвертируем RGB в BGR для OpenCV
                        bgr_color = (int(color[2]), int(color[1]), int(color[0]))
                        cv2.drawContours(img_rgb, contours, -1, bgr_color, thickness=2)
                
                # Конвертируем в PIL Image для добавления текста
                pil_image = Image.fromarray(img_rgb)
                draw = ImageDraw.Draw(pil_image)
                
                # Добавляем подписи ��озвонков
                try:
                    font = ImageFont.truetype("arial.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                for label_value in unique_labels:
                    if label_value in vertebra_descriptions:
                        # Находим центр позвонка
                        mask = (seg_slice == label_value)
                        if np.any(mask):
                            coords = np.argwhere(mask)
                            center_y, center_x = coords.mean(axis=0).astype(int)
                            
                            # Добавляем подпись
                            text = vertebra_descriptions[label_value]
                            
                            # Рисуем фон для текста
                            bbox = draw.textbbox((center_x, center_y), text, font=font)
                            draw.rectangle(bbox, fill=(0, 0, 0, 128))
                            
                            # Рисуем текст
                            draw.text((center_x, center_y), text, fill=(255, 255, 255), font=font)
                
                # Добавляем информацию о срезе
                slice_info = f"Sagittal slice {slice_idx + 1}/{num_slices}"
                draw.text((10, 10), slice_info, fill=(255, 255, 255), font=font)
                
                # Сохраняем изображение
                filename = f"sagittal_slice_{slice_idx + 1:03d}.png"
                filepath = images_dir / filename
                pil_image.save(filepath)
                
                if logger and slice_idx % 10 == 0:
                    logger.info(f"Создан срез {slice_idx + 1}/{num_slices}")
                    
            except Exception as e:
                if logger:
                    logger.error(f"Ошибка при создании среза {slice_idx}: {e}")
                continue
        
        if logger:
            logger.info(f"Создано {num_slices} сагиттальных изображений")
            
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при создании сагиттальных изображений: {e}")


def parse_args():
    """Парсинг аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Анализатор позвоночника с grading и измерениями патологий"
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
    
    parser.add_argument(
        '--grading_model', 
        type=str, 
        default='model/weights/grading.pth',
        help='Путь к модели grading'
    )
    
    parser.add_argument(
        '--disable_dual_channel', 
        action='store_true',
        help='Отключить двухканальный режим grading'
    )
    
    # Настройки измерений
    parser.add_argument(
        '--voxel_spacing', 
        type=float, 
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help='Размер вокселя в мм (depth height width)'
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
    Основная функция с grading анализом и измерениями патологий.
    """
    start_time = time.time()
    
    try:
        # Парсинг аргументов
        args = parse_args()
        
        # Создание выходной папки
        args.output.mkdir(exist_ok=True, parents=True)
        
        # Настройка логирования
        logger = setup_logger(args.output)
        logger.info("=== Запуск анализатора позвоночника с измерениями патологий ===")
        logger.info(f"Источник данных: {args.source}")
        logger.info(f"Использование Triton: {args.use_triton}")
        logger.info(f"Модель grading: {args.grading_model}")
        logger.info(f"Двухканальный режим: {not args.disable_dual_channel}")
        logger.info(f"Размер вокселя: {args.voxel_spacing} мм")
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
            
            # Проверяем наличие модели grading
            grading_model_path = args.grading_model
            use_dual_channel = not args.disable_dual_channel
            
            if os.path.exists(grading_model_path):
                logger.info(f"Найдена модель grading: {grading_model_path}")
                models = initialize_models_fixed(
                    grading_model_path=grading_model_path,
                    use_dual_channel_grading=use_dual_channel
                )
            else:
                logger.warning(f"Модель {grading_model_path} не найдена, используем заглушку")
                models = initialize_models_fixed(
                    grading_model_path='model/weights/grading.pth',
                    use_dual_channel_grading=False
                )
            
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
            ax_model, sag_step_1_model, sag_step_2_model, grading_processor = models
            results = process_study_with_measurements(
                studies, 
                ax_model, 
                sag_step_1_model, 
                sag_step_2_model, 
                grading_processor,
                tuple(args.voxel_spacing),
                logger
            )
        
        # 4. Сохранение результатов
        logger.info("Сохранение результатов...")
        
        if 'error' in results:
            logger.error(f"Ошибка обработки: {results['error']}")
        else:
            # Сохранение результатов в файл
            import json
            
            # Конвертируем numpy типы в стандартные Python типы
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {str(key): convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj
            
            results_converted = convert_numpy_types(results)
            
            results_file = args.output / "analysis_results_with_measurements.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_converted, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Результаты сохранены в: {results_file}")
            
            # Сохраняем результаты в CSV
            logger.info("Сохранение результатов в CSV...")
            save_results_to_csv(results, args.output, logger)
            
            # Создаем сагиттальные изображения с сегментацией
            logger.info("Создание сагиттальных изображений с сегментацией...")
            # Получаем nifti_img и nifti_seg из результатов сегментации
            # Нужно повторно выполнить сегментацию для получения изображений
            if not args.use_triton:
                ax_model, sag_step_1_model, sag_step_2_model, grading_processor = models
                nifti_img, nifti_seg = run_segmentation(studies, ax_model, sag_step_1_model, sag_step_2_model, logger)

                first_img = next(img for img in nifti_img if img is not None)
                if nifti_img is not None and nifti_seg is not None:
                    create_sagittal_images_with_segmentation(first_img, nifti_seg, args.output, logger)
                else:
                    logger.warning("Не удалось создать сагиттальные изображения - сегментация недоступна")
            
            # Краткая статистика
            if 'disk_results' in results:
                logger.info(f"Обработано дисков: {len(results['disk_results'])}")
                
                # Выводим результаты grading для каждого диска
                for disk_label, disk_result in results['disk_results'].items():
                    if 'error' not in disk_result and 'predictions' in disk_result:
                        level_name = disk_result.get('level_name', f'Disk_{disk_label}')
                        predictions = disk_result['predictions']
                        
                        bbox = disk_result.get('bounds')
                        center = disk_result.get('center')
                        if bbox is not None:
                            min_c, max_c = bbox
                            logger.info(f"Диск {level_name}: bbox_min={tuple(min_c)}, bbox_max={tuple(max_c)}, center={tuple(center) if center is not None else 'N/A'}")
                        else:
                            logger.info(f"Диск {level_name}: bbox_min=None, bbox_max=None, center={tuple(center) if center is not None else 'N/A'}")
                        logger.info(f"  - Pfirrmann: {predictions.get('Pfirrmann grade', 'N/A')}")
                        logger.info(f"  - Modic: {predictions.get('Modic', 'N/A')}")
                        logger.info(f"  - Herniation: {predictions.get('Disc herniation', 'N/A')}")
                        logger.info(f"  - Bulging: {predictions.get('Disc bulging', 'N/A')}")
                        logger.info(f"  - Narrowing: {predictions.get('Disc narrowing', 'N/A')}")
                        logger.info(f"  - Spondylolisthesis: {predictions.get('Spondylolisthesis', 'N/A')}")
                    elif 'error' in disk_result:
                        logger.warning(f"Диск {disk_label}: {disk_result['error']}")
            
            # Сводка по grading
            if 'grading_summary' in results and 'error' not in results['grading_summary']:
                summary = results['grading_summary']
                logger.info("=== Сводка по grading анализу ===")
                logger.info(f"Успешно обработано дисков: {summary.get('successful_discs', 0)}")
                logger.info(f"Ошибок обработки: {summary.get('failed_discs', 0)}")
                
                if 'pathology_counts' in summary:
                    for pathology, counts in summary['pathology_counts'].items():
                        positive_rate = (counts['positive'] / counts['total']) * 100 if counts['total'] > 0 else 0
                        logger.info(f"{pathology}: {counts['positive']}/{counts['total']} ({positive_rate:.1f}%) положительных")
            
            # # Сводка по измерениям патологий
            # if 'pathology_measurements' in results and results['pathology_measurements']:
            #     logger.info("=== Сводка по измерениям патологий ===")
            #     measurements = results['pathology_measurements']
            #
            #     total_herniation_volume = 0
            #     total_displacement = 0
            #     herniation_count = 0
            #     spondylolisthesis_count = 0
                
                # for disk_label, disk_measurements in measurements.items():
                #     if isinstance(disk_measurements, dict) and 'error' not in disk_measurements:
                #         if disk_measurements.get('herniation', {}).get('detected', False):
                #             herniation_count += 1
                #             volume = disk_measurements['herniation']['volume_mm3']
                #             total_herniation_volume += volume
                #
                #         if disk_measurements.get('spondylolisthesis', {}).get('detected', False):
                #             spondylolisthesis_count += 1
                #             displacement = disk_measurements['spondylolisthesis']['displacement_mm']
                #             total_displacement += displacement
                #
                # logger.info(f"Всего грыж: {herniation_count}")
                # if herniation_count > 0:
                #     logger.info(f"Общий объем грыж: {total_herniation_volume:.1f} мм³")
                #     logger.info(f"Средний объем грыжи: {total_herniation_volume/herniation_count:.1f} мм³")
                #
                # logger.info(f"Всего листезов: {spondylolisthesis_count}")
                # if spondylolisthesis_count > 0:
                #     logger.info(f"Общее смещение: {total_displacement:.1f} мм")
                #     logger.info(f"Среднее смещение: {total_displacement/spondylolisthesis_count:.1f} мм")
        
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