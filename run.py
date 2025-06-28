import argparse
import textwrap
from pathlib import Path
import csv
import pandas as pd
import logging
import traceback

import nibabel
import numpy as np
import torch
from nibabel.nifti1 import Nifti1Image
from nibabel.loadsave import save
from scipy.ndimage import affine_transform
from scipy.ndimage import binary_dilation

from utils import pad_nd_image, average4d, reorient_canonical, resample, DefaultPreprocessor, largest_component, iterative_label, transform_seg2image, extract_alternate, fill_canal, crop_image2seg, recalculate_correspondence
from model import internal_predict_sliding_window_return_logits, internal_get_sliding_window_slicers, GradingModel, BasicBlock, Bottleneck
from dicom_io import load_dicoms_from_folder, load_study_dicoms

# --- Настройка логирования ---
def setup_logger(output_dir: Path, log_filename: str = "spine_analysis.log"):
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

# --- Константы ---
DEFAULT_CROP_MARGIN = 10
DEFAULT_CROP_SHAPE_PAD = (16, 16, 0)  # (D, H, W) добавка к crop_shape
LANDMARK_LABELS = [63, 64, 65, 66, 67, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 91, 92, 93, 94, 95, 96, 100]

# Пороговые значения и параметры анализа
HERN_THRESHOLD_STD = 1.5
BULGE_THRESHOLD_STD = 0.8
HERN_THRESHOLD_MAX = 0.9
BULGE_THRESHOLD_MAX = 0.7
MIN_HERNIA_SIZE = 10  # минимальный размер компонента для грыжи (вокселей)
MIN_BULGING_SIZE = 5  # минимальный размер компонента для выбухания (вокселей)
MAX_BULGING_SIZE = 100  # максимальный размер компонента для выбухания (вокселей)
MIN_BULGING_SHAPE_SIZE = 10  # минимальный размер компонента для выбухания по форме
MAX_BULGING_SHAPE_SIZE = 100  # максимальный размер компонента для выбухания по форме
BULGING_SHAPE_THRESHOLD_STD = 1.5  # порог для выбухания по форме
MIN_BULGING_COORDS = 5  # минимальное количество координат для выбухания
DILATE_SIZE = 5  # размер дилатации для largest_component
SPONDY_SIGNIFICANT_MM = 1.0  # минимальный листез, считающийся значимым
VERTEBRA_SEARCH_RANGE = 20  # диапазон поиска позвонков (1-20)
VERTEBRA_NEAR_DISK_DISTANCE = 50  # макс. расстояние до диска для поиска позвонков

# Метки для сегментации
CANAL_LABEL = 2
CORD_LABEL = 1
SACRUM_LABEL = 50
HERNIA_LABEL = 200
BULGING_LABEL = 201

# Размеры кропа и патчей
SAG_PATCH_SIZE = (128, 96, 96)
AX_PATCH_SIZE = (224, 224)

# Диапазоны для extract_alternate
EXTRACT_LABELS_RANGE = list(range(63, 101))

# Индексы классов в предсказаниях модели
IDX_MODIC = 0
IDX_UP_ENDPLATE = 1
IDX_LOW_ENDPLATE = 2
IDX_SPONDY = 3
IDX_HERN = 4
IDX_NARROW = 5
IDX_BULGE = 6
IDX_PFIRRMAN = 7

# Коэффициент вариации для симметрии выбухания
BULGING_SYMMETRY_CV = 0.3

# BBox для crop_shape
MEDIAN_BBOX = np.array([[136, 168], [104, 150], [16, 70]]).astype(int)


def find_bounding_box(mask: np.ndarray, target_class: int):
    """
    Находит bounding box для заданного класса в маске.
    Возвращает tuple из срезов по каждой оси.
    """
    indices = np.argwhere(mask == target_class)
    if indices.size == 0:
        return None
    min_coords = indices.min(axis=0)
    max_coords = indices.max(axis=0) + 1
    return tuple(slice(start, end) for start, end in zip(min_coords, max_coords))


def compute_principal_angle(mask_disk: np.ndarray) -> float:
    """
    Вычисляет угол главной оси объекта (например, диска) в градусах.
    """
    coords = np.argwhere(mask_disk)
    if len(coords) < 2:
        return 0.0
    coords_xy = coords[:, :2]  # Ignore z-axis
    coords_centered = coords_xy - coords_xy.mean(axis=0)
    cov = np.cov(coords_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, np.argmax(eigvals)]
    angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def is_continuous_with_disk(component_mask: np.ndarray, disk_mask: np.ndarray) -> bool:
    """
    Проверяет, соединён ли компонент с маской диска (есть ли смежные воксели).
    """
    from scipy.ndimage import binary_dilation
    dilated_component = binary_dilation(component_mask, iterations=1)
    return np.any(dilated_component & disk_mask)

def deforms_canal(component_mask: np.ndarray, canal_mask: np.ndarray, rotated_mask: np.ndarray) -> bool:
    """
    Проверяет, пересекается ли компонент с каналом (деформирует ли канал).
    """
    return np.any(component_mask & canal_mask)

def rotate_volume_and_mask(volume: np.ndarray, mask: np.ndarray, angle_deg: float):
    """
    Поворачивает volume и mask на заданный угол (в градусах) вокруг z-оси.
    """
    angle_rad = -np.radians(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                0,                1]
    ])
    center = 0.5 * np.array(volume.shape)
    offset = center - rotation_matrix @ center
    rotated_volume = affine_transform(volume, rotation_matrix, offset=offset.tolist(), order=1)
    rotated_mask = affine_transform(mask, rotation_matrix, offset=offset.tolist(), order=0)
    return rotated_volume, rotated_mask


def pad_or_crop_to_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    Паддит или обрезает массив до нужной формы target_shape.
    """
    pad = []
    slices = []
    for i, (sz, tsz) in enumerate(zip(arr.shape, target_shape)):
        if sz < tsz:
            before = (tsz - sz) // 2
            after = tsz - sz - before
            pad.append((before, after))
            slices.append(slice(0, sz))
        elif sz > tsz:
            pad.append((0, 0))
            start = (sz - tsz) // 2
            slices.append(slice(start, start + tsz))
        else:
            pad.append((0, 0))
            slices.append(slice(0, sz))
    arr = arr[tuple(slices)]
    if any(p != (0, 0) for p in pad):
        arr = np.pad(arr, pad, mode='constant')
    return arr


def get_crop_shape(median_bbox: np.ndarray, pad=DEFAULT_CROP_SHAPE_PAD) -> tuple:
    """
    Возвращает форму кропа с учётом паддинга.
    """
    crop_shape = tuple((stop - start) + p for (start, stop), p in zip(median_bbox, pad))
    return crop_shape


def get_centered_slices(bbox_center: list, crop_shape: tuple, data_shape: tuple) -> tuple:
    """
    Возвращает tuple из срезов, центрированных по bbox_center, с формой crop_shape.
    """
    slices = []
    for dim in range(3):
        start = bbox_center[dim] - (crop_shape[dim] // 2)
        stop = start + crop_shape[dim]
        start = max(0, start)
        stop = min(data_shape[dim], stop)
        slices.append(slice(start, stop))
    return tuple(slices)


def crop_and_pad(data: np.ndarray, slices: tuple, crop_shape: tuple) -> np.ndarray:
    """
    Кроп и паддинг массива data по срезам slices до формы crop_shape.
    """
    cropped = data[slices]
    return pad_or_crop_to_shape(cropped, crop_shape)


def measure_spondylolisthesis_alternative(rotated_mri: np.ndarray, rotated_mask: np.ndarray, disk_label: int, spacing_mm: tuple = (1.0, 1.0, 1.0), logger=None) -> float:
    """
    Альтернативный метод измерения листеза на основе анализа формы диска.
    
    :param rotated_mri: Выровненное МРТ изображение
    :param rotated_mask: Выровненная сегментация
    :param disk_label: Метка диска для анализа
    :param spacing_mm: Разрешение вокселей в мм (x, y, z)
    :param logger: Logger для записи сообщений
    :return: Листез в миллиметрах (0 если не обнаружен)
    """
    try:
        # Находим диск
        disk_mask = (rotated_mask == disk_label)
        if not np.any(disk_mask):
            if logger:
                logger.warning(f"Диск {disk_label}: диск не найден в альтернативном методе")
            return 0.0
        
        # Находим координаты диска
        disk_coords = np.argwhere(disk_mask)
        
        # Находим центр диска
        disk_center = disk_coords.mean(axis=0)
        
        # Вычисляем границы диска по каждой оси
        min_coords = disk_coords.min(axis=0)
        max_coords = disk_coords.max(axis=0)
        
        # Находим центр масс диска
        center_of_mass = disk_coords.mean(axis=0)
        
        # Вычисляем смещение центра масс от геометрического центра
        geometric_center = (min_coords + max_coords) / 2
        displacement = center_of_mass - geometric_center
        
        # Листез - это смещение по X-оси (горизонтальное смещение)
        spondylolisthesis_mm = abs(displacement[0]) * spacing_mm[0]
        
        if logger:
            logger.info(f"Диск {disk_label}: альтернативный метод - смещение {spondylolisthesis_mm:.1f} мм")
        
        # Порог для определения значимого листеза
        if spondylolisthesis_mm < 1.0:
            if logger:
                logger.info(f"Диск {disk_label}: альтернативный метод - листез менее 1 мм")
            return 0.0
            
        return spondylolisthesis_mm
        
    except Exception as e:
        if logger:
            logger.error(f"Ошибка в альтернативном методе измерения листеза для диска {disk_label}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return 0.0


def measure_spondylolisthesis(rotated_mri: np.ndarray, rotated_mask: np.ndarray, disk_label: int, spacing_mm: tuple = (1.0, 1.0, 1.0), logger=None) -> float:
    """
    Измеряет листез в миллиметрах на основе смещения позвонков.
    
    :param rotated_mri: Выровненное МРТ изображение
    :param rotated_mask: Выровненная сегментация
    :param disk_label: Метка диска для анализа
    :param spacing_mm: Разрешение вокселей в мм (x, y, z)
    :param logger: Logger для записи сообщений
    :return: Листез в миллиметрах (0 если не обнаружен)
    """
    try:
        # Находим диск
        disk_mask = (rotated_mask == disk_label)
        if not np.any(disk_mask):
            if logger:
                logger.warning(f"Диск {disk_label}: диск не найден")
            return 0.0
        
        # Находим центр диска
        disk_coords = np.argwhere(disk_mask)
        disk_center = disk_coords.mean(axis=0)
        
        # Ищем позвонки в окрестности диска (расширенный диапазон)
        vertebra_centers = []
        
        # Расширяем поиск позвонков - ищем все возможные метки позвонков
        for vertebra_label in range(1, VERTEBRA_SEARCH_RANGE):  # Расширенный диапазон
            vertebra_mask = (rotated_mask == vertebra_label)
            if np.any(vertebra_mask):
                # Находим центр масс позвонка
                coords = np.argwhere(vertebra_mask)
                center = coords.mean(axis=0)
                
                # Проверяем, что позвонок находится рядом с диском (в пределах 50 вокселей)
                distance_to_disk = np.linalg.norm(center - disk_center)
                if distance_to_disk < VERTEBRA_NEAR_DISK_DISTANCE:  # Фильтруем только близкие позвонки
                    vertebra_centers.append((vertebra_label, center, distance_to_disk))
        
        if len(vertebra_centers) < 2:
            if logger:
                logger.info(f"Диск {disk_label}: найдено менее 2 позвонков рядом с диском, используем альтернативный метод")
            return measure_spondylolisthesis_alternative(rotated_mri, rotated_mask, disk_label, spacing_mm, logger)
        
        # Сортируем позвонки по расстоянию к диску
        vertebra_centers.sort(key=lambda x: x[2])
        
        # Берем два ближайших позвонка
        vertebra1 = vertebra_centers[0]
        vertebra2 = vertebra_centers[1]
        
        if logger:
            logger.info(f"Диск {disk_label}: позвонок 1 ({vertebra1[0]}) в позиции {vertebra1[1]}, расстояние {vertebra1[2]:.1f}")
            logger.info(f"Диск {disk_label}: позвонок 2 ({vertebra2[0]}) в позиции {vertebra2[1]}, расстояние {vertebra2[2]:.1f}")
        
        # Вычисляем смещение по X-оси (горизонтальное смещение)
        displacement_x_pixels = abs(vertebra1[1][0] - vertebra2[1][0])
        
        # Конвертируем в миллиметры с учетом реального разрешения
        spondylolisthesis_mm = displacement_x_pixels * spacing_mm[0]
        
        if logger:
            logger.info(f"Диск {disk_label}: смещение в пикселях = {displacement_x_pixels}, в мм = {spondylolisthesis_mm:.1f}")
        
        # Снижаем порог для определения значимого листеза (например, > 1 мм)
        if spondylolisthesis_mm < SPONDY_SIGNIFICANT_MM:
            if logger:
                logger.info(f"Диск {disk_label}: листез менее {SPONDY_SIGNIFICANT_MM} мм, считаем отсутствующим")
            return 0.0
            
        if logger:
            logger.info(f"Диск {disk_label}: обнаружен листез {spondylolisthesis_mm:.1f} мм")
        return spondylolisthesis_mm
        
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при измерении листеза для диска {disk_label}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return measure_spondylolisthesis_alternative(rotated_mri, rotated_mask, disk_label, spacing_mm, logger)


def detect_herniation_bulging(rotated_mri: np.ndarray, rotated_mask: np.ndarray, disk_label: int, spacing_mm: tuple = (1.0, 1.0, 1.0), logger=None) -> tuple:
    """
    Определяет локализацию грыжи/выбухания на диске и возвращает маску пораженной области.
    
    :param rotated_mri: Выровненное МРТ изображение
    :param rotated_mask: Выровненная сегментация
    :param disk_label: Метка диска
    :param spacing_mm: Разрешение вокселей в мм (x, y, z)
    :param logger: Logger для записи сообщений
    :return: (herniation_mask, bulging_mask, herniation_volume, bulging_volume, herniation_size, bulging_size)
    """
    try:
        # Создаем маски для грыжи и выбухания
        herniation_mask = np.zeros_like(rotated_mask, dtype=bool)
        bulging_mask = np.zeros_like(rotated_mask, dtype=bool)
        
        # Находим диск
        disk_mask = (rotated_mask == disk_label)
        if not np.any(disk_mask):
            if logger:
                logger.warning(f"Диск {disk_label}: диск не найден")
            return herniation_mask, bulging_mask, 0.0, 0.0, 0.0, 0.0
        
        # Находим канал и спинной мозг
        canal_mask = (rotated_mask == 2)
        cord_mask = (rotated_mask == 1)
        
        # Определяем границы диска
        disk_coords = np.argwhere(disk_mask)
        if len(disk_coords) == 0:
            if logger:
                logger.warning(f"Диск {disk_label}: координаты диска не найдены")
            return herniation_mask, bulging_mask, 0.0, 0.0, 0.0, 0.0
        
        # Находим центр диска
        disk_center = disk_coords.mean(axis=0)
        
        # Определяем заднюю часть диска (ближе к каналу)
        # Предполагаем, что канал находится сзади диска
        canal_coords = np.argwhere(canal_mask)
        if len(canal_coords) > 0:
            canal_center = canal_coords.mean(axis=0)
            # Вектор от центра диска к центру канала
            direction_to_canal = canal_center - disk_center
            direction_to_canal = direction_to_canal / np.linalg.norm(direction_to_canal)
        else:
            # Если канал не найден, используем стандартное направление
            direction_to_canal = np.array([0, 1, 0])  # Назад
        
        # Нормализуем интенсивность МРТ для лучшего анализа
        mri_normalized = (rotated_mri - rotated_mri.min()) / (rotated_mri.max() - rotated_mri.min())
        
        # Анализируем интенсивность в задней части диска
        posterior_voxels = []
        for coord in disk_coords:
            # Проверяем, находится ли точка в задней части диска
            relative_pos = coord - disk_center
            projection = np.dot(relative_pos, direction_to_canal)
            
            if projection > 0:  # Задняя часть диска
                intensity = mri_normalized[coord[0], coord[1], coord[2]]
                posterior_voxels.append((coord, intensity))
        
        if not posterior_voxels:
            if logger:
                logger.warning(f"Диск {disk_label}: задние воксели не найдены")
            return herniation_mask, bulging_mask, 0.0, 0.0, 0.0, 0.0
        
        # Вычисляем статистики интенсивности для определения порогов
        intensities = [v[1] for v in posterior_voxels]
        mean_intensity = np.mean(intensities)
        std_intensity = np.std(intensities)
        
        if logger:
            logger.info(f"Диск {disk_label}: средняя интенсивность = {mean_intensity:.3f}, std = {std_intensity:.3f}")
        
        # Адаптивные пороги на основе статистики
        herniation_threshold = mean_intensity + HERN_THRESHOLD_STD * std_intensity
        bulging_threshold = mean_intensity + BULGE_THRESHOLD_STD * std_intensity
        # Ограничиваем пороги разумными значениями
        herniation_threshold = min(herniation_threshold, HERN_THRESHOLD_MAX)
        bulging_threshold = min(bulging_threshold, BULGE_THRESHOLD_MAX)
        
        if logger:
            logger.info(f"Диск {disk_label}: пороги - грыжа: {herniation_threshold:.3f}, выбухание: {bulging_threshold:.3f}")
        
        # Классифицируем воксели
        herniation_candidates = []
        bulging_candidates = []
        
        for coord, intensity in posterior_voxels:
            if intensity > herniation_threshold:
                herniation_candidates.append(coord)
            elif intensity > bulging_threshold:
                bulging_candidates.append(coord)
        
        # Проверяем непрерывность и деформацию канала для грыж
        if herniation_candidates:
            herniation_mask = detect_continuous_herniation(
                herniation_candidates, disk_mask, canal_mask, rotated_mask, logger
            )
        
        # Улучшенное обнаружение выбуханий
        if bulging_candidates or np.any(disk_mask):
            bulging_mask = detect_bulging_improved(
                bulging_candidates, disk_mask, herniation_mask, rotated_mri, rotated_mask, 
                disk_center, direction_to_canal, logger
            )
        
        if logger:
            logger.info(f"Диск {disk_label}: найдено грыж: {np.sum(herniation_mask)}, выбуханий: {np.sum(bulging_mask)}")
        
        # Вычисляем объемы в мм³ с учетом реального разрешения
        voxel_volume_mm3 = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
        herniation_volume = np.sum(herniation_mask) * voxel_volume_mm3
        bulging_volume = np.sum(bulging_mask) * voxel_volume_mm3
        
        # Вычисляем размеры (максимальный диаметр) в мм
        if np.any(herniation_mask):
            herniation_coords = np.argwhere(herniation_mask)
            herniation_center = herniation_coords.mean(axis=0)
            herniation_distances = np.linalg.norm(herniation_coords - herniation_center, axis=1)
            herniation_size = 2 * np.max(herniation_distances) * spacing_mm[0]  # Диаметр в мм
        else:
            herniation_size = 0.0
        
        if np.any(bulging_mask):
            bulging_coords = np.argwhere(bulging_mask)
            bulging_center = bulging_coords.mean(axis=0)
            bulging_distances = np.linalg.norm(bulging_coords - bulging_center, axis=1)
            bulging_size = 2 * np.max(bulging_distances) * spacing_mm[0]  # Диаметр в мм
        else:
            bulging_size = 0.0
        
        if logger:
            logger.info(f"Диск {disk_label}: объемы - грыжа: {herniation_volume:.1f} мм³, выбухание: {bulging_volume:.1f} мм³")
            logger.info(f"Диск {disk_label}: размеры - грыжа: {herniation_size:.1f} мм, выбухание: {bulging_size:.1f} мм")
        
        return herniation_mask, bulging_mask, herniation_volume, bulging_volume, herniation_size, bulging_size
        
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при определении грыжи/выбухания для диска {disk_label}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return (np.zeros_like(rotated_mask, dtype=bool), 
                np.zeros_like(rotated_mask, dtype=bool), 
                0.0, 0.0, 0.0, 0.0)


def detect_continuous_herniation(candidate_coords: list, disk_mask: np.ndarray, canal_mask: np.ndarray, 
                               rotated_mask: np.ndarray, logger=None) -> np.ndarray:
    """
    Обнаруживает непрерывную грыжу, которая деформирует канал.
    
    :param candidate_coords: Координаты кандидатов в грыжу
    :param disk_mask: Маска диска
    :param canal_mask: Маска канала
    :param rotated_mask: Полная сегментация
    :param logger: Logger для записи сообщений
    :return: Маска непрерывной грыжи
    """
    try:
        from scipy.ndimage import label, binary_closing, binary_opening
        
        # Создаем маску кандидатов
        herniation_candidates = np.zeros_like(disk_mask, dtype=bool)
        for coord in candidate_coords:
            herniation_candidates[coord[0], coord[1], coord[2]] = True
        
        # Находим связанные компоненты
        labeled_candidates, num_features = label(herniation_candidates)
        
        if logger:
            logger.info(f"Найдено {num_features} компонентов-кандидатов в грыжу")
        
        # Проверяем каждый компонент на деформацию канала
        valid_herniations = np.zeros_like(herniation_candidates, dtype=bool)
        
        for component_id in range(1, num_features + 1):
            component_mask = (labeled_candidates == component_id)
            
            # Проверяем размер компонента (должен быть достаточно большим)
            component_size = np.sum(component_mask)
            if component_size < MIN_HERNIA_SIZE:  # Минимальный размер для грыжи
                if logger:
                    logger.info(f"Компонент {component_id} слишком мал ({component_size} вокселей)")
                continue
            
            # Проверяем, деформирует ли компонент канал
            if deforms_canal(component_mask, canal_mask, rotated_mask):
                valid_herniations |= component_mask
                if logger:
                    logger.info(f"Компонент {component_id} деформирует канал - принят как грыжа")
            else:
                if logger:
                    logger.info(f"Компонент {component_id} не деформирует канал - отклонен")
        
        # Морфологические операции для сглаживания
        if np.any(valid_herniations):
            # Закрытие для заполнения мелких отверстий
            valid_herniations = binary_closing(valid_herniations, iterations=1)
            # Открытие для удаления мелких шумов
            valid_herniations = binary_opening(valid_herniations, iterations=1)
        
        return valid_herniations
        
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при обнаружении непрерывной грыжи: {e}")
        return np.zeros_like(disk_mask, dtype=bool)


def detect_bulging_improved(bulging_candidates: list, disk_mask: np.ndarray, herniation_mask: np.ndarray,
                          rotated_mri: np.ndarray, rotated_mask: np.ndarray, disk_center: np.ndarray,
                          direction_to_canal: np.ndarray, logger=None) -> np.ndarray:
    """
    Улучшенное обнаружение выбуханий диска на основе анализа формы и симметрии.
    
    :param bulging_candidates: Координаты кандидатов в выбухание
    :param disk_mask: Маска диска
    :param herniation_mask: Маска грыжи
    :param rotated_mri: МРТ изображение
    :param rotated_mask: Сегментация
    :param disk_center: Центр диска
    :param direction_to_canal: Направление к каналу
    :param logger: Logger для записи сообщений
    :return: Маска выбухания
    """
    try:
        from scipy.ndimage import label, binary_closing, binary_opening, binary_dilation
        
        # Создаем маску кандидатов
        bulging_candidates_mask = np.zeros_like(disk_mask, dtype=bool)
        for coord in bulging_candidates:
            bulging_candidates_mask[coord[0], coord[1], coord[2]] = True
        
        # Исключаем области, уже занятые грыжей
        bulging_candidates_mask = bulging_candidates_mask & ~herniation_mask
        
        # Анализируем форму диска для обнаружения выбуханий
        bulging_from_shape = analyze_disk_shape_for_bulging(
            disk_mask, rotated_mri, disk_center, direction_to_canal, logger
        )
        
        # Объединяем результаты
        combined_bulging = bulging_candidates_mask | bulging_from_shape
        
        # Находим связанные компоненты
        labeled_candidates, num_features = label(combined_bulging)
        
        if logger:
            logger.info(f"Найдено {num_features} компонентов-кандидатов в выбухание")
        
        # Проверяем каждый компонент
        valid_bulgings = np.zeros_like(combined_bulging, dtype=bool)
        
        for component_id in range(1, num_features + 1):
            component_mask = (labeled_candidates == component_id)
            
            # Проверяем размер компонента
            component_size = np.sum(component_mask)
            if component_size < MIN_BULGING_SIZE:  # Минимальный размер для выбухания
                if logger:
                    logger.info(f"Компонент выбухания {component_id} слишком мал ({component_size} вокселей)")
                continue
            
            # Проверяем непрерывность с диском
            if is_continuous_with_disk(component_mask, disk_mask):
                # Дополнительная проверка: выбухание должно быть симметричным
                if is_symmetric_bulging(component_mask, disk_center, direction_to_canal):
                    valid_bulgings |= component_mask
                    if logger:
                        logger.info(f"Компонент выбухания {component_id} непрерывен и симметричен - принят")
                else:
                    if logger:
                        logger.info(f"Компонент выбухания {component_id} не симметричен - отклонен")
            else:
                if logger:
                    logger.info(f"Компонент выбухания {component_id} не непрерывен с диском - отклонен")
        
        # Морфологические операции для сглаживания
        if np.any(valid_bulgings):
            valid_bulgings = binary_closing(valid_bulgings, iterations=1)
            valid_bulgings = binary_opening(valid_bulgings, iterations=1)
        
        return valid_bulgings
        
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при улучшенном обнаружении выбухания: {e}")
        return np.zeros_like(disk_mask, dtype=bool)


def analyze_disk_shape_for_bulging(disk_mask: np.ndarray, rotated_mri: np.ndarray, 
                                 disk_center: np.ndarray, direction_to_canal: np.ndarray, 
                                 logger=None) -> np.ndarray:
    """
    Анализирует форму диска для обнаружения выбуханий.
    
    :param disk_mask: Маска диска
    :param rotated_mri: МРТ изображение
    :param disk_center: Центр диска
    :param direction_to_canal: Направление к каналу
    :param logger: Logger для записи сообщений
    :return: Маска выбухания на основе формы
    """
    try:
        from scipy.ndimage import binary_dilation, distance_transform_edt
        
        # Находим границы диска
        disk_boundary = binary_dilation(disk_mask, iterations=1) & ~disk_mask
        
        # Вычисляем расстояние от центра диска до каждой точки границы
        disk_coords = np.argwhere(disk_mask)
        boundary_coords = np.argwhere(disk_boundary)
        
        if len(disk_coords) == 0 or len(boundary_coords) == 0:
            return np.zeros_like(disk_mask, dtype=bool)
        
        # Вычисляем среднее расстояние от центра до границы
        distances_to_center = []
        for coord in boundary_coords:
            distance = np.linalg.norm(coord - disk_center)
            distances_to_center.append(distance)
        
        mean_distance = np.mean(distances_to_center)
        std_distance = np.std(distances_to_center)
        
        if logger:
            logger.info(f"Среднее расстояние до границы диска: {mean_distance:.2f} ± {std_distance:.2f}")
        
        # Находим области, где расстояние больше среднего + 1.5*std (выбухание)
        bulging_threshold = mean_distance + BULGING_SHAPE_THRESHOLD_STD * std_distance
        
        # Создаем маску выбухания на основе расстояния
        bulging_mask = np.zeros_like(disk_mask, dtype=bool)
        
        for coord in disk_coords:
            distance = np.linalg.norm(coord - disk_center)
            if distance > bulging_threshold:
                # Проверяем, что это задняя часть диска (выбухание обычно сзади)
                relative_pos = coord - disk_center
                projection = np.dot(relative_pos, direction_to_canal)
                if projection > 0:  # Задняя часть
                    bulging_mask[coord[0], coord[1], coord[2]] = True
        
        # Дополнительная проверка: выбухание должно быть локальным
        if np.any(bulging_mask):
            # Находим связанные компоненты выбухания
            from scipy.ndimage import label
            labeled_bulging, num_components = label(bulging_mask)
            
            # Оставляем только самые большие компоненты
            valid_bulging = np.zeros_like(bulging_mask, dtype=bool)
            for component_id in range(1, num_components + 1):
                component_mask = (labeled_bulging == component_id)
                component_size = np.sum(component_mask)
                
                # Компонент должен быть достаточно большим, но не слишком большим
                if MIN_BULGING_SHAPE_SIZE <= component_size <= MAX_BULGING_SHAPE_SIZE:  # Разумный диапазон для выбухания
                    valid_bulging |= component_mask
                    if logger:
                        logger.info(f"Найдено выбухание формы: компонент {component_id}, размер {component_size}")
        
        return valid_bulging
        
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при анализе формы диска: {e}")
        return np.zeros_like(disk_mask, dtype=bool)


def is_symmetric_bulging(bulging_mask: np.ndarray, disk_center: np.ndarray, 
                        direction_to_canal: np.ndarray) -> bool:
    """
    Проверяет, является ли выбухание симметричным относительно центра диска.
    
    :param bulging_mask: Маска выбухания
    :param disk_center: Центр диска
    :param direction_to_canal: Направление к каналу
    :return: True если выбухание симметрично
    """
    try:
        bulging_coords = np.argwhere(bulging_mask)
        
        if len(bulging_coords) < MIN_BULGING_COORDS:
            return False
        
        # Вычисляем центр масс выбухания
        bulging_center = bulging_coords.mean(axis=0)
        
        # Вычисляем смещение от центра диска
        displacement = bulging_center - disk_center
        
        # Проекция смещения на направление к каналу
        projection = np.dot(displacement, direction_to_canal)
        
        # Выбухание должно быть направлено к каналу (положительная проекция)
        if projection < 0:
            return False
        
        # Проверяем симметрию относительно центра диска
        # Вычисляем расстояния от центра диска до точек выбухания
        distances = []
        for coord in bulging_coords:
            distance = np.linalg.norm(coord - disk_center)
            distances.append(distance)
        
        # Проверяем, что расстояния не слишком сильно различаются (симметрия)
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Коэффициент вариации должен быть небольшим для симметричного выбухания
        cv = std_distance / mean_distance if mean_distance > 0 else 0
        
        return cv < BULGING_SYMMETRY_CV  # Коэффициент вариации менее 30%
        
    except Exception as e:
        return False


def save_segmentation_with_herniations(rotated_mask: np.ndarray, disk_label: int, 
                                     herniation_mask: np.ndarray, bulging_mask: np.ndarray,
                                     output_dir: Path, logger=None,
                                     herniation_label: int = HERNIA_LABEL, bulging_label: int = BULGING_LABEL) -> np.ndarray:
    """
    Помечает грыжи и выбухания на сегментации и сохраняет обновленную сегментацию.
    
    :param rotated_mask: Выровненная сегментация
    :param disk_label: Метка диска
    :param herniation_mask: Маска грыжи
    :param bulging_mask: Маска выбухания
    :param output_dir: Папка для сохранения
    :param logger: Logger для записи сообщений
    :param herniation_label: Новая метка для грыжи
    :param bulging_label: Новая метка для выбухания
    :return: Обновленная сегментация
    """
    try:
        updated_mask = rotated_mask.copy()
        
        # Помечаем грыжи
        if np.any(herniation_mask):
            # Перекрашиваем воксели диска в грыжу
            disk_and_herniation = (rotated_mask == disk_label) & herniation_mask
            updated_mask[disk_and_herniation] = herniation_label
            if logger:
                logger.info(f"Диск {disk_label}: помечено {np.sum(disk_and_herniation)} вокселей как грыжа (метка {herniation_label})")
        
        # Помечаем выбухания
        if np.any(bulging_mask):
            # Перекрашиваем воксели диска в выбухание (но не те, что уже грыжа)
            disk_and_bulging = (rotated_mask == disk_label) & bulging_mask & ~herniation_mask
            updated_mask[disk_and_bulging] = bulging_label
            if logger:
                logger.info(f"Диск {disk_label}: помечено {np.sum(disk_and_bulging)} вокселей как выбухание (метка {bulging_label})")
        
        return updated_mask
        
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при пометке грыж/выбуханий для диска {disk_label}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return rotated_mask


def process_disk(mri_data: np.ndarray, mask_data: np.ndarray, disk_label: int, crop_shape: tuple, nifti_img: Nifti1Image, nifti_seg: Nifti1Image, output_dir: Path, model, logger=None):
    """
    Кроп, выравнивание и сохранение ROI для одного диска.
    Возвращает (predictions, herniation_mask, bulging_mask) или None при ошибке.
    """
    try:
        disk_mask = (mask_data == disk_label)
        if not np.any(disk_mask):
            if logger:
                logger.warning(f"Диск {disk_label}: диск не найден в маске")
            return None
        bbox = find_bounding_box(mask_data, disk_label)
        if bbox is None:
            if logger:
                logger.warning(f"Диск {disk_label}: не удалось найти bounding box")
            return None
        bbox_center = [((sl.start + sl.stop) // 2) for sl in bbox]
        slices = get_centered_slices(bbox_center, crop_shape, mask_data.shape)
        mri_crop = crop_and_pad(mri_data, slices, crop_shape)
        mask_crop = crop_and_pad(mask_data, slices, crop_shape)
        angle_deg = compute_principal_angle(mask_crop == disk_label)
        rotated_mri, rotated_mask = rotate_volume_and_mask(mri_crop, mask_crop, angle_deg)

        mean = rotated_mri.mean()
        std = rotated_mri.std() if rotated_mri.std() > 0 else 1.0
        img = (rotated_mri - mean) / std
        img = img[np.newaxis, ...]
        img = torch.tensor(img).unsqueeze(0).float().to('cuda')
        model.to('cuda')
        model.eval()
        with torch.no_grad():
            grading_outputs = model(img)
        
        # Получаем предсказания как скалярные значения
        predictions = [torch.argmax(output).detach().cpu().numpy().item() for output in grading_outputs]
        
        if logger:
            logger.info(f"Диск {disk_label}: исходные предсказания модели = {predictions}")
        
        # Получаем spacing из nifti файла (в мм)
        spacing_mm = tuple(nifti_img.header.get_zooms()[:3])  # Берем первые 3 измерения
        if logger:
            logger.info(f"Диск {disk_label}: spacing = {spacing_mm} мм")
        
        # Если модель предсказала листез, измеряем его в миллиметрах
        spondylolisthesis_predicted = predictions[IDX_SPONDY]  # Индекс 3 - Spondylolisthesis
        if logger:
            logger.info(f"Диск {disk_label}: модель предсказала листез = {spondylolisthesis_predicted}")
        
        if spondylolisthesis_predicted > 0:
            spondylolisthesis_mm = measure_spondylolisthesis(rotated_mri, rotated_mask, disk_label, spacing_mm=spacing_mm, logger=logger)
            # Заменяем бинарное предсказание на измеренное значение в мм
            predictions[IDX_SPONDY] = spondylolisthesis_mm
            if logger:
                logger.info(f"Диск {disk_label}: измеренный листез = {spondylolisthesis_mm:.1f} мм")
        else:
            if logger:
                logger.info(f"Диск {disk_label}: модель не предсказала листез")
        
        # Если модель предсказала грыжу или выбухание, определяем их локализацию
        herniation_predicted = predictions[IDX_HERN]  # Индекс 4 - Disc herniation
        bulging_predicted = predictions[IDX_BULGE]     # Индекс 6 - Disc bulging
        
        if logger:
            logger.info(f"Диск {disk_label}: модель предсказала грыжу = {herniation_predicted}, выбухание = {bulging_predicted}")
        
        herniation_volume = 0.0
        bulging_volume = 0.0
        herniation_size = 0.0
        bulging_size = 0.0
        
        # Создаем маски для грыж и выбуханий
        herniation_mask = np.zeros_like(rotated_mask, dtype=bool)
        bulging_mask = np.zeros_like(rotated_mask, dtype=bool)
        
        if herniation_predicted > 0 or bulging_predicted > 0:
            herniation_mask, bulging_mask, herniation_volume, bulging_volume, herniation_size, bulging_size = \
                detect_herniation_bulging(rotated_mri, rotated_mask, disk_label, spacing_mm=spacing_mm, logger=logger)
            
            # Помечаем на сегментации
            updated_mask = save_segmentation_with_herniations(
                rotated_mask, disk_label, herniation_mask, bulging_mask, output_dir, logger
            )
            
            # Сохраняем обновленную сегментацию для этого диска
            try:
                # Создаем nifti изображение из обновленной маски
                updated_nifti = Nifti1Image(updated_mask, nifti_img.affine, nifti_img.header)
                
                # Сохраняем в папку результатов
                seg_filename = f"seg_disk_{disk_label}_with_herniations.nii.gz"
                seg_path = output_dir / seg_filename
                save(updated_nifti, str(seg_path))
                
                if logger:
                    logger.info(f"Диск {disk_label}: сохранена сегментация с грыжами в {seg_path}")
                    
            except Exception as e:
                if logger:
                    logger.error(f"Ошибка при сохранении сегментации для диска {disk_label}: {e}")
        
        # Добавляем информацию о размерах к предсказаниям
        predictions.extend([herniation_volume, bulging_volume, herniation_size, bulging_size])
        
        if logger:
            logger.info(f"Диск {disk_label}: финальные предсказания = {predictions}")
            logger.info(f"Диск {disk_label}: объемы - грыжа: {herniation_volume:.1f} мм³, выбухание: {bulging_volume:.1f} мм³")
            logger.info(f"Диск {disk_label}: размеры - грыжа: {herniation_size:.1f} мм, выбухание: {bulging_size:.1f} мм")
            logger.info("-" * 50)
        
        # Возвращаем кортеж с предсказаниями и масками
        return (predictions, herniation_mask, bulging_mask)
        
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при обработке диска {disk_label}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def parse_args():
    """
    Парсинг аргументов командной строки.
    """
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''
            This script processes spinal MRI data using sagittal and axial DICOM images.
            It performs segmentation and saves the results to the specified output folder.
        '''),
        epilog=textwrap.dedent('''
            Example:
                python main.py input_sag/ input_ax/ output/
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--studies_folder', type=Path, help='The input DICOM folder containing the sagittal images.', default=Path(r'F:\WorkSpace\Z-Union\100 МРТ ПК\Зверева Татьяна Николаевна\DICOM\PA000000\ST000000'))
    parser.add_argument('--output', type=Path, help='The output folder where the segmentation results will be saved.', default=Path(r'./results'))
    return parser.parse_args()


def run_segmentation_pipeline(args, logger=None) -> tuple:
    """
    Основной пайплайн сегментации: загрузка, препроцессинг, сегментация, постобработка.
    Возвращает итоговые nifti_img, nifti_seg.
    """
    med_data, _ = load_study_dicoms(args.studies_folder, require_extensions=False)
    med_data = average4d(med_data)
    med_data = reorient_canonical(med_data)
    med_data = resample(med_data)
    # Получить актуальное correspondence после преобразований
    sag_img, ax_img = med_data[0], med_data[1]
    save(sag_img, 'sag.nii.gz')
    save(ax_img, 'ax.nii.gz')

    new_correspondence = recalculate_correspondence(sag_img, ax_img)

    nifti_img = Nifti1Image(med_data[0].dataobj[np.newaxis, ...], med_data[0].affine, med_data[0].header)
    preprocessor = DefaultPreprocessor()
    for step in ['step_1', 'step_2']:
        data, seg, properties  = preprocessor.run_case(nifti_img, transpose_forward=[0, 1, 2])
        img, slicer_revert_padding = pad_nd_image(data, SAG_PATCH_SIZE, 'constant', {"constant_values": 0}, True)
        slicers = internal_get_sliding_window_slicers(img.shape[1:])
        model = torch.load(rf'model/weights/sag_{step}.pth', weights_only=False)
        num_segmentation_heads = 9 if step == 'step_1' else 11
        predicted_logits = internal_predict_sliding_window_return_logits(
            img, slicers, model, patch_size=SAG_PATCH_SIZE, results_device='cuda', num_segmentation_heads=num_segmentation_heads)
        predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        predicted_logits = predicted_logits.detach().cpu().numpy()
        segmentation_reverted_cropping, predicted_probabilities = preprocessor.convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits)
        nifti_seg = largest_component(segmentation_reverted_cropping, binarize=True, dilate=DILATE_SIZE)
        iterative_label_params = dict(
            seg=nifti_seg,
            selected_disc_landmarks=[2, 5, 3, 4],
            disc_labels=[1, 2, 3, 4, 5],
            disc_landmark_labels=[2, 3, 4, 5],
            disc_landmark_output_labels=[63, 71, 91, 100],
        )
        if step == 'step_1':
            iterative_label_params.update(
                canal_labels=[8],
                canal_output_label=2,
                cord_labels=[9],
                cord_output_label=1,
                sacrum_labels=[6],
                sacrum_output_label=50,
                map_input_dict={7: 11},
            )
        else:
            iterative_label_params.update(
                vertebrae_labels=[7, 8, 9],
                vertebrae_landmark_output_labels=[13, 21, 41, 50],
                vertebrae_extra_labels=[6],
                canal_labels=[10],
                canal_output_label=2,
                cord_labels=[11],
                cord_output_label=1,
                sacrum_labels=[9],
                sacrum_output_label=50,
            )
        nifti_seg = iterative_label(**iterative_label_params)
        nifti_seg = fill_canal(nifti_seg, canal_label=2, cord_label=1)
        nifti_seg = transform_seg2image(med_data[0], nifti_seg)
        nifti_img = crop_image2seg(med_data[0], nifti_seg, margin=DEFAULT_CROP_MARGIN)
        nifti_seg = transform_seg2image(nifti_img, nifti_seg)
        if step == 'step_1':
            nifti_seg = extract_alternate(nifti_seg, labels=EXTRACT_LABELS_RANGE)
            img_data = np.asanyarray(nifti_img.dataobj)
            seg_data = np.asanyarray(nifti_seg.dataobj)
            assert img_data.shape == seg_data.shape, f"Shapes do not match: {img_data.shape} vs {seg_data.shape}"
            multi_channel = np.stack([img_data, seg_data], axis=0)
            nifti_img = Nifti1Image(multi_channel, nifti_img.affine, nifti_img.header)


    axial = Nifti1Image(med_data[1].dataobj[np.newaxis, ...], med_data[1].affine, med_data[1].header)

    data, seg, properties = preprocessor.run_case(axial, transpose_forward=[0, 1, 2])
    img, slicer_revert_padding = pad_nd_image(data, AX_PATCH_SIZE, 'constant', {"constant_values": 0}, True)
    slicers = internal_get_sliding_window_slicers(img.shape[1:], patch_size=AX_PATCH_SIZE)
    model = torch.load(r'model/weights/ax.pth', weights_only=False)
    predicted_logits = internal_predict_sliding_window_return_logits(
        img, slicers, model, patch_size=AX_PATCH_SIZE, results_device='cuda',
        num_segmentation_heads=4, mode='2d')
    predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
    predicted_logits = predicted_logits.detach().cpu().numpy()
    segmentation_reverted_cropping, predicted_probabilities = preprocessor.convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_logits)
    
    # Убедимся, что это числовой numpy-массив
    # Проверяем, является ли segmentation_reverted_cropping объектом Nifti1Image
    if hasattr(segmentation_reverted_cropping, 'get_fdata'):
        # Это Nifti1Image объект - извлекаем данные
        segmentation_reverted_cropping = segmentation_reverted_cropping.get_fdata()
        if logger:
            logger.info(f"Извлечены данные из Nifti1Image: форма {segmentation_reverted_cropping.shape}, тип {segmentation_reverted_cropping.dtype}")
    else:
        # Это обычный массив
        segmentation_reverted_cropping = np.asarray(segmentation_reverted_cropping)
    
    # Проверяем, что это не кортеж или список массивов
    if segmentation_reverted_cropping.dtype == object:
        # Если это объект, возможно это кортеж массивов - берем первый элемент
        if isinstance(segmentation_reverted_cropping, np.ndarray) and segmentation_reverted_cropping.size == 1:
            segmentation_reverted_cropping = segmentation_reverted_cropping.item()
        # Если все еще объект, попробуем извлечь данные
        if segmentation_reverted_cropping.dtype == object:
            if logger:
                logger.warning(f"segmentation_reverted_cropping имеет тип object, пытаемся извлечь данные")
            # Попробуем получить данные из первого элемента
            try:
                if hasattr(segmentation_reverted_cropping, 'shape'):
                    # Это уже массив, но с неправильным dtype
                    segmentation_reverted_cropping = segmentation_reverted_cropping.astype(np.float32)
                else:
                    # Это кортеж или список
                    segmentation_reverted_cropping = np.asarray(segmentation_reverted_cropping[0] if len(segmentation_reverted_cropping) > 0 else segmentation_reverted_cropping)
            except Exception as e:
                if logger:
                    logger.error(f"Не удалось преобразовать segmentation_reverted_cropping: {e}")
                # Создаем пустой массив как fallback
                segmentation_reverted_cropping = np.zeros((1, 1, 1), dtype=np.float32)

    # Создаем правильный Nifti1Image для axial сегментации
    ax_seg_nifti = Nifti1Image(segmentation_reverted_cropping, med_data[1].affine, med_data[1].header)
    
    # Проверяем данные перед сохранением
    ax_data = np.asanyarray(ax_seg_nifti.dataobj)
    if logger:
        logger.info(f"AX сегментация - форма: {ax_data.shape}, тип: {ax_data.dtype}")
        logger.info(f"AX сегментация - уникальные значения: {np.unique(ax_data)}")
        logger.info(f"AX сегментация - affine: {ax_seg_nifti.affine}")
    
    save(ax_seg_nifti, 'ax.nii.gz')
    if logger:
        logger.info("AX сегментация сохранена в ax.nii.gz")

    return nifti_img, nifti_seg


def save_grading_results(results, output_dir: Path, filename="grading_results.csv", logger=None):
    """
    Сохраняет результаты grading в CSV файл.
    
    :param results: Список результатов grading для каждого диска
    :param output_dir: Папка для сохранения
    :param filename: Имя файла CSV
    :param logger: Logger для записи сообщений
    """
    try:
        # Создаем DataFrame с результатами
        grading_data = []
        
        # Обрабатываем только успешно обработанные диски
        for i, result in enumerate(results):
            if result is not None and len(result) > 0:
                disk_label = LANDMARK_LABELS[i]
                
                # Модель возвращает 8 выходов + 4 дополнительных параметра:
                # [modic, up_endplate, low_endplate, spondy, hern, narrow, bulge, pfirrman, hern_vol, bulge_vol, hern_size, bulge_size]
                if len(result) >= 12:
                    modic_score = result[0]
                    up_endplate_score = result[1]
                    low_endplate_score = result[2]
                    spondylolisthesis_score = result[3]  # В миллиметрах (0 если нет листеза)
                    disc_herniation_score = result[4]
                    disc_narrowing_score = result[5]
                    disc_bulging_score = result[6]
                    pfirrman_grade = result[7]
                    herniation_volume = result[8]  # Объем грыжи в мм³
                    bulging_volume = result[9]     # Объем выбухания в мм³
                    herniation_size = result[10]   # Размер грыжи в мм
                    bulging_size = result[11]      # Размер выбухания в мм
                elif len(result) == 8:
                    # Старый формат без измерений
                    modic_score = result[0]
                    up_endplate_score = result[1]
                    low_endplate_score = result[2]
                    spondylolisthesis_score = result[3]
                    disc_herniation_score = result[4]
                    disc_narrowing_score = result[5]
                    disc_bulging_score = result[6]
                    pfirrman_grade = result[7]
                    herniation_volume = 0.0
                    bulging_volume = 0.0
                    herniation_size = 0.0
                    bulging_size = 0.0
                else:
                    # Fallback если модель возвращает только один score
                    modic_score = result[0] if isinstance(result, list) else result
                    up_endplate_score = 0
                    low_endplate_score = 0
                    spondylolisthesis_score = 0
                    disc_herniation_score = 0
                    disc_narrowing_score = 0
                    disc_bulging_score = 0
                    pfirrman_grade = result[0] if isinstance(result, list) else result
                    herniation_volume = 0.0
                    bulging_volume = 0.0
                    herniation_size = 0.0
                    bulging_size = 0.0
                
                grading_data.append({
                    'disk_label': disk_label,
                    'Modic': modic_score,
                    'UP endplate': up_endplate_score,
                    'LOW endplate': low_endplate_score,
                    'Spondylolisthesis_mm': spondylolisthesis_score,  # В миллиметрах
                    'Disc herniation': disc_herniation_score,
                    'Disc narrowing': disc_narrowing_score,
                    'Disc bulging': disc_bulging_score,
                    'Pfirrman grade': pfirrman_grade,
                    'Herniation_volume_mm3': herniation_volume,  # Объем грыжи в мм³
                    'Bulging_volume_mm3': bulging_volume,        # Объем выбухания в мм³
                    'Herniation_size_mm': herniation_size,       # Размер грыжи в мм
                    'Bulging_size_mm': bulging_size              # Размер выбухания в мм
                })
        
        # Создаем DataFrame
        df = pd.DataFrame(grading_data)
        
        # Добавляем описания дисков
        disk_descriptions = {
            63: 'C2-C3', 64: 'C3-C4', 65: 'C4-C5', 66: 'C5-C6', 67: 'C6-C7',
            71: 'C7-T1', 72: 'T1-T2', 73: 'T2-T3', 74: 'T3-T4', 75: 'T4-T5',
            76: 'T5-T6', 77: 'T6-T7', 78: 'T7-T8', 79: 'T8-T9', 80: 'T9-T10',
            81: 'T10-T11', 82: 'T11-T12', 91: 'T12-L1', 92: 'L1-L2', 93: 'L2-L3',
            94: 'L3-L4', 95: 'L4-L5', 96: 'L5-S1', 100: 'S1-S2'
        }
        
        df['disk_description'] = df['disk_label'].map(disk_descriptions)
        
        # Сохраняем в CSV
        output_path = output_dir / filename
        df.to_csv(output_path, index=False)
        if logger:
            logger.info(f"Результаты grading сохранены в: {output_path}")
        
        # Выводим краткую статистику
        if not df.empty:
            if logger:
                logger.info(f"Статистика grading:")
                logger.info(f"Успешно обработано дисков: {len(df)}")
            
            # Статистика по Pfirrman grade
            if 'Pfirrman grade' in df.columns:
                if logger:
                    logger.info(f"Средний Pfirrman grade: {df['Pfirrman grade'].mean():.2f}")
                    logger.info(f"Медианный Pfirrman grade: {df['Pfirrman grade'].median():.2f}")
                
                # Распределение по степеням Pfirrman
                pfirrman_counts = df['Pfirrman grade'].value_counts().sort_index()
                if logger:
                    logger.info("Распределение по Pfirrman grade:")
                    for grade, count in pfirrman_counts.items():
                        logger.info(f"  Степень {grade}: {count} дисков")
            
            # Статистика по листезу (в миллиметрах)
            if 'Spondylolisthesis_mm' in df.columns:
                spondy_positive = (df['Spondylolisthesis_mm'] > 0).sum()
                if logger:
                    logger.info(f"Листез обнаружен: {spondy_positive} дисков")
                if spondy_positive > 0:
                    spondy_values = df[df['Spondylolisthesis_mm'] > 0]['Spondylolisthesis_mm']
                    if logger:
                        logger.info(f"Средний листез: {spondy_values.mean():.1f} мм")
                        logger.info(f"Максимальный листез: {spondy_values.max():.1f} мм")
                        logger.info(f"Минимальный листез: {spondy_values.min():.1f} мм")
            
            # Статистика по грыжам
            if 'Disc herniation' in df.columns:
                herniation_positive = (df['Disc herniation'] > 0).sum()
                if logger:
                    logger.info(f"Грыжи дисков: {herniation_positive} дисков")
                if herniation_positive > 0 and 'Herniation_volume_mm3' in df.columns:
                    herniation_volumes = df[df['Disc herniation'] > 0]['Herniation_volume_mm3']
                    herniation_sizes = df[df['Disc herniation'] > 0]['Herniation_size_mm']
                    if logger:
                        logger.info(f"Средний объем грыж: {herniation_volumes.mean():.1f} мм³")
                        logger.info(f"Средний размер грыж: {herniation_sizes.mean():.1f} мм")
            
            # Статистика по выбуханиям
            if 'Disc bulging' in df.columns:
                bulging_positive = (df['Disc bulging'] > 0).sum()
                if logger:
                    logger.info(f"Выбухания дисков: {bulging_positive} дисков")
                if bulging_positive > 0 and 'Bulging_volume_mm3' in df.columns:
                    bulging_volumes = df[df['Disc bulging'] > 0]['Bulging_volume_mm3']
                    bulging_sizes = df[df['Disc bulging'] > 0]['Bulging_size_mm']
                    if logger:
                        logger.info(f"Средний объем выбуханий: {bulging_volumes.mean():.1f} мм³")
                        logger.info(f"Средний размер выбуханий: {bulging_sizes.mean():.1f} мм")
            
            if 'Disc narrowing' in df.columns:
                narrowing_positive = (df['Disc narrowing'] > 0).sum()
                if logger:
                    logger.info(f"Сужение дисков: {narrowing_positive} дисков")
                    
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при сохранении результатов grading: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")


def create_global_segmentation_with_herniations(nifti_seg: Nifti1Image, all_herniation_results: dict, logger=None) -> Nifti1Image:
    """
    Создает общую сегментацию со всеми найденными грыжами и выбуханиями.
    
    :param nifti_seg: Исходная сегментация
    :param all_herniation_results: Словарь с результатами грыж для каждого диска
    :param logger: Logger для записи сообщений
    :return: Обновленная сегментация с грыжами
    """
    try:
        # Получаем данные сегментации
        seg_data = np.asanyarray(nifti_seg.dataobj)
        updated_seg_data = seg_data.copy()
        
        total_herniations = 0
        total_bulgings = 0
        
        # Проходим по всем дискам с найденными грыжами/выбуханиями
        for disk_label, (herniation_mask, bulging_mask) in all_herniation_results.items():
            if np.any(herniation_mask):
                # Помечаем грыжи (метка 200)
                disk_voxels = (seg_data == disk_label)
                herniation_voxels = disk_voxels & herniation_mask
                updated_seg_data[herniation_voxels] = HERNIA_LABEL
                total_herniations += np.sum(herniation_voxels)
                
                if logger:
                    logger.info(f"Глобальная сегментация: диск {disk_label} - {np.sum(herniation_voxels)} вокселей грыжи")
            
            if np.any(bulging_mask):
                # Помечаем выбухания (метка 201), но не те, что уже грыжа
                disk_voxels = (seg_data == disk_label)
                bulging_voxels = disk_voxels & bulging_mask & ~herniation_mask
                updated_seg_data[bulging_voxels] = BULGING_LABEL
                total_bulgings += np.sum(bulging_voxels)
                
                if logger:
                    logger.info(f"Глобальная сегментация: диск {disk_label} - {np.sum(bulging_voxels)} вокселей выбухания")
        
        # Создаем новое nifti изображение
        updated_nifti = Nifti1Image(updated_seg_data, nifti_seg.affine, nifti_seg.header)
        
        if logger:
            logger.info(f"Глобальная сегментация: всего помечено {total_herniations} вокселей грыж и {total_bulgings} вокселей выбуханий")
        
        return updated_nifti
        
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при создании глобальной сегментации: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return nifti_seg


def main():
    """
    Основная точка входа: парсинг аргументов, запуск пайплайна, обработка и сохранение ROI для каждого диска.
    """
    try:
        args = parse_args()
        output_dir = args.output
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Инициализируем logger
        logger = setup_logger(output_dir)
        logger.info("Запуск анализа позвоночника")
        logger.info(f"Входная папка: sagittal={args.studies_folder}")
        logger.info(f"Выходная папка: {output_dir}")
        
        nifti_img, nifti_seg = run_segmentation_pipeline(args, logger)
        save(nifti_img, 'img.nii.gz')
        save(nifti_seg, 'seg.nii.gz')
        median_bbox = MEDIAN_BBOX
        median_bbox = median_bbox.astype(int)
        crop_shape = get_crop_shape(median_bbox)
        mri_data_sag = nifti_img.get_fdata().transpose(2, 1, 0)
        mask_data_sag = nifti_seg.get_fdata().transpose(2, 1, 0)

        model = torch.load('model/weights/grading.pth', weights_only=False)
        logger.info("Модель grading загружена")

        first_pass = False
        result = []
        all_herniation_results = {}  # Словарь для хранения результатов грыж всех дисков
        
        for disk_label in LANDMARK_LABELS:
            disk_result = process_disk(mri_data_sag, mask_data_sag, disk_label, crop_shape, nifti_img, nifti_seg, output_dir, model=model, logger=logger)
            if disk_result is None:
                continue
            if not first_pass and disk_result is not None:
                first_pass = True
                continue

            # Распаковываем результат
            predictions, herniation_mask, bulging_mask = disk_result
            result.append(predictions)
            
            # Сохраняем маски грыж/выбуханий для глобальной сегментации
            all_herniation_results[disk_label] = (herniation_mask, bulging_mask)

        # Создаем и сохраняем глобальную сегментацию с грыжами
        try:
            global_seg_with_herniations = create_global_segmentation_with_herniations(nifti_seg, all_herniation_results, logger)
            global_seg_path = output_dir / "seg_with_herniations.nii.gz"
            save(global_seg_with_herniations, str(global_seg_path))
            logger.info(f"Сохранена глобальная сегментация с грыжами: {global_seg_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении глобальной сегментации: {e}")

        # Сохраняем результаты grading в CSV
        save_grading_results(result, output_dir, logger=logger)
        logger.info("Анализ завершен успешно")
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Критическая ошибка в main: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            # Если logger не определен, используем print как fallback
            print(f"Критическая ошибка в main: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == '__main__':
    main()