import argparse
import textwrap
from pathlib import Path
import csv
import pandas as pd
import logging
import traceback
import time

import nibabel
import numpy as np
import torch
from nibabel.nifti1 import Nifti1Image
from nibabel.loadsave import save
from scipy.ndimage import affine_transform
from scipy.ndimage import binary_dilation
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_erosion, label
from scipy.ndimage import binary_closing, binary_opening
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

from utils import pad_nd_image, average4d, reorient_canonical, resample, DefaultPreprocessor, largest_component, iterative_label, transform_seg2image, extract_alternate, fill_canal, crop_image2seg, recalculate_correspondence
from model import internal_predict_sliding_window_return_logits, internal_get_sliding_window_slicers, GradingModel, BasicBlock, Bottleneck
from dicom_io import load_dicoms_from_folder, load_study_dicoms
from utils.constant import (
    LANDMARK_LABELS, VERTEBRA_DESCRIPTIONS, COLORS,
    DEFAULT_CROP_MARGIN, DEFAULT_CROP_SHAPE_PAD, HERN_THRESHOLD_STD, BULGE_THRESHOLD_STD, HERN_THRESHOLD_MAX, BULGE_THRESHOLD_MAX,
    MIN_HERNIA_SIZE, MIN_BULGING_SIZE, MAX_BULGING_SIZE, MIN_BULGING_SHAPE_SIZE, MAX_BULGING_SHAPE_SIZE, BULGING_SHAPE_THRESHOLD_STD,
    MIN_BULGING_COORDS, DILATE_SIZE, SPONDY_SIGNIFICANT_MM, VERTEBRA_SEARCH_RANGE, VERTEBRA_NEAR_DISK_DISTANCE,
    MAX_DISTANCE_FROM_DISK_BORDER, MIN_DISTANCE_FROM_DISK_CENTER, MIN_DISTANCE_FROM_DISK_BORDER,
    CANAL_LABEL, CORD_LABEL, SACRUM_LABEL, HERNIA_LABEL, BULGING_LABEL,
    SAG_PATCH_SIZE, AX_PATCH_SIZE, EXTRACT_LABELS_RANGE,
    IDX_MODIC, IDX_UP_ENDPLATE, IDX_LOW_ENDPLATE, IDX_SPONDY, IDX_HERN, IDX_NARROW, IDX_BULGE, IDX_PFIRRMAN,
    BULGING_SYMMETRY_CV, CROP_SHAPE
)

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
    # Убеждаемся, что маски имеют правильный тип
    component_mask = component_mask.astype(bool)
    disk_mask = disk_mask.astype(bool)

    dilated_component = binary_dilation(component_mask, iterations=1)
    return bool(np.any(dilated_component & disk_mask))

def deforms_canal(component_mask: np.ndarray, canal_mask: np.ndarray, rotated_mask: np.ndarray) -> bool:
    """
    Проверяет, пересекается ли компонент с каналом (деформирует ли канал).
    """
    # Убеждаемся, что маски имеют правильный тип
    component_mask = component_mask.astype(bool)
    canal_mask = canal_mask.astype(bool)

    return bool(np.any(component_mask & canal_mask))

def is_near_disk_boundary(coord: np.ndarray, disk_mask: np.ndarray, max_distance: int = MAX_DISTANCE_FROM_DISK_BORDER) -> bool:
    """
    Проверяет, находится ли координата близко к границе диска.
    :param coord: Координата для проверки
    :param disk_mask: Маска диска
    :param max_distance: Максимальное расстояние от границы в вокселях
    :return: True если координата близко к границе
    """
    disk_mask = disk_mask.astype(bool)
    disk_boundary = binary_dilation(disk_mask, iterations=1).astype(bool) & ~disk_mask
    if np.any(disk_boundary):
        x, y, z = [int(round(c)) for c in coord]
        shape = disk_mask.shape
        if 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]:
            point_mask = np.zeros_like(disk_mask, dtype=bool)
            point_mask[x, y, z] = True
            disk_boundary_bool = disk_boundary.astype(bool)
            distance_to_boundary = distance_transform_edt(np.logical_not(disk_boundary_bool))
            # Проверка на валидность массива и shape
            if (isinstance(distance_to_boundary, np.ndarray) and
                distance_to_boundary.shape == disk_mask.shape and
                distance_to_boundary.ndim == 3):
                distance = distance_to_boundary[x, y, z]
                return bool(distance <= max_distance)
            else:
                return False
    return False

def is_far_from_disk_center(coord: np.ndarray, disk_center: np.ndarray, min_distance: int = MIN_DISTANCE_FROM_DISK_CENTER) -> bool:
    """
    Проверяет, находится ли координата достаточно далеко от центра диска.

    :param coord: Координата для проверки
    :param disk_center: Центр диска
    :param min_distance: Минимальное расстояние от центра в вокселях
    :return: True если координата далеко от центра
    """
    distance_from_center = np.linalg.norm(coord - disk_center)
    return bool(distance_from_center >= min_distance)

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

def get_centered_slices(bbox_center: list, crop_shape: tuple, data_shape: tuple) -> tuple:
    """
    Возвращает tuple из срезов, центрированных по bbox_center, с формой crop_shape.
    """
    slices = []
    for dim in range(3):
        # Убеждаемся, что координаты являются числами
        center_coord = int(bbox_center[dim])
        crop_size = int(crop_shape[dim])
        data_size = int(data_shape[dim])

        start = center_coord - (crop_size // 2)
        stop = start + crop_size
        start = max(0, start)
        stop = min(data_size, stop)
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
    Сначала морфологический (геометрический) способ, если не найдено — fallback на старый (по яркости).
    Только задние/заднебоковые компоненты (угол < 60° к направлению к каналу).
    """
    try:
        disk_mask = (rotated_mask == disk_label)
        canal_mask = (rotated_mask == CANAL_LABEL)
        core = binary_erosion(disk_mask, iterations=2)
        protrusion = disk_mask & (~core.astype(bool))
        labeled, num = label(protrusion)
        herniation_mask = np.zeros_like(disk_mask, dtype=bool)
        bulging_mask = np.zeros_like(disk_mask, dtype=bool)
        # Вычисляем направление к каналу
        disk_coords = np.argwhere(disk_mask)
        disk_center = disk_coords.mean(axis=0)
        canal_coords = np.argwhere(canal_mask)
        if len(canal_coords) > 0:
            canal_center = canal_coords.mean(axis=0)
            direction_to_canal = canal_center - disk_center
            direction_to_canal = direction_to_canal / np.linalg.norm(direction_to_canal)
        else:
            direction_to_canal = np.array([0, 1, 0])
        angle_threshold_deg = 60
        for i in range(1, int(num)+1):
            comp = (labeled == i)
            size = np.sum(comp)
            if size < MIN_HERNIA_SIZE:
                continue
            # Центр компонента
            comp_coords = np.argwhere(comp)
            comp_center = comp_coords.mean(axis=0)
            vector = comp_center - disk_center
            if np.linalg.norm(vector) == 0:
                continue
            vector_norm = vector / np.linalg.norm(vector)
            # Угол между вектором к компоненту и направлением к каналу
            cos_angle = np.dot(vector_norm, direction_to_canal)
            angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            if angle_deg > angle_threshold_deg:
                # Не задний/заднебоковой компонент
                continue
            # Грыжа — контакт с каналом
            if np.any(binary_dilation(comp, iterations=1) & canal_mask):
                herniation_mask |= comp
                if logger:
                    logger.info(f'Грыжа (морфология): компонент {i}, размер {size}, угол {angle_deg:.1f}')
            elif size > MIN_BULGING_SIZE:
                bulging_mask |= comp
                if logger:
                    logger.info(f'Выбухание (морфология): компонент {i}, размер {size}, угол {angle_deg:.1f}')
        # Если что-то найдено — возвращаем
        if np.any(herniation_mask) or np.any(bulging_mask):
            voxel_volume_mm3 = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
            herniation_volume = np.sum(herniation_mask) * voxel_volume_mm3
            bulging_volume = np.sum(bulging_mask) * voxel_volume_mm3
            herniation_size = 0.0
            bulging_size = 0.0
            if np.any(herniation_mask):
                herniation_coords = np.argwhere(herniation_mask)
                herniation_center = herniation_coords.mean(axis=0)
                herniation_distances = np.linalg.norm(herniation_coords - herniation_center, axis=1)
                herniation_size = 2 * np.max(herniation_distances) * spacing_mm[0]
            if np.any(bulging_mask):
                bulging_coords = np.argwhere(bulging_mask)
                bulging_center = bulging_coords.mean(axis=0)
                bulging_distances = np.linalg.norm(bulging_coords - bulging_center, axis=1)
                bulging_size = 2 * np.max(bulging_distances) * spacing_mm[0]
            return herniation_mask, bulging_mask, herniation_volume, bulging_volume, herniation_size, bulging_size
        # Fallback: старый способ по яркости
        if logger:
            logger.info('Морфологический способ не дал результата, fallback на интенсивностный метод')
        return detect_herniation_bulging_intensity(rotated_mri, rotated_mask, disk_label, spacing_mm, logger)
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при определении грыжи/выбухания для диска {disk_label}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return (np.zeros_like(rotated_mask, dtype=bool),
                np.zeros_like(rotated_mask, dtype=bool),
                0.0, 0.0, 0.0, 0.0)


def detect_herniation_bulging_intensity(rotated_mri, rotated_mask, disk_label, spacing_mm, logger=None):
    try:
        # --- СТАРЫЙ КОД ПО ЯРКОСТИ ---
        # (скопировано из предыдущей версии detect_herniation_bulging)
        herniation_mask = np.zeros_like(rotated_mask, dtype=bool)
        bulging_mask = np.zeros_like(rotated_mask, dtype=bool)
        disk_mask = (rotated_mask == disk_label)
        if not np.any(disk_mask):
            if logger:
                logger.warning(f"Диск {disk_label}: диск не найден")
            return herniation_mask, bulging_mask, 0.0, 0.0, 0.0, 0.0
        canal_mask = (rotated_mask == CANAL_LABEL)
        disk_coords = np.argwhere(disk_mask)
        if len(disk_coords) == 0:
            if logger:
                logger.warning(f"Диск {disk_label}: координаты диска не найдены")
            return herniation_mask, bulging_mask, 0.0, 0.0, 0.0, 0.0
        disk_center = disk_coords.mean(axis=0)
        canal_coords = np.argwhere(canal_mask)
        if len(canal_coords) > 0:
            canal_center = canal_coords.mean(axis=0)
            direction_to_canal = canal_center - disk_center
            direction_to_canal = direction_to_canal / np.linalg.norm(direction_to_canal)
        else:
            direction_to_canal = np.array([0, 1, 0])
        mri_normalized = (rotated_mri - rotated_mri.min()) / (rotated_mri.max() - rotated_mri.min())
        posterior_disk = np.zeros_like(disk_mask, dtype=bool)
        for coord in disk_coords:
            relative_pos = coord - disk_center
            projection = np.dot(relative_pos, direction_to_canal)
            if projection > 0:
                posterior_disk[coord[0], coord[1], coord[2]] = True
        if not np.any(posterior_disk):
            if logger:
                logger.warning(f"Диск {disk_label}: задняя часть диска не найдена")
            return herniation_mask, bulging_mask, 0.0, 0.0, 0.0, 0.0
        posterior_intensities = []
        posterior_coords = []
        for coord in np.argwhere(posterior_disk):
            x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
            if 0 <= x < mri_normalized.shape[0] and 0 <= y < mri_normalized.shape[1] and 0 <= z < mri_normalized.shape[2]:
                intensity = mri_normalized[x, y, z]
                posterior_intensities.append(intensity)
                posterior_coords.append(coord)
        if not posterior_intensities:
            if logger:
                logger.warning(f"Диск {disk_label}: интенсивности в задней части не найдены")
            return herniation_mask, bulging_mask, 0.0, 0.0, 0.0, 0.0
        mean_intensity = np.mean(posterior_intensities)
        std_intensity = np.std(posterior_intensities)
        if logger:
            logger.info(f"Диск {disk_label}: средняя интенсивность = {mean_intensity:.3f}, std = {std_intensity:.3f}")
        herniation_threshold = min(mean_intensity + HERN_THRESHOLD_STD * std_intensity, HERN_THRESHOLD_MAX)
        bulging_threshold = min(mean_intensity + BULGE_THRESHOLD_STD * std_intensity, BULGE_THRESHOLD_MAX)
        if logger:
            logger.info(f"Диск {disk_label}: пороги - грыжа: {herniation_threshold:.3f}, выбухание: {bulging_threshold:.3f}")
        herniation_candidates = np.zeros_like(disk_mask, dtype=bool)
        bulging_candidates = np.zeros_like(disk_mask, dtype=bool)
        for i, coord in enumerate(posterior_coords):
            intensity = posterior_intensities[i]
            x, y, z = int(coord[0]), int(coord[1]), int(coord[2])
            if not is_near_disk_boundary(coord, disk_mask, max_distance=MAX_DISTANCE_FROM_DISK_BORDER):
                continue
            if not is_far_from_disk_center(coord, disk_center):
                continue
            if intensity > herniation_threshold:
                herniation_candidates[x, y, z] = True
            elif intensity > bulging_threshold:
                bulging_candidates[x, y, z] = True
        if np.any(herniation_candidates):
            herniation_mask = detect_continuous_herniation(
                herniation_candidates, disk_mask, canal_mask, rotated_mask, logger
            )
        if np.any(bulging_candidates):
            # Для fallback используем простое морфологическое объединение
            bulging_mask = bulging_candidates & ~herniation_mask
        voxel_volume_mm3 = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
        herniation_volume = np.sum(herniation_mask) * voxel_volume_mm3
        bulging_volume = np.sum(bulging_mask) * voxel_volume_mm3
        herniation_size = 0.0
        bulging_size = 0.0
        if np.any(herniation_mask):
            herniation_coords = np.argwhere(herniation_mask)
            herniation_center = herniation_coords.mean(axis=0)
            herniation_distances = np.linalg.norm(herniation_coords - herniation_center, axis=1)
            herniation_size = 2 * np.max(herniation_distances) * spacing_mm[0]
        if np.any(bulging_mask):
            bulging_coords = np.argwhere(bulging_mask)
            bulging_center = bulging_coords.mean(axis=0)
            bulging_distances = np.linalg.norm(bulging_coords - bulging_center, axis=1)
            bulging_size = 2 * np.max(bulging_distances) * spacing_mm[0]
        return herniation_mask, bulging_mask, herniation_volume, bulging_volume, herniation_size, bulging_size
    except Exception as e:
        if logger:
            logger.error(f"Ошибка в fallback (интенсивностный) методе для диска {disk_label}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return (np.zeros_like(rotated_mask, dtype=bool),
                np.zeros_like(rotated_mask, dtype=bool),
                0.0, 0.0, 0.0, 0.0)


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


def transform_mask_to_global(rotated_mask: np.ndarray, original_slices: tuple, original_shape: tuple,
                           angle_deg: float, bbox_center: list, crop_shape: tuple) -> np.ndarray:
    """
    Трансформирует маску из локальных координат обратно в глобальные координаты.

    :param rotated_mask: Маска в повернутых координатах
    :param original_slices: Срезы, использованные для кропа
    :param original_shape: Исходная форма данных
    :param angle_deg: Угол поворота в градусах
    :param bbox_center: Центр bounding box
    :param crop_shape: Форма кропа
    :return: Маска в глобальных координатах
    """
    try:
        # Создаем пустую маску в глобальных координатах
        global_mask = np.zeros(original_shape, dtype=bool)

        # Обратный поворот маски
        angle_rad = np.radians(angle_deg)  # Положительный угол для обратного поворота
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad),  np.cos(angle_rad), 0],
            [0,                0,                1]
        ])

        # Применяем обратный поворот
        center = 0.5 * np.array(rotated_mask.shape)
        offset = center - rotation_matrix @ center
        unrotated_mask = affine_transform(rotated_mask.astype(float), rotation_matrix,
                                        offset=offset.tolist(), order=0)
        unrotated_mask = unrotated_mask > 0.5  # Преобразуем обратно в булев массив

        # Размещаем в глобальной маске по исходным срезам
        global_mask[original_slices] = unrotated_mask

        return global_mask

    except Exception as e:
        # В случае ошибки возвращаем пустую маску
        return np.zeros(original_shape, dtype=bool)


def process_disk(mri_data: np.ndarray, mask_data: np.ndarray, disk_label: int, crop_shape: tuple, nifti_img: Nifti1Image, nifti_seg: Nifti1Image, output_dir: Path, model, logger=None):
    """
    Кроп, выравнивание и сохранение ROI для одного диска.
    Возвращает (predictions, herniation_mask_global, bulging_mask_global) или None при ошибке.
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

        create_sagittal_bmp_images(rotated_mri, rotated_mask, output_dir, logger, slice_axis=0, variation=0, show_labels=False)

        mean = rotated_mri.mean()
        std = rotated_mri.std() if rotated_mri.std() > 0 else 1.0
        img = (rotated_mri - mean) / std
        img = img[np.newaxis, ...]
        img = torch.tensor(img).unsqueeze(0).float().to(DEVICE)
        model.to(DEVICE)
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

        # Трансформируем маски обратно в глобальные координаты
        herniation_mask_global = transform_mask_to_global(
            herniation_mask, slices, mask_data.shape, angle_deg, bbox_center, crop_shape
        )
        bulging_mask_global = transform_mask_to_global(
            bulging_mask, slices, mask_data.shape, angle_deg, bbox_center, crop_shape
        )

        # Добавляем информацию о размерах к предсказаниям
        predictions.extend([herniation_volume, bulging_volume, herniation_size, bulging_size])

        if logger:
            logger.info(f"Диск {disk_label}: финальные предсказания = {predictions}")
            logger.info(f"Диск {disk_label}: объемы - грыжа: {herniation_volume:.1f} мм³, выбухание: {bulging_volume:.1f} мм³")
            logger.info(f"Диск {disk_label}: размеры - грыжа: {herniation_size:.1f} мм, выбухание: {bulging_size:.1f} мм")
            logger.info("-" * 50)

        # Возвращаем кортеж с предсказаниями и масками в глобальных координатах
        return (predictions, herniation_mask_global, bulging_mask_global)

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
    parser.add_argument('--studies_folder', type=Path, help='The input DICOM folder containing the sagittal images.', default=Path(r'ST000000'))
    parser.add_argument('--output', type=Path, help='The output folder where the segmentation results will be saved.', default=Path(r'./results'))
    return parser.parse_args()

def pick_best(scan_tuple):
    # Сначала ищем T1
    if scan_tuple[0] is not None:  # T1
        return scan_tuple[0]

    # Затем ищем T2 без подавления жира (приоритет над T2 с FS)
    if scan_tuple[1] is not None:  # T2
        return scan_tuple[1]

    # Наконец, ищем STIR
    if scan_tuple[2] is not None:  # STIR
        return scan_tuple[2]

    return None

def run_segmentation_pipeline(args, logger=None) -> tuple:
    """
    Основной пайплайн сегментации: загружает, препроцессит и сегментирует только лучшие доступные контрасты для сагиттальной и аксиальной проекций (T1 > T2 > STIR).
    Если проекция отсутствует — пишет в лог и пропускает.
    Возвращает кортеж (nifti_img, nifti_seg) для сагиттали или аксиала (что есть), либо (None, None) если ничего не найдено.
    """
    # Загрузка всех доступных проекций и контрастов
    all_scans, _ = load_study_dicoms(args.studies_folder, require_extensions=False)
    sag_scans, ax_scans, _ = all_scans  # (T1, T2, STIR) для каждой проекции

    # Приоритет: T1 > T2 (без FS) > T2 (с FS) > STIR

    sag_scans = average4d(sag_scans)
    ax_scans = average4d(ax_scans)

    sag_scans = reorient_canonical(sag_scans)
    ax_scans = reorient_canonical(ax_scans)

    sag_scans = resample(sag_scans)
    ax_scans = resample(ax_scans)


    sag_scan = pick_best(sag_scans)
    ax_scan = pick_best(ax_scans)

    if sag_scan is None and ax_scan is None:
        if logger:
            logger.error("Нет доступных сагиттальных или аксиальных сканов для сегментации!")
        return None, None

    preprocessor = DefaultPreprocessor()
    nifti_img = None
    nifti_seg = None

    # Сегментация сагиттала, если есть
    if sag_scan is not None:
        if logger:
            logger.info("Выбран сагиттальный скан для сегментации.")

        save(sag_scan, 'sagittal_original.nii.gz')
        nifti_img = Nifti1Image(sag_scan.get_fdata()[np.newaxis, ...], sag_scan.affine, sag_scan.header)
        for step in ['step_1', 'step_2']:
            data, seg, properties = preprocessor.run_case(nifti_img, transpose_forward=[0, 1, 2])
            img, slicer_revert_padding = pad_nd_image(data, SAG_PATCH_SIZE, 'constant', {"constant_values": 0}, True)
            slicers = internal_get_sliding_window_slicers(img.shape[1:])
            model = torch.load(rf'model/weights/sag_{step}.pth', weights_only=False)
            num_segmentation_heads = 9 if step == 'step_1' else 11
            predicted_logits = internal_predict_sliding_window_return_logits(
                img, slicers, model, patch_size=SAG_PATCH_SIZE, results_device=str(DEVICE), num_segmentation_heads=num_segmentation_heads)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
            predicted_logits = predicted_logits.detach().cpu().numpy()
            segmentation_reverted_cropping, predicted_probabilities = preprocessor.convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits)
            # Ensure seg is always a Nifti1Image
            if not isinstance(segmentation_reverted_cropping, Nifti1Image):
                seg_nifti = Nifti1Image(segmentation_reverted_cropping.astype(np.uint8), sag_scan.affine, sag_scan.header)
            else:
                seg_nifti = segmentation_reverted_cropping
            nifti_seg = largest_component(seg_nifti, binarize=True, dilate=DILATE_SIZE)
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
            nifti_seg = transform_seg2image(sag_scan, nifti_seg)
            sagittals = [crop_image2seg(sag, nifti_seg, margin=DEFAULT_CROP_MARGIN) if sag is not None else None for sag in sag_scans]
            # nifti_img = crop_image2seg(sag_scan, nifti_seg, margin=DEFAULT_CROP_MARGIN)
            nifti_img = pick_best(sagittals)
            nifti_seg = transform_seg2image(nifti_img, nifti_seg)
            if step == 'step_1':
                nifti_seg = extract_alternate(nifti_seg, labels=EXTRACT_LABELS_RANGE)
                img_data = np.asanyarray(nifti_img.dataobj)
                seg_data = np.asanyarray(nifti_seg.dataobj)
                assert img_data.shape == seg_data.shape, f"Shapes do not match: {img_data.shape} vs {seg_data.shape}"
                multi_channel = np.stack([img_data, seg_data], axis=0)
                nifti_img = Nifti1Image(multi_channel, nifti_img.affine, nifti_img.header)
        save(nifti_img, 'sagittal_processed.nii.gz')
        save(nifti_seg, 'sagittal_segmentation.nii.gz')
        if logger:
            logger.info("Обработанные сагиттальные файлы сохранены: sagittal_processed.nii.gz, sagittal_segmentation.nii.gz")

        # sagittals = [crop_image2seg(sag, nifti_seg, margin=DEFAULT_CROP_MARGIN) for sag in sag_scans if sag is not None]
    else:
        if logger:
            logger.warning("Сагиттальный скан не найден. Сегментация сагиттала пропущена.")

    # Сегментация аксиала, если есть
    if ax_scan is not None:
        if logger:
            logger.info("Выбран аксиальный скан для сегментации.")
        save(ax_scan, 'axial_original.nii.gz')
        axial = Nifti1Image(ax_scan.get_fdata()[np.newaxis, ...], ax_scan.affine, ax_scan.header)
        data, seg, properties = preprocessor.run_case(axial, transpose_forward=[0, 1, 2])
        img, slicer_revert_padding = pad_nd_image(data, AX_PATCH_SIZE, 'constant', {"constant_values": 0}, True)
        slicers = internal_get_sliding_window_slicers(img.shape[1:], patch_size=AX_PATCH_SIZE)
        model = torch.load(r'model/weights/ax.pth', weights_only=False)
        predicted_logits = internal_predict_sliding_window_return_logits(
            img, slicers, model, patch_size=AX_PATCH_SIZE, results_device=str(DEVICE),
            num_segmentation_heads=5, mode='2d', use_gaussian=False)
        predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        predicted_logits = predicted_logits.detach().cpu().numpy()
        segmentation_reverted_cropping, predicted_probabilities = preprocessor.convert_predicted_logits_to_segmentation_with_correct_shape(
            predicted_logits)
        # Ensure seg is always a Nifti1Image
        if not isinstance(segmentation_reverted_cropping, Nifti1Image):
            seg_nifti = Nifti1Image(segmentation_reverted_cropping.astype(np.uint8), ax_scan.affine, ax_scan.header)
        else:
            seg_nifti = segmentation_reverted_cropping
        processed_axial_data = np.asanyarray(data)
        processed_axial = Nifti1Image(processed_axial_data, ax_scan.affine, ax_scan.header)
        ax_seg_nifti = Nifti1Image(np.asanyarray(seg_nifti.dataobj), ax_scan.affine, ax_scan.header)
        ax_data = np.asanyarray(ax_seg_nifti.dataobj)
        if logger:
            logger.debug(f"AX сегментация - форма: {ax_data.shape}, тип: {ax_data.dtype}")
            logger.debug(f"AX сегментация - уникальные значения: {np.unique(ax_data)}")
            logger.debug(f"AX сегментация - affine: {ax_seg_nifti.affine}")
            logger.debug(f"AX обработанное - форма: {processed_axial_data.shape}, affine: {processed_axial.affine}")
        save(processed_axial, 'axial_processed.nii.gz')
        if logger:
            logger.info("Обработанное AX изображение сохранено в axial_processed.nii.gz")
        save(ax_seg_nifti, 'axial_segmentation.nii.gz')
        if logger:
            logger.info("AX сегментация сохранена в axial_segmentation.nii.gz")
    else:
        if logger:
            logger.warning("Аксиальный скан не найден. Сегментация аксиала пропущена.")

    # Возвращаем только то, что есть (приоритет: сагиттал -> аксиал)
    if sagittals is not None and nifti_seg is not None:
        return sagittals, nifti_seg
    elif ax_scan is not None:
        # Вернуть аксиальные результаты (если нужно)
        return axial, ax_seg_nifti
    else:
        return None, None


def save_grading_results(results, processed_disks, output_dir: Path, filename="grading_results.csv", logger=None):
    """
    Сохраняет результаты grading в CSV файл.

    :param results: Список результатов grading для каждого диска
    :param processed_disks: Список успешно обработанных дисков
    :param output_dir: Папка для сохранения
    :param filename: Имя файла CSV
    :param logger: Logger для записи сообщений
    """
    try:
        # Создаем DataFrame с результатами
        grading_data = []

        # Добавляем описания дисков
        disk_descriptions = VERTEBRA_DESCRIPTIONS

        # Обрабатываем только успешно обработанные диски
        for i, disk_label in enumerate(processed_disks):
            if results[i] is not None and len(results[i]) > 0:
                if logger:
                    logger.info(f"Обрабатываем результат для диска {disk_label} (индекс {i})")

                # Модель возвращает 8 выходов + 4 дополнительных параметра:
                # [modic, up_endplate, low_endplate, spondy, hern, narrow, bulge, pfirrman, hern_vol, bulge_vol, hern_size, bulge_size]
                if len(results[i]) >= 12:
                    modic_score = results[i][0]
                    up_endplate_score = results[i][1]
                    low_endplate_score = results[i][2]
                    spondylolisthesis_score = results[i][3]  # В миллиметрах (0 если нет листеза)
                    disc_herniation_score = results[i][4]
                    disc_narrowing_score = results[i][5]
                    disc_bulging_score = results[i][6]
                    pfirrman_grade = results[i][7]
                    herniation_volume = results[i][8]  # Объем грыжи в мм³
                    bulging_volume = results[i][9]     # Объем выбухания в мм³
                    herniation_size = results[i][10]   # Размер грыжи в мм
                    bulging_size = results[i][11]      # Размер выбухания в мм
                elif len(results[i]) == 8:
                    # Старый формат без измерений
                    modic_score = results[i][0]
                    up_endplate_score = results[i][1]
                    low_endplate_score = results[i][2]
                    spondylolisthesis_score = results[i][3]
                    disc_herniation_score = results[i][4]
                    disc_narrowing_score = results[i][5]
                    disc_bulging_score = results[i][6]
                    pfirrman_grade = results[i][7]
                    herniation_volume = 0.0
                    bulging_volume = 0.0
                    herniation_size = 0.0
                    bulging_size = 0.0
                else:
                    # Fallback если модель возвращает только один score
                    modic_score = results[i][0] if isinstance(results[i], list) else results[i]
                    up_endplate_score = 0
                    low_endplate_score = 0
                    spondylolisthesis_score = 0
                    disc_herniation_score = 0
                    disc_narrowing_score = 0
                    disc_bulging_score = 0
                    pfirrman_grade = results[i][0] if isinstance(results[i], list) else results[i]
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

                if logger:
                    logger.info(f"Добавлен в CSV: диск {disk_label} -> {disk_descriptions.get(disk_label, 'Неизвестный диск')}")
            else:
                if logger:
                    logger.warning(f"Пропущен результат для индекса {i}: result is None or empty")

        # Создаем DataFrame
        df = pd.DataFrame(grading_data)

        # Добавляем описания дисков
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
            # Проверяем, что маски имеют правильную форму
            if herniation_mask.shape != seg_data.shape or bulging_mask.shape != seg_data.shape:
                if logger:
                    logger.warning(f"Диск {disk_label}: маски имеют неправильную форму. "
                                 f"Ожидается {seg_data.shape}, получено {herniation_mask.shape}, {bulging_mask.shape}")
                    logger.warning(f"Пропускаем диск {disk_label} из-за несоответствия размеров")
                continue

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


def detect_continuous_herniation(herniation_candidates: np.ndarray, disk_mask: np.ndarray, canal_mask: np.ndarray, rotated_mask: np.ndarray, logger=None) -> np.ndarray:
    try:
        from scipy.ndimage import label, binary_closing, binary_opening
        labeled_candidates, num_features = label(herniation_candidates)
        if logger:
            logger.info(f"Найдено {num_features} компонентов-кандидатов в грыжу")
        valid_herniations = np.zeros_like(herniation_candidates, dtype=bool)
        if num_features is not None and num_features > 0:
            for component_id in range(1, int(num_features) + 1):
                component_mask = (labeled_candidates == component_id)
                component_size = np.sum(component_mask)
                if component_size < MIN_HERNIA_SIZE:
                    if logger:
                        logger.info(f"Компонент {component_id} слишком мал ({component_size} вокселей)")
                    continue
                if deforms_canal(component_mask, canal_mask, rotated_mask):
                    valid_herniations |= component_mask
                    if logger:
                        logger.info(f"Компонент {component_id} деформирует канал - принят как грыжа")
                else:
                    if logger:
                        logger.info(f"Компонент {component_id} не деформирует канал - отклонен")
        if np.any(valid_herniations):
            valid_herniations = binary_closing(valid_herniations, iterations=1)
            valid_herniations = binary_opening(valid_herniations, iterations=1)
        return valid_herniations
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при обнаружении непрерывной грыжи: {e}")
        return np.zeros_like(herniation_candidates, dtype=bool)


def create_sagittal_bmp_images(nifti_img: Nifti1Image, nifti_seg: Nifti1Image, output_dir: Path, logger=None, slice_axis=0, variation=0, show_labels=True):
    """
    Создает BMP изображения с подписанными позвонками для каждого среза.
    
    :param nifti_img: NIfTI изображение МРТ
    :param nifti_seg: NIfTI изображение сегментации
    :param output_dir: Папка для сохранения результатов
    :param logger: Logger для записи сообщений
    :param slice_axis: Ось для нарезки (0=X, 1=Y, 2=Z), по умолчанию 0
    :param variation: Вариация отображения (0=контуры, 1=заливка, 2=контуры+заливка)
    :param show_labels: Показывать ли подписи (позвонки и справочная информация)
    """
    try:
        # Создаем папку для BMP изображений
        bmp_dir = output_dir / "segments" / "sag"
        bmp_dir.mkdir(parents=True, exist_ok=True)
        
        if logger:
            logger.info(f"Создаем BMP изображения в папке: {bmp_dir}")
            logger.info(f"Ось нарезки: {slice_axis}, вариация: {variation}")

        # Получаем данные
        if hasattr(nifti_img, 'get_fdata'):
            img_data = nifti_img.get_fdata()
        else:
            img_data = np.asarray(nifti_img)
        if hasattr(nifti_seg, 'get_fdata'):
            seg_data = nifti_seg.get_fdata()
        else:
            seg_data = np.asarray(nifti_seg)
        
        # Словарь для описания позвонков (не дисков)
        vertebra_descriptions = {
            11: "C1",
            12: "C2",
            13: "C3",
            14: "C4",
            15: "C5",
            16: "C6",
            17: "C7",
            21: "Th1",
            22: "Th2",
            23: "Th3",
            24: "Th4",
            25: "Th2",
            26: "Th6",
            27: "Th7",
            28: "Th8",
            29: "Th9",
            30: "Th10",
            31: "Th11",
            32: "Th12",
            41: "L1",
            42: "L2",
            43: "L3",
            44: "L4",
            45: "L5",
            50: "S1",
        }
        
        # Цвета для разных типов структур
        colors = COLORS
        
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
                # Отражаем по горизонтали, чтобы спина смотрела влево
                mri_normalized = np.fliplr(mri_normalized)
                seg_slice = np.fliplr(seg_slice)
                
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
                
                if show_labels:
                    # Пытаемся загрузить шрифт, если не получится - используем стандартный
                    try:
                        font_large = ImageFont.truetype("arial.ttf", 12)  # Для позвонков (уменьшено)
                        font_small = ImageFont.truetype("arial.ttf", 10)  # Для остального
                    except:
                        font_large = ImageFont.load_default()
                        font_small = ImageFont.load_default()
                    
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
                                text_bbox = draw.textbbox((0, 0), description, font=font_large)
                                text_width = text_bbox[2] - text_bbox[0]
                                text_height = text_bbox[3] - text_bbox[1]
                                text_x = max(0, center_x - text_width // 2)
                                text_y = max(0, center_y - text_height // 2)
                                draw.rectangle([text_x-2, text_y-2, text_x+text_width+2, text_y+text_height+2], fill=(0, 0, 0), outline=(255, 255, 255))
                                draw.text((text_x, text_y), description, fill=(255, 255, 255), font=font_large)
                                found_vertebrae.append(description)
                    # --- Справочная информация в нижнем левом углу ---
                    axis_names = ['X', 'Y', 'Z']
                    slice_info = f"Slice {slice_idx+1}/{num_slices} ({axis_names[slice_axis]})"
                    variation_names = ['Контуры', 'Заливка', 'Контуры+Заливка']
                    variation_info = f"Вариация: {variation_names[variation]}"
                    vertebrae_info = f"Found: {', '.join(found_vertebrae)}" if found_vertebrae else ""
                    info_lines = [slice_info, variation_info]
                    if vertebrae_info:
                        info_lines.append(vertebrae_info)
                    img_w, img_h = pil_image.size
                    margin = 8
                    line_height = font_small.getbbox("Ag")[3] - font_small.getbbox("Ag")[1] + 2
                    total_height = line_height * len(info_lines)
                    info_x = margin
                    info_y = img_h - total_height - margin
                    for i, line in enumerate(info_lines):
                        y = info_y + i * line_height
                        text_bbox = draw.textbbox((0, 0), line, font=font_small)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([info_x-2, y-2, info_x+text_width+2, y+text_height+2], fill=(0,0,0), outline=None)
                        draw.text((info_x, y), line, fill=(255,255,255), font=font_small)
                # Если show_labels == False, не рисуем подписи вообще
                
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


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    else:
        return torch.device('cpu')

DEVICE = get_device()

def main():
    """
    Основная точка входа: парсинг аргументов, запуск пайплайна, обработка и сохранение ROI для каждого диска.
    """
    start_time = time.time()
    try:
        args = parse_args()
        output_dir = args.output
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Инициализируем logger
        logger = setup_logger(output_dir)
        logger.info("Запуск анализа позвоночника")
        logger.info(f"Входная папка: {args.studies_folder}")
        logger.info(f"Выходная папка: {output_dir}")
        logger.info(f"Тип устройства: {DEVICE}")
        
        nifti_img, nifti_seg = run_segmentation_pipeline(args, logger)
        if logger:
            logger.info("Обработанные сагиттальные файлы сохранены: sagittal_processed.nii.gz, sagittal_segmentation.nii.gz")
        nifti_img = nifti_img[1]
        # Анализируем, какие диски присутствуют в сегментации
        seg_data = nifti_seg.get_fdata()
        unique_labels = np.unique(seg_data)
        if logger:
            logger.info(f"Уникальные метки в сегментации: {unique_labels}")
            logger.info(f"Ожидаемые метки дисков: {LANDMARK_LABELS}")
        
        # Проверяем, какие из ожидаемых дисков присутствуют
        present_disks = [label for label in LANDMARK_LABELS if label in unique_labels]
        if logger:
            logger.info(f"Присутствующие диски: {present_disks}")
            for disk_label in present_disks:
                disk_description = VERTEBRA_DESCRIPTIONS.get(disk_label, 'Неизвестный')
                logger.info(f"  Диск {disk_label} -> {disk_description}")
        
        # Проверяем отсутствующие диски
        missing_disks = [label for label in LANDMARK_LABELS if label not in unique_labels]
        if missing_disks:
            logger.info(f"Отсутствующие диски: {missing_disks}")
            for disk_label in missing_disks:
                disk_description = VERTEBRA_DESCRIPTIONS.get(disk_label, 'Неизвестный')
                logger.info(f"  Отсутствует диск {disk_label} -> {disk_description}")

        crop_shape = CROP_SHAPE
        # Убираем транспонирование, чтобы маски имели правильную форму
        mri_data_sag = nifti_img.get_fdata()
        mask_data_sag = nifti_seg.get_fdata()

        model = torch.load('model/weights/grading.pth', weights_only=False)
        logger.info("Модель grading загружена")

        first_pass = False  # Флаг для пропуска первого диска (может быть неполным из-за обрезки МРТ)
        result = []
        all_herniation_results = {}  # Словарь для хранения результатов грыж всех дисков
        processed_disks = []  # Список успешно обработанных дисков
        
        for disk_label in LANDMARK_LABELS:
            if logger:
                logger.info(f"Обрабатываем диск с меткой: {disk_label}")
            
            disk_result = process_disk(mri_data_sag, mask_data_sag, disk_label, crop_shape, nifti_img, nifti_seg, output_dir, model=model, logger=logger)
            if disk_result is None:
                if logger:
                    logger.warning(f"Диск {disk_label} не был обработан (результат None)")
                continue

            # Пропускаем первый успешно обработанный диск (может быть неполным из-за обрезки МРТ)
            if not first_pass:
                first_pass = True
                if logger:
                    logger.info(f"Пропускаем первый диск {disk_label} (может быть неполным из-за обрезки МРТ)")
                continue

            # Распаковываем результат
            predictions, herniation_mask_global, bulging_mask_global = disk_result
            result.append(predictions)
            processed_disks.append(disk_label)  # Добавляем в список обработанных дисков
            
            # Сохраняем маски грыж/выбуханий для глобальной сегментации
            all_herniation_results[disk_label] = (herniation_mask_global, bulging_mask_global)
            
            if logger:
                logger.info(f"Диск {disk_label} успешно обработан и добавлен в результаты")

        # Создаем и сохраняем глобальную сегментацию с грыжами
        try:
            global_seg_with_herniations = create_global_segmentation_with_herniations(nifti_seg, all_herniation_results, logger)
            global_seg_path = output_dir / "seg_with_herniations.nii.gz"
            save(global_seg_with_herniations, str(global_seg_path))
            logger.info(f"Сохранена глобальная сегментация с грыжами: {global_seg_path}")
        except Exception as e:
            logger.error(f"Ошибка при сохранении глобальной сегментации: {e}")

        save_grading_results(result, processed_disks, output_dir, logger=logger)
        logger.info("Анализ завершен успешно")
        
        # Создаем BMP изображения с подписанными позвонками
        # По умолчанию режем по оси 0 (X) для сагиттальной проекции, вариация 0 (контуры)
        create_sagittal_bmp_images(nifti_img, nifti_seg, output_dir, logger, slice_axis=0, variation=0)
        
        # В конце main, после всех обработок:
        elapsed = time.time() - start_time
        logger.info(f"Время обработки: {elapsed:.1f} секунд ({elapsed/60:.2f} минут)")
        
    except Exception as e:
        if logger:
            logger.error(f"Критическая ошибка в main: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        else:
            print(f"Критическая ошибка: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()