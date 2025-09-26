import numpy as np
import cv2
import os
from typing import Optional, Dict, List, Tuple
import logging
from nibabel import Nifti1Image

logger = logging.getLogger("dicom-pipeline")


def create_enhanced_segmentation_mask(
        original_mask: np.ndarray,
        pathology_measurements: Dict[int, Dict],
        herniation_label_offset: int = 1000,
        spondylolisthesis_label_offset: int = 2000
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Создает расширенную маску сегментации с отдельными метками для патологий.

    Args:
        original_mask: Исходная маска сегментации
        pathology_measurements: Результаты измерения патологий
        herniation_label_offset: Смещение меток для грыж (1000+ для грыж)
        spondylolisthesis_label_offset: Смещение меток для листезов (2000+ для листезов)

    Returns:
        enhanced_mask: Расширенная маска с патологиями
        label_map: Словарь соответствия меток и названий
    """
    enhanced_mask = original_mask.copy()
    label_map = {}

    # Добавляем исходные метки в карту
    unique_labels = np.unique(original_mask)
    for label in unique_labels:
        if label == 0:
            label_map[0] = "Background"
        elif label < 100:  # Предполагаем, что диски имеют метки < 100
            label_map[int(label)] = f"Disk_{label}"
        else:
            label_map[int(label)] = f"Structure_{label}"

    try:
        for disk_label, measurements in pathology_measurements.items():
            if 'error' in measurements:
                continue

            level_name = measurements.get('level_name', f'Disk_{disk_label}')

            # Добавляем грыжи как отдельные области
            if measurements.get('herniation') and measurements['herniation'].get('detected'):
                herniation_data = measurements['herniation']

                # Создаем маску грыжи из исходного измерения
                herniation_mask = create_herniation_visualization_mask(
                    original_mask, disk_label, herniation_data
                )

                if np.any(herniation_mask):
                    herniation_label = herniation_label_offset + disk_label
                    enhanced_mask[herniation_mask] = herniation_label

                    severity = herniation_data.get('severity', 'unknown')
                    herniation_type = herniation_data.get('herniation_type', 'unknown')
                    label_map[herniation_label] = f"Herniation_{level_name}_{herniation_type}_{severity}"

            # Можно добавить визуализацию листеза как смещения позвонков
            # (это сложнее визуализировать как отдельную область)

        return enhanced_mask, label_map

    except Exception as e:
        logger.error(f"Error creating enhanced segmentation mask: {e}")
        return original_mask, label_map


def create_herniation_visualization_mask(
        original_mask: np.ndarray,
        disk_label: int,
        herniation_data: Dict
) -> np.ndarray:
    """
    Создает маску для визуализации грыжи на основе измерений.
    Эта функция воссоздает области грыжи для визуализации.
    """
    try:
        from .pathology_measurements import PathologyMeasurements

        # Получаем маску диска
        disk_mask = (original_mask == disk_label)
        if not np.any(disk_mask):
            return np.zeros_like(original_mask, dtype=bool)

        # Воссоздаем области грыжи используя тот же алгоритм
        # Это приблизительная реконструкция на основе существующего алгоритма
        measurer = PathologyMeasurements()

        # Простая аппроксимация: расширяем диск в сторону канала
        canal_mask = (original_mask == getattr(settings, 'CANAL_LABEL', 1))

        # Применяем морфологические операции для выделения выступающих частей
        zdim, ydim, xdim = disk_mask.shape
        herniation_mask = np.zeros_like(disk_mask, dtype=bool)

        # Обрабатываем центральные срезы
        coords = np.argwhere(disk_mask)
        if coords.size == 0:
            return herniation_mask

        zc = int(np.round(coords[:, 0].mean()))
        z_from = max(0, zc - 2)
        z_to = min(zdim, zc + 3)

        for z in range(z_from, z_to):
            disk_slice = disk_mask[z].astype(np.uint8)
            if disk_slice.sum() == 0:
                continue

            # Создаем "нормальную" форму диска
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            normal_disk = cv2.morphologyEx(disk_slice, cv2.MORPH_OPEN, kernel)

            # Выступающие части
            protruding = disk_slice & (~normal_disk.astype(bool))

            # Фильтруем по направлению к каналу
            if np.any(canal_mask[z]):
                canal_slice = canal_mask[z]
                # Простая фильтрация: оставляем только части, ближайшие к каналу
                if np.any(protruding):
                    herniation_mask[z] = protruding

        return herniation_mask

    except Exception as e:
        logger.error(f"Error creating herniation visualization mask: {e}")
        return np.zeros_like(original_mask, dtype=bool)


def save_enhanced_segmentation_overlay(
        image_nifti: Nifti1Image,
        original_seg_nifti: Nifti1Image,
        pathology_measurements: Dict[int, Dict],
        output_dir: str,
        slice_axis: int = 0,
        thickness: int = 2
):
    """
    Сохраняет оверлей с расширенной сегментацией, включая патологии.

    Args:
        image_nifti: NIfTI изображение
        original_seg_nifti: Исходная сегментация
        pathology_measurements: Измерения патологий
        output_dir: Директория для сохранения
        slice_axis: Ось срезов
        thickness: Толщина контуров
    """
    try:
        # Создаем расширенную маску
        original_mask = np.asanyarray(original_seg_nifti.dataobj).astype(np.uint8)
        enhanced_mask, label_map = create_enhanced_segmentation_mask(
            original_mask, pathology_measurements
        )

        # Создаем цветовую карту с особыми цветами для патологий
        colormap = create_pathology_colormap(label_map)

        # Создаем временный NIfTI объект для расширенной маски
        from nibabel import Nifti1Image
        enhanced_seg_nifti = Nifti1Image(enhanced_mask, original_seg_nifti.affine, original_seg_nifti.header)

        # Сохраняем оверлей
        save_segmentation_overlay_multiclass(
            image_nifti=image_nifti,
            seg_nifti=enhanced_seg_nifti,
            output_dir=output_dir,
            slice_axis=slice_axis,
            thickness=thickness,
            colormap=colormap
        )

        # Сохраняем легенду
        save_colormap_legend(colormap, label_map, output_dir)

        logger.info(f"Saved enhanced segmentation overlay with pathologies to {output_dir}")

    except Exception as e:
        logger.error(f"Error saving enhanced segmentation overlay: {e}")


def create_pathology_colormap(label_map: Dict[int, str]) -> Dict[int, Tuple[int, int, int]]:
    """
    Создает цветовую карту с особыми цветами для патологий.
    """
    colormap = {}

    for label, name in label_map.items():
        if label == 0:  # Background
            continue
        elif "Herniation" in name:
            # Красные оттенки для грыж
            if "mild" in name:
                colormap[label] = (0, 100, 255)  # Светло-красный
            elif "moderate" in name:
                colormap[label] = (0, 50, 255)  # Красный
            else:  # severe
                colormap[label] = (0, 0, 255)  # Ярко-красный
        elif "Spondylolisthesis" in name:
            # Синие оттенки для листезов
            colormap[label] = (255, 100, 0)  # Синий
        elif "Disk" in name:
            # Зеленые оттенки для дисков
            colormap[label] = (0, 255, 0)  # Зеленый
        elif "Canal" in name:
            # Желтый для канала
            colormap[label] = (0, 255, 255)  # Желтый
        else:
            # Случайные цвета для других структур
            np.random.seed(label)
            colormap[label] = tuple(np.random.randint(50, 255, size=3).tolist())

    return colormap


def save_colormap_legend(colormap: Dict[int, Tuple[int, int, int]],
                         label_map: Dict[int, str],
                         output_dir: str):
    """
    Сохраняет легенду цветовой карты.
    """
    try:
        legend_path = os.path.join(output_dir, "legend.txt")
        with open(legend_path, 'w') as f:
            f.write("Label\tColor(BGR)\tDescription\n")
            for label, name in label_map.items():
                if label in colormap:
                    color = colormap[label]
                    f.write(f"{label}\t{color}\t{name}\n")

        logger.info(f"Saved colormap legend to {legend_path}")

    except Exception as e:
        logger.error(f"Error saving colormap legend: {e}")


# Интеграция в основной pipeline
def save_segmentation_with_pathologies(
        image_nifti: Nifti1Image,
        seg_nifti: Nifti1Image,
        pathology_measurements: Dict[int, Dict],
        output_dir: str = None,
        debug: bool = False
) -> Tuple[Nifti1Image, Nifti1Image]:
    """
    Основная функция для создания сегментации с выделенными патологиями.
    Возвращает исходные данные и расширенную сегментацию.

    Args:
        image_nifti: NIfTI изображение
        seg_nifti: Исходная сегментация
        pathology_measurements: Измерения патологий
        output_dir: Директория для сохранения (только если debug=True)
        debug: Сохранять ли файлы визуализации на диск

    Returns:
        Tuple[image_nifti, enhanced_seg_nifti]: Исходное изображение и расширенная сегментация
    """
    try:
        # Создаем расширенную маску сегментации
        original_mask = np.asanyarray(seg_nifti.dataobj).astype(np.uint8)
        enhanced_mask, label_map = create_enhanced_segmentation_mask(
            original_mask, pathology_measurements
        )

        # Создаем новый NIfTI объект для расширенной маски
        from nibabel import Nifti1Image
        enhanced_seg_nifti = Nifti1Image(
            enhanced_mask,
            seg_nifti.affine,
            seg_nifti.header
        )

        # Сохраняем файлы только в debug режиме
        if debug and output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

            # Сохраняем стандартный оверлей
            standard_output = os.path.join(output_dir, "standard_overlay")
            save_segmentation_overlay_multiclass(
                image_nifti, seg_nifti, standard_output
            )

            # Сохраняем расширенный оверлей с патологиями
            enhanced_output = os.path.join(output_dir, "enhanced_overlay")
            save_enhanced_segmentation_overlay(
                image_nifti, seg_nifti, pathology_measurements, enhanced_output
            )

            # Сохраняем расширенную маску как NIfTI
            enhanced_nifti_path = os.path.join(output_dir, "enhanced_segmentation.nii.gz")
            enhanced_seg_nifti.to_filename(enhanced_nifti_path)

            logger.info(f"Debug: Saved segmentation files to {output_dir}")

        logger.info("Created enhanced segmentation with pathology labels")
        return image_nifti, enhanced_seg_nifti

    except Exception as e:
        logger.error(f"Error creating enhanced segmentation: {e}")
        # В случае ошибки возвращаем исходные данные
        return image_nifti, seg_nifti


def create_pathology_enhanced_nifti(
        seg_nifti: Nifti1Image,
        pathology_measurements: Dict[int, Dict]
) -> Tuple[Nifti1Image, Dict[int, str]]:
    """
    Создает NIfTI с расширенной сегментацией без сохранения файлов.

    Args:
        seg_nifti: Исходная сегментация
        pathology_measurements: Измерения патологий

    Returns:
        Tuple[enhanced_nifti, label_map]: Расширенная сегментация и карта меток
    """
    try:
        original_mask = np.asanyarray(seg_nifti.dataobj).astype(np.uint8)
        enhanced_mask, label_map = create_enhanced_segmentation_mask(
            original_mask, pathology_measurements
        )

        from nibabel import Nifti1Image
        enhanced_nifti = Nifti1Image(
            enhanced_mask,
            seg_nifti.affine,
            seg_nifti.header
        )

        return enhanced_nifti, label_map

    except Exception as e:
        logger.error(f"Error creating pathology enhanced NIfTI: {e}")
        return seg_nifti, {}


def save_debug_visualization(
        image_nifti: Nifti1Image,
        seg_nifti: Nifti1Image,
        enhanced_seg_nifti: Nifti1Image,
        pathology_measurements: Dict[int, Dict],
        output_dir: str
):
    """
    Сохраняет debug визуализацию с подробной информацией.

    Args:
        image_nifti: Исходное изображение
        seg_nifti: Исходная сегментация
        enhanced_seg_nifti: Расширенная сегментация
        pathology_measurements: Измерения патологий
        output_dir: Директория для сохранения
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Создаем цветовую карту
        original_mask = np.asanyarray(seg_nifti.dataobj).astype(np.uint8)
        enhanced_mask = np.asanyarray(enhanced_seg_nifti.dataobj).astype(np.uint8)

        _, label_map = create_enhanced_segmentation_mask(
            original_mask, pathology_measurements
        )
        colormap = create_pathology_colormap(label_map)

        # Сохраняем оверлеи
        standard_dir = os.path.join(output_dir, "standard_overlay")
        enhanced_dir = os.path.join(output_dir, "enhanced_overlay")

        save_segmentation_overlay_multiclass(
            image_nifti, seg_nifti, standard_dir, colormap=None
        )

        save_segmentation_overlay_multiclass(
            image_nifti, enhanced_seg_nifti, enhanced_dir, colormap=colormap
        )

        # Сохраняем NIfTI файлы
        seg_path = os.path.join(output_dir, "original_segmentation.nii.gz")
        enhanced_path = os.path.join(output_dir, "enhanced_segmentation.nii.gz")

        seg_nifti.to_filename(seg_path)
        enhanced_seg_nifti.to_filename(enhanced_path)

        # Сохраняем метаданные
        save_pathology_metadata(pathology_measurements, label_map, output_dir)
        save_colormap_legend(colormap, label_map, output_dir)

        logger.info(f"Saved debug visualization to {output_dir}")

    except Exception as e:
        logger.error(f"Error saving debug visualization: {e}")


def save_pathology_metadata(
        pathology_measurements: Dict[int, Dict],
        label_map: Dict[int, str],
        output_dir: str
):
    """
    Сохраняет метаданные о патологиях.
    """
    try:
        import json

        metadata = {
            'pathology_measurements': pathology_measurements,
            'label_map': {str(k): v for k, v in label_map.items()},
            'enhanced_labels': {}
        }

        # Добавляем информацию о расширенных метках
        for label, name in label_map.items():
            if label >= 1000:
                metadata['enhanced_labels'][str(label)] = {
                    'name': name,
                    'type': 'herniation' if label < 2000 else 'spondylolisthesis',
                    'original_disk': label % 1000 if label < 2000 else label % 1000
                }

        metadata_path = os.path.join(output_dir, "pathology_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved pathology metadata to {metadata_path}")

    except Exception as e:
        logger.error(f"Error saving pathology metadata: {e}")


# Упрощенная функция для использования в pipeline
def process_segmentation_with_pathologies(
        image_nifti: Nifti1Image,
        seg_nifti: Nifti1Image,
        pathology_measurements: Dict[int, Dict],
        debug: bool = False,
        debug_output_dir: str = None
) -> Tuple[Nifti1Image, Nifti1Image]:
    """
    Упрощенная функция для обработки сегментации с патологиями.
    Всегда возвращает исходные данные и расширенную сегментацию.

    Args:
        image_nifti: Исходное изображение
        seg_nifti: Исходная сегментация
        pathology_measurements: Измерения патологий
        debug: Включить debug режим
        debug_output_dir: Директория для debug файлов

    Returns:
        Tuple[image_nifti, enhanced_seg_nifti]: Исходное изображение и расширенная сегментация
    """
    try:
        # Создаем расширенную сегментацию
        enhanced_seg_nifti, label_map = create_pathology_enhanced_nifti(
            seg_nifti, pathology_measurements
        )

        # Сохраняем debug информацию если нужно
        if debug and debug_output_dir:
            save_debug_visualization(
                image_nifti, seg_nifti, enhanced_seg_nifti,
                pathology_measurements, debug_output_dir
            )

        return image_nifti, enhanced_seg_nifti

    except Exception as e:
        logger.error(f"Error processing segmentation with pathologies: {e}")
        return image_nifti, seg_nifti