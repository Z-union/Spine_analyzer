"""
Модуль для измерения патологий позвоночника после grading анализа.
Измеряет размеры и объемы грыж, размеры листезов.
Улучшенная версия с более точным выделением грыж/выбуханий.
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Union
from scipy import ndimage
from skimage import measure, morphology
from .config import settings

# Используем единый логгер из main
logger = logging.getLogger("dicom-pipeline")


class PathologyMeasurements:
    """
    Класс для измерения патологий позвоночника.
    """

    def __init__(self, voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Инициализация измерителя патологий.

        Args:
            voxel_spacing: Размер вокселя в мм (depth, height, width)
        """
        self.voxel_spacing = voxel_spacing
        logger.info(f"Инициализирован измеритель патологий с voxel_spacing={voxel_spacing}")

    def measure_disk_pathologies(self,
                                mri_data: np.ndarray,
                                mask_data: np.ndarray,
                                disk_label: int,
                                grading_results: Dict,
                                level_name: str) -> Dict:
        """
        Измеряет патологии для конкретного диска.

        Args:
            mri_data: Данные МРТ изображения
            mask_data: Данные сегментации
            disk_label: Метка диска
            grading_results: Результаты grading анализа
            level_name: Название уровня диска

        Returns:
            Словарь с измерениями патологий
        """
        measurements = {
            'disk_label': disk_label,
            'level_name': level_name,
            'herniation': None,
            'spondylolisthesis': None,
            'disk_measurements': None
        }

        try:
            # Получаем маску диска
            disk_mask = (mask_data == disk_label)
            if not np.any(disk_mask):
                return {'error': 'Disk not found in mask'}

            # Основные измерения диска
            measurements['disk_measurements'] = self._measure_disk_basic(disk_mask)

            # Измеряем грыжу если обнаружена
            if grading_results.get('Herniation', 0) > 0:
                logger.info(f"Измеряем грыжу для диска {level_name}")
                measurements['herniation'] = self._measure_herniation(
                    mri_data, mask_data, disk_label, disk_mask
                )

            # Измеряем листез если обнаружен
            if grading_results.get('Spondylolisthesis', 0) > 0:
                logger.info(f"Измеряем листез для диска {level_name}")
                measurements['spondylolisthesis'] = self._measure_spondylolisthesis(
                    mri_data, mask_data, disk_label, disk_mask
                )

            return measurements

        except Exception as e:
            logger.error(f"Ошибка при измерении патологий диска {disk_label}: {e}")
            return {'error': str(e)}

    def _measure_disk_basic(self, disk_mask: np.ndarray) -> Dict:
        """
        Основные измерения диска.

        Args:
            disk_mask: Маска диска

        Returns:
            Словарь с основными измерениями
        """
        try:
            # Объем диска
            volume_voxels = np.sum(disk_mask)
            volume_mm3 = volume_voxels * np.prod(self.voxel_spacing)

            # Центроид диска
            coords = np.argwhere(disk_mask)
            centroid = coords.mean(axis=0)

            # Размеры bounding box
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            bbox_size = (max_coords - min_coords) * np.array(self.voxel_spacing)

            # Высота диска (в сагиттальной проекции обычно по оси Y)
            disk_height_mm = bbox_size[2]  # height dimension

            # Диаметр диска (максимальный размер в аксиальной плоскости)
            disk_diameter_mm = max(bbox_size[0], bbox_size[1])  # depth или width

            return {
                'volume_voxels': int(volume_voxels),
                'volume_mm3': float(volume_mm3),
                'centroid': centroid.tolist(),
                'bbox_size_mm': bbox_size.tolist(),
                'height_mm': float(disk_height_mm),
                'diameter_mm': float(disk_diameter_mm),
                'aspect_ratio': float(disk_diameter_mm / disk_height_mm) if disk_height_mm > 0 else 0
            }

        except Exception as e:
            logger.error(f"Ошибка при основных измерениях диска: {e}")
            return {'error': str(e)}

    def _measure_herniation(self,
                           mri_data: np.ndarray,
                           mask_data: np.ndarray,
                           disk_label: int,
                           disk_mask: np.ndarray) -> Dict:
        """
        Измеряет параметры грыжи/выбухания как части диска, выходящие за пределы позвонков
        в направлении спинномозгового канала.
        """
        try:
            canal_mask = (mask_data == settings.CANAL_LABEL)

            # Улучшенное выделение грыжи/выбухания
            herniation_mask, disk_boundary = self._detect_herniation_advanced(
                mri_data, mask_data, disk_mask, canal_mask
            )

            if not np.any(herniation_mask):
                return {
                    'detected': False,
                    'volume_mm3': 0.0,
                    'max_protrusion_mm': 0.0,
                    'regions_count': 0
                }

            herniation_volume_voxels = int(np.sum(herniation_mask))
            herniation_volume_mm3 = float(herniation_volume_voxels * np.prod(self.voxel_spacing))

            max_protrusion_mm = float(self._calculate_max_protrusion_advanced(
                herniation_mask, disk_boundary, canal_mask
            ))

            # Анализ компонент грыжи
            labeled_regions, num_regions = ndimage.label(herniation_mask)
            regions_info = []
            for region_id in range(1, num_regions + 1):
                region_mask = (labeled_regions == region_id)
                region_info = self._analyze_herniation_region(
                    region_mask, region_id, canal_mask
                )
                regions_info.append(region_info)

            # Классификация типа грыжи
            herniation_type = self._classify_herniation_type(
                herniation_mask, canal_mask, max_protrusion_mm
            )

            return {
                'detected': True,
                'volume_mm3': herniation_volume_mm3,
                'volume_voxels': herniation_volume_voxels,
                'max_protrusion_mm': max_protrusion_mm,
                'regions_count': int(num_regions),
                'regions_info': regions_info,
                'herniation_type': herniation_type,
                'severity': self._classify_herniation_severity(herniation_volume_mm3, max_protrusion_mm),
                'canal_involvement': self._assess_canal_involvement(herniation_mask, canal_mask)
            }
        except Exception as e:
            logger.error(f"Ошибка при измерении грыжи: {e}")
            return {'error': str(e)}

    def _detect_herniation_advanced(self,
                                   mri_data: np.ndarray,
                                   mask_data: np.ndarray,
                                   disk_mask: np.ndarray,
                                   canal_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Улучшенное выделение грыжи/выбухания с учетом анатомических границ.
        Определяет части диска, выходящие за пределы позвонков в сторону канала.
        """
        try:
            zdim, ydim, xdim = disk_mask.shape
            herniation_mask = np.zeros_like(disk_mask, dtype=bool)

            # Находим соседние позвонки для определения границ
            vertebra_boundaries = self._find_vertebral_boundaries(mask_data, disk_mask)

            # Определяем направление к каналу
            canal_direction = self._determine_canal_direction(disk_mask, canal_mask)

            # Обрабатываем срезы в центральной области диска
            coords = np.argwhere(disk_mask)
            if coords.size == 0:
                return herniation_mask, np.zeros_like(disk_mask, dtype=bool)

            zc = int(np.round(coords[:, 0].mean()))
            z_from = max(0, zc - 3)
            z_to = min(zdim, zc + 4)

            disk_boundary = np.zeros_like(disk_mask, dtype=bool)

            for z in range(z_from, z_to):
                disk_slice = disk_mask[z]
                if not disk_slice.any():
                    continue

                # Определяем нормальную границу диска на этом срезе
                normal_boundary = self._define_normal_disk_boundary(
                    disk_slice, vertebra_boundaries, canal_direction, z
                )

                # Находим выступающие части
                protruding_parts = disk_slice & (~normal_boundary)

                # Фильтруем по направлению к каналу
                canal_directed_protrusion = self._filter_by_canal_direction(
                    protruding_parts, disk_slice, canal_mask[z], canal_direction
                )

                herniation_mask[z] = canal_directed_protrusion
                disk_boundary[z] = self._find_disk_boundary(disk_slice)

            # Постобработка: удаляем мелкие артефакты
            herniation_mask = morphology.remove_small_objects(herniation_mask, min_size=20)

            return herniation_mask, disk_boundary

        except Exception as e:
            logger.error(f"Ошибка при улучшенном выделении грыжи: {e}")
            return np.zeros_like(disk_mask, dtype=bool), np.zeros_like(disk_mask, dtype=bool)

    def _find_vertebral_boundaries(self, mask_data: np.ndarray, disk_mask: np.ndarray) -> Dict:
        """
        Находит границы соседних позвонков для определения нормальных пределов диска.
        """
        try:
            # Получаем координаты диска
            disk_coords = np.argwhere(disk_mask)
            disk_centroid = disk_coords.mean(axis=0)

            # Ищем позвонки в окрестности
            unique_labels = np.unique(mask_data)
            vertebra_labels = [label for label in unique_labels if 10 <= label <= 50]

            boundaries = {
                'upper': None,
                'lower': None,
                'anterior': None,
                'posterior': None
            }

            for vertebra_label in vertebra_labels:
                vertebra_mask = (mask_data == vertebra_label)
                if not np.any(vertebra_mask):
                    continue

                vertebra_coords = np.argwhere(vertebra_mask)
                vertebra_centroid = vertebra_coords.mean(axis=0)

                # Определяем положение относительно диска
                relative_pos = vertebra_centroid - disk_centroid

                # Классифицируем позвонок по положению
                if abs(relative_pos[1]) > abs(relative_pos[0]) and abs(relative_pos[1]) > abs(relative_pos[2]):
                    if relative_pos[1] < -10:  # Выше
                        boundaries['upper'] = vertebra_mask
                    elif relative_pos[1] > 10:  # Ниже
                        boundaries['lower'] = vertebra_mask

                if abs(relative_pos[2]) > abs(relative_pos[0]) and abs(relative_pos[2]) > abs(relative_pos[1]):
                    if relative_pos[2] < -5:  # Спереди
                        boundaries['anterior'] = vertebra_mask
                    elif relative_pos[2] > 5:  # Сзади
                        boundaries['posterior'] = vertebra_mask

            return boundaries

        except Exception as e:
            logger.error(f"Ошибка при поиске границ позвонков: {e}")
            return {}

    def _determine_canal_direction(self, disk_mask: np.ndarray, canal_mask: np.ndarray) -> np.ndarray:
        """
        Определяет направление от диска к спинномозговому каналу.
        """
        try:
            if not np.any(canal_mask):
                # Если канал не найден, предполагаем заднее направление
                return np.array([0, 0, 1])  # Posterior direction

            disk_coords = np.argwhere(disk_mask)
            canal_coords = np.argwhere(canal_mask)

            disk_centroid = disk_coords.mean(axis=0)
            canal_centroid = canal_coords.mean(axis=0)

            # Вектор направления от диска к каналу
            direction = canal_centroid - disk_centroid
            norm = np.linalg.norm(direction)

            if norm > 0:
                direction = direction / norm
            else:
                direction = np.array([0, 0, 1])  # Default posterior

            return direction

        except Exception as e:
            logger.error(f"Ошибка при определении направления канала: {e}")
            return np.array([0, 0, 1])

    def _define_normal_disk_boundary(self,
                                    disk_slice: np.ndarray,
                                    vertebra_boundaries: Dict,
                                    canal_direction: np.ndarray,
                                    z_level: int) -> np.ndarray:
        """
        Определяет нормальную границу диска с учетом соседних позвонков.
        """
        try:
            # Используем морфологическое закрытие для создания выпуклой оболочки
            kernel_size = max(3, int(3.0 / min(self.voxel_spacing[1], self.voxel_spacing[2])))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

            # Создаем базовую выпуклую форму диска
            convex_disk = cv2.morphologyEx(disk_slice.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

            # Корректируем границы с учетом позвонков
            if vertebra_boundaries.get('upper') is not None:
                upper_slice = vertebra_boundaries['upper'][z_level]
                convex_disk = convex_disk & (~upper_slice)

            if vertebra_boundaries.get('lower') is not None:
                lower_slice = vertebra_boundaries['lower'][z_level]
                convex_disk = convex_disk & (~lower_slice)

            return convex_disk.astype(bool)

        except Exception as e:
            logger.error(f"Ошибка при определении нормальной границы диска: {e}")
            return disk_slice

    def _filter_by_canal_direction(self,
                                  protruding_parts: np.ndarray,
                                  disk_slice: np.ndarray,
                                  canal_slice: np.ndarray,
                                  canal_direction: np.ndarray) -> np.ndarray:
        """
        Фильтрует выступающие части по направлению к каналу.
        """
        try:
            if not np.any(protruding_parts):
                return protruding_parts

            # Получаем центроид диска на срезе
            disk_coords = np.argwhere(disk_slice)
            if disk_coords.size == 0:
                return protruding_parts

            disk_center = disk_coords.mean(axis=0)

            # Создаем маску для области, направленной к каналу
            ydim, xdim = disk_slice.shape
            yy, xx = np.mgrid[0:ydim, 0:xdim]

            if np.any(canal_slice):
                # Используем фактическое положение канала
                canal_coords = np.argwhere(canal_slice)
                canal_center = canal_coords.mean(axis=0)
                direction_to_canal = canal_center - disk_center
            else:
                # Используем предполагаемое направление
                direction_to_canal = canal_direction[1:]  # Только Y и X компоненты

            # Нормализуем направление
            norm = np.linalg.norm(direction_to_canal)
            if norm > 0:
                direction_to_canal = direction_to_canal / norm

                # Создаем полуплоскость в направлении канала
                canal_direction_mask = ((yy - disk_center[0]) * direction_to_canal[0] +
                                       (xx - disk_center[1]) * direction_to_canal[1]) > 0

                return protruding_parts & canal_direction_mask

            return protruding_parts

        except Exception as e:
            logger.error(f"Ошибка при фильтрации по направлению канала: {e}")
            return protruding_parts

    def _calculate_max_protrusion_advanced(self,
                                          herniation_mask: np.ndarray,
                                          disk_boundary: np.ndarray,
                                          canal_mask: np.ndarray) -> float:
        """
        Улучшенное вычисление максимального выпячивания с учетом близости к каналу.
        """
        try:
            if not np.any(herniation_mask):
                return 0.0

            boundary_coords = np.argwhere(disk_boundary)
            herniation_coords = np.argwhere(herniation_mask)

            if boundary_coords.size == 0 or herniation_coords.size == 0:
                return 0.0

            # Преобразуем в миллиметры
            zs, ys, xs = self.voxel_spacing
            boundary_mm = boundary_coords.astype(np.float32)
            boundary_mm[:, 0] *= zs
            boundary_mm[:, 1] *= ys
            boundary_mm[:, 2] *= xs

            max_protrusion = 0.0

            for p in herniation_coords:
                pmm = p.astype(np.float32)
                pmm[0] *= zs; pmm[1] *= ys; pmm[2] *= xs

                # Расстояние до ближайшей точки границы диска
                distances = np.sqrt(np.sum((boundary_mm - pmm) ** 2, axis=1))
                min_distance = float(np.min(distances))

                # Дополнительный вес, если точка близко к каналу
                if np.any(canal_mask):
                    canal_coords = np.argwhere(canal_mask)
                    if canal_coords.size > 0:
                        canal_mm = canal_coords.astype(np.float32)
                        canal_mm[:, 0] *= zs
                        canal_mm[:, 1] *= ys
                        canal_mm[:, 2] *= xs

                        canal_distances = np.sqrt(np.sum((canal_mm - pmm) ** 2, axis=1))
                        min_canal_distance = np.min(canal_distances)

                        # Увеличиваем значимость точек близко к каналу
                        if min_canal_distance < 10:  # Менее 10 мм от канала
                            weight = 1.2
                        else:
                            weight = 1.0

                        weighted_distance = min_distance * weight
                        max_protrusion = max(max_protrusion, weighted_distance)
                    else:
                        max_protrusion = max(max_protrusion, min_distance)
                else:
                    max_protrusion = max(max_protrusion, min_distance)

            return max_protrusion

        except Exception as e:
            logger.error(f"Ошибка при улучшенном вычислении выпячивания: {e}")
            return 0.0

    def _analyze_herniation_region(self,
                                  region_mask: np.ndarray,
                                  region_id: int,
                                  canal_mask: np.ndarray) -> Dict:
        """
        Анализирует отдельную область грыжи.
        """
        try:
            region_volume = float(np.sum(region_mask) * np.prod(self.voxel_spacing))
            region_coords = np.argwhere(region_mask)
            region_centroid = region_coords.mean(axis=0)

            # Расстояние до канала
            canal_distance = float('inf')
            if np.any(canal_mask):
                canal_coords = np.argwhere(canal_mask)
                if canal_coords.size > 0:
                    # Преобразуем в мм
                    region_centroid_mm = region_centroid * np.array(self.voxel_spacing)
                    canal_coords_mm = canal_coords * np.array(self.voxel_spacing)

                    distances = np.sqrt(np.sum((canal_coords_mm - region_centroid_mm) ** 2, axis=1))
                    canal_distance = float(np.min(distances))

            return {
                'region_id': int(region_id),
                'volume_mm3': region_volume,
                'centroid': region_centroid.tolist(),
                'voxel_count': int(np.sum(region_mask)),
                'canal_distance_mm': canal_distance,
                'severity_score': self._calculate_region_severity_score(region_volume, canal_distance)
            }

        except Exception as e:
            logger.error(f"Ошибка при анализе области грыжи: {e}")
            return {'error': str(e)}

    def _classify_herniation_type(self,
                                 herniation_mask: np.ndarray,
                                 canal_mask: np.ndarray,
                                 max_protrusion_mm: float) -> str:
        """
        Классифицирует тип грыжи/выбухания.
        """
        try:
            if max_protrusion_mm < 3:
                return 'bulging'  # Выбухание
            elif max_protrusion_mm < 6:
                return 'protrusion'  # Протрузия
            elif max_protrusion_mm < 12:
                return 'extrusion'  # Экструзия
            else:
                return 'sequestration'  # Секвестрация

        except Exception as e:
            logger.error(f"Ошибка при классификации типа грыжи: {e}")
            return 'unknown'

    def _assess_canal_involvement(self,
                                 herniation_mask: np.ndarray,
                                 canal_mask: np.ndarray) -> Dict:
        """
        Оценивает вовлечение спинномозгового канала.
        """
        try:
            if not np.any(canal_mask) or not np.any(herniation_mask):
                return {
                    'involved': False,
                    'compression_percentage': 0.0,
                    'min_distance_mm': float('inf')
                }

            # Проверяем пересечение
            intersection = herniation_mask & canal_mask
            involvement = np.any(intersection)

            # Процент сжатия канала
            if involvement:
                compression_percentage = (np.sum(intersection) / np.sum(canal_mask)) * 100
            else:
                compression_percentage = 0.0

            # Минимальное расстояние до канала
            if not involvement:
                herniation_coords = np.argwhere(herniation_mask)
                canal_coords = np.argwhere(canal_mask)

                # Преобразуем в мм
                herniation_mm = herniation_coords * np.array(self.voxel_spacing)
                canal_mm = canal_coords * np.array(self.voxel_spacing)

                min_distance = float('inf')
                for h_point in herniation_mm:
                    distances = np.sqrt(np.sum((canal_mm - h_point) ** 2, axis=1))
                    min_distance = min(min_distance, np.min(distances))
            else:
                min_distance = 0.0

            return {
                'involved': bool(involvement),
                'compression_percentage': float(compression_percentage),
                'min_distance_mm': float(min_distance)
            }

        except Exception as e:
            logger.error(f"Ошибка при оценке вовлечения канала: {e}")
            return {'error': str(e)}

    def _calculate_region_severity_score(self, volume_mm3: float, canal_distance_mm: float) -> float:
        """
        Вычисляет оценку тяжести для области грыжи.
        """
        try:
            # Базовая оценка по объему
            volume_score = min(volume_mm3 / 100.0, 10.0)  # Максимум 10 баллов

            # Оценка по близости к каналу
            if canal_distance_mm < 2:
                distance_score = 10.0
            elif canal_distance_mm < 5:
                distance_score = 7.0
            elif canal_distance_mm < 10:
                distance_score = 4.0
            else:
                distance_score = 1.0

            # Общая оценка
            total_score = (volume_score + distance_score) / 2.0
            return float(min(total_score, 10.0))

        except Exception as e:
            logger.error(f"Ошибка при расчете оценки тяжести: {e}")
            return 0.0

    def _measure_spondylolisthesis(self,
                                  mri_data: np.ndarray,
                                  mask_data: np.ndarray,
                                  disk_label: int,
                                  disk_mask: np.ndarray) -> Dict:
        """
        Измеряет параметры спондилолистеза.

        Args:
            mri_data: Данные МРТ изображения
            mask_data: Данные сегментации
            disk_label: Метка диска
            disk_mask: Маска диска

        Returns:
            Словарь с измерениями листеза
        """
        try:
            # Находим соседние позвонки
            upper_vertebra, lower_vertebra = self._find_adjacent_vertebrae(
                mask_data, disk_label, disk_mask
            )

            if upper_vertebra is None or lower_vertebra is None:
                return {
                    'detected': False,
                    'displacement_mm': 0,
                    'displacement_percentage': 0,
                    'grade': 0
                }

            # Вычисляем смещение
            displacement_mm, displacement_percentage = self._calculate_vertebral_displacement(
                upper_vertebra, lower_vertebra
            )

            # Классифицируем степень листеза по Meyerding
            grade = self._classify_spondylolisthesis_grade(displacement_percentage)

            # Направление смещения
            displacement_direction = self._determine_displacement_direction(
                upper_vertebra, lower_vertebra
            )

            return {
                'detected': True,
                'displacement_mm': float(displacement_mm),
                'displacement_percentage': float(displacement_percentage),
                'grade': int(grade),
                'direction': displacement_direction,
                'severity': self._classify_spondylolisthesis_severity(grade),
                'upper_vertebra_centroid': upper_vertebra['centroid'],
                'lower_vertebra_centroid': lower_vertebra['centroid']
            }

        except Exception as e:
            logger.error(f"Ошибка при измерении листеза: {e}")
            return {'error': str(e)}

    def _find_disk_boundary(self, disk_mask: np.ndarray) -> np.ndarray:
        """
        Находит границу диска.

        Args:
            disk_mask: Маска диска

        Returns:
            Маска границы диска
        """
        # Используем морфологические операции для нахождения границы
        eroded = morphology.binary_erosion(disk_mask)
        boundary = disk_mask & ~eroded
        return boundary

    def _find_adjacent_vertebrae(self,
                                mask_data: np.ndarray,
                                disk_label: int,
                                disk_mask: np.ndarray) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Находит соседние позвонки для измерения листеза.
        Ищет позвонки, непосредственно граничащие с диском.

        Args:
            mask_data: Данные сегментации
            disk_label: Метка диска
            disk_mask: Маска диска

        Returns:
            Кортеж (верхний_позвонок, нижний_позвонок)
        """
        try:
            # Получаем центроид и границы диска
            disk_coords = np.argwhere(disk_mask)
            disk_centroid = disk_coords.mean(axis=0)

            # Определяем границы диска по вертикальной оси (обычно Y)
            disk_y_min, disk_y_max = disk_coords[:, 1].min(), disk_coords[:, 1].max()

            # Ищем позвонки в окрестности диска
            unique_labels = np.unique(mask_data)
            # Более широкий диапазон для поиска позвонков
            vertebra_labels = [label for label in unique_labels if 1 <= label <= 100 and label != disk_label]

            upper_vertebra = None
            lower_vertebra = None
            min_upper_gap = float('inf')
            min_lower_gap = float('inf')

            for vertebra_label in vertebra_labels:
                vertebra_mask = (mask_data == vertebra_label)
                if not np.any(vertebra_mask):
                    continue

                vertebra_coords = np.argwhere(vertebra_mask)
                vertebra_centroid = vertebra_coords.mean(axis=0)

                # Границы позвонка по Y
                vert_y_min, vert_y_max = vertebra_coords[:, 1].min(), vertebra_coords[:, 1].max()

                # Проверяем, находится ли позвонок выше диска
                if vert_y_max < disk_y_min:  # Позвонок полностью выше диска
                    gap = disk_y_min - vert_y_max
                    if gap < min_upper_gap:
                        min_upper_gap = gap
                        upper_vertebra = {
                            'label': vertebra_label,
                            'mask': vertebra_mask,
                            'centroid': vertebra_centroid.tolist(),
                            'coords': vertebra_coords,
                            'y_bounds': (vert_y_min, vert_y_max)
                        }

                # Проверяем, находится ли позвонок ниже диска
                elif vert_y_min > disk_y_max:  # Позвонок полностью ниже диска
                    gap = vert_y_min - disk_y_max
                    if gap < min_lower_gap:
                        min_lower_gap = gap
                        lower_vertebra = {
                            'label': vertebra_label,
                            'mask': vertebra_mask,
                            'centroid': vertebra_centroid.tolist(),
                            'coords': vertebra_coords,
                            'y_bounds': (vert_y_min, vert_y_max)
                        }

            return upper_vertebra, lower_vertebra

        except Exception as e:
            logger.error(f"Ошибка при поиске соседних позвонков: {e}")
            return None, None

    def _calculate_vertebral_displacement(self,
                                        upper_vertebra: Dict,
                                        lower_vertebra: Dict) -> Tuple[float, float]:
        """
        Вычисляет смещение позвонков для спондилолистеза.
        Измеряет смещение верхнего позвонка относительно нижнего в сагиттальной плоскости.

        Args:
            upper_vertebra: Данные верхнего позвонка
            lower_vertebra: Данные нижнего позвонка

        Returns:
            Кортеж (смещение_в_мм, смещение_в_процентах)
        """
        try:
            upper_coords = upper_vertebra['coords']
            lower_coords = lower_vertebra['coords']

            # Для точного измерения листеза нужно найти задние края позвонков
            # В сагиттальной плоскости это обычно максимальные Z координаты
            upper_posterior_edge = self._find_posterior_edge(upper_coords)
            lower_posterior_edge = self._find_posterior_edge(lower_coords)

            # Также находим передние края для определения размера позвонка
            lower_anterior_edge = self._find_anterior_edge(lower_coords)

            # Смещение в передне-заднем направлении (сагиттальная плоскость)
            # Положительное значение = антелистез, отрицательное = ретролистез
            displacement_voxels = upper_posterior_edge[2] - lower_posterior_edge[2]
            displacement_mm = displacement_voxels * self.voxel_spacing[2]

            # Размер нижнего позвонка в передне-заднем направлении для расчета процента
            lower_ap_size_voxels = lower_posterior_edge[2] - lower_anterior_edge[2]
            lower_ap_size_mm = lower_ap_size_voxels * self.voxel_spacing[2]

            # Процентное смещение по классификации Meyerding
            displacement_percentage = abs(displacement_mm / lower_ap_size_mm) * 100 if lower_ap_size_mm > 0 else 0

            logger.info(f"Displacement: {displacement_mm:.2f}mm ({displacement_percentage:.1f}%)")

            return displacement_mm, displacement_percentage

        except Exception as e:
            logger.error(f"Ошибка при вычислении смещения позвонков: {e}")
            return 0.0, 0.0

    def _find_posterior_edge(self, vertebra_coords: np.ndarray) -> np.ndarray:
        """
        Находит задний край позвонка (максимальная Z координата).

        Args:
            vertebra_coords: Координаты позвонка

        Returns:
            Координаты заднего края
        """
        try:
            # Находим точки с максимальной Z координатой
            max_z = vertebra_coords[:, 2].max()
            posterior_points = vertebra_coords[vertebra_coords[:, 2] == max_z]

            # Возвращаем центроид задних точек
            return posterior_points.mean(axis=0)

        except Exception as e:
            logger.error(f"Ошибка при поиске заднего края позвонка: {e}")
            return vertebra_coords.mean(axis=0)

    def _find_anterior_edge(self, vertebra_coords: np.ndarray) -> np.ndarray:
        """
        Находит передний край позвонка (минимальная Z координата).

        Args:
            vertebra_coords: Координаты позвонка

        Returns:
            Координаты переднего края
        """
        try:
            # Находим точки с минимальной Z координатой
            min_z = vertebra_coords[:, 2].min()
            anterior_points = vertebra_coords[vertebra_coords[:, 2] == min_z]

            # Возвращаем центроид передних точек
            return anterior_points.mean(axis=0)

        except Exception as e:
            logger.error(f"Ошибка при поиске переднего края позвонка: {e}")
            return vertebra_coords.mean(axis=0)

    def _classify_spondylolisthesis_grade(self, displacement_percentage: float) -> int:
        """
        Классифицирует степень спондилолистеза по Meyerding.

        Grade I: 0-25% смещения
        Grade II: 25-50% смещения
        Grade III: 50-75% смещения
        Grade IV: 75-100% смещения
        Grade V: >100% смещения (спондилоптоз)

        Args:
            displacement_percentage: Процент смещения

        Returns:
            Степень листеза (1-5)
        """
        if displacement_percentage < 5:
            return 0  # Нет значимого листеза
        elif displacement_percentage <= 25:
            return 1
        elif displacement_percentage <= 50:
            return 2
        elif displacement_percentage <= 75:
            return 3
        elif displacement_percentage <= 100:
            return 4
        else:
            return 5  # Спондилоптоз

    def _determine_displacement_direction(self,
                                        upper_vertebra: Dict,
                                        lower_vertebra: Dict) -> str:
        """
        Определяет направление смещения при спондилолистезе.

        Args:
            upper_vertebra: Данные верхнего позвонка
            lower_vertebra: Данные нижнего позвонка

        Returns:
            Направление смещения ('anterolisthesis', 'retrolisthesis', 'lateral_listhesis')
        """
        try:
            upper_coords = upper_vertebra['coords']
            lower_coords = lower_vertebra['coords']

            # Находим задние края для точного определения смещения
            upper_posterior = self._find_posterior_edge(upper_coords)
            lower_posterior = self._find_posterior_edge(lower_coords)

            # Смещение в передне-заднем направлении (Z ось)
            ap_displacement = upper_posterior[2] - lower_posterior[2]

            # Смещение в боковом направлении (X ось)
            upper_centroid = np.array(upper_vertebra['centroid'])
            lower_centroid = np.array(lower_vertebra['centroid'])
            lateral_displacement = abs(upper_centroid[0] - lower_centroid[0])

            # Определяем преобладающее направление
            if abs(ap_displacement) > lateral_displacement:
                if ap_displacement > 0:
                    return 'anterolisthesis'  # Передний листез
                else:
                    return 'retrolisthesis'   # Задний листез
            else:
                return 'lateral_listhesis'    # Боковой листез

        except Exception as e:
            logger.error(f"Ошибка при определении направления смещения: {e}")
            return 'unknown'

    def _classify_herniation_severity(self, volume_mm3: float, max_protrusion_mm: float) -> str:
        """
        Классифицирует тяжесть грыжи.

        Args:
            volume_mm3: Объем грыжи в мм³
            max_protrusion_mm: Максимальное выпячивание в мм

        Returns:
            Степень тяжести ('mild', 'moderate', 'severe')
        """
        if max_protrusion_mm < 3 and volume_mm3 < 100:
            return 'mild'
        elif max_protrusion_mm < 6 and volume_mm3 < 500:
            return 'moderate'
        else:
            return 'severe'

    def _classify_spondylolisthesis_severity(self, grade: int) -> str:
        """
        Классифицирует тяжесть спондилолистеза.

        Args:
            grade: Степень по Meyerding

        Returns:
            Степень тяжести
        """
        severity_map = {
            0: 'none',
            1: 'mild',
            2: 'moderate',
            3: 'severe',
            4: 'very_severe',
            5: 'spondyloptosis'
        }
        return severity_map.get(grade, 'unknown')


def measure_all_pathologies(mri_data: List[np.ndarray],
                           mask_data: np.ndarray,
                           disk_results: Dict[int, Dict],
                           voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[int, Dict]:
    """
    Измеряет патологии для всех дисков с обнаруженными патологиями.
    
    Args:
        mri_data: Данные МРТ изображения
        mask_data: Данные сегментации
        disk_results: Результаты grading анализа для всех дисков
        voxel_spacing: Размер вокселя в мм
        
    Returns:
        Словарь с измерениями патологий для каждого диска
    """
    measurements = {}
    measurer = PathologyMeasurements(voxel_spacing)
    
    for disk_label, disk_result in disk_results.items():
        if 'error' in disk_result or 'predictions' not in disk_result:
            continue
        
        predictions = disk_result['predictions']
        level_name = disk_result.get('level_name', f'Disk_{disk_label}')
        
        # Проверяем, есть ли патологии для измерения
        has_herniation = predictions.get('Herniation', 0) > 0
        has_spondylolisthesis = predictions.get('Spondylolisthesis', 0) > 0
        
        if has_herniation or has_spondylolisthesis:
            logger.info(f"Измеряем патологии для диска {level_name}: "
                       f"грыжа={has_herniation}, листез={has_spondylolisthesis}")
            
            disk_measurements = measurer.measure_disk_pathologies(
                mri_data, mask_data, disk_label, predictions, level_name
            )
            measurements[disk_label] = disk_measurements
        else:
            # Даже если нет патологий, сохраняем базовые измерения диска
            disk_mask = (mask_data == disk_label)
            if np.any(disk_mask):
                basic_measurements = measurer._measure_disk_basic(disk_mask)
                measurements[disk_label] = {
                    'disk_label': disk_label,
                    'level_name': level_name,
                    'disk_measurements': basic_measurements,
                    'herniation': None,
                    'spondylolisthesis': None
                }
    
    return measurements