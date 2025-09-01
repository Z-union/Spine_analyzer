"""
Модуль для измерения патологий позвоночника после grading анализа.
Измеряет размеры и объемы грыж, размеры листезов.
"""

import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Union
from scipy import ndimage
from skimage import measure, morphology
from config import settings

logger = logging.getLogger(__name__)


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
        Измеряет параметры грыжи/выбухания как геометрические излишки диска относительно сглаженной базовой формы
        в задней полуплоскости (по направлению к каналу, если доступен).
        """
        try:
            canal_mask = (mask_data == settings.CANAL_LABEL)
            herniation_mask, base_mask = self._detect_herniation_regions(mri_data, mask_data, disk_mask, canal_mask)

            if not np.any(herniation_mask):
                return {
                    'detected': False,
                    'volume_mm3': 0.0,
                    'max_protrusion_mm': 0.0,
                    'regions_count': 0
                }

            herniation_volume_voxels = int(np.sum(herniation_mask))
            herniation_volume_mm3 = float(herniation_volume_voxels * np.prod(self.voxel_spacing))

            base_boundary = self._find_disk_boundary(base_mask)
            max_protrusion_mm = float(self._calculate_max_protrusion(disk_mask, herniation_mask, base_boundary))

            labeled_regions, num_regions = ndimage.label(herniation_mask)
            regions_info = []
            for region_id in range(1, num_regions + 1):
                region_mask = (labeled_regions == region_id)
                region_volume = float(np.sum(region_mask) * np.prod(self.voxel_spacing))
                region_coords = np.argwhere(region_mask)
                region_centroid = region_coords.mean(axis=0)
                regions_info.append({
                    'region_id': int(region_id),
                    'volume_mm3': region_volume,
                    'centroid': region_centroid.tolist(),
                    'voxel_count': int(np.sum(region_mask))
                })

            return {
                'detected': True,
                'volume_mm3': herniation_volume_mm3,
                'volume_voxels': herniation_volume_voxels,
                'max_protrusion_mm': max_protrusion_mm,
                'regions_count': int(num_regions),
                'regions_info': regions_info,
                'severity': self._classify_herniation_severity(herniation_volume_mm3, max_protrusion_mm)
            }
        except Exception as e:
            logger.error(f"Ошибка при измерении грыжи: {e}")
            return {'error': str(e)}
    
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
    
    def _detect_herniation_regions(self, 
                                  mri_data: np.ndarray,
                                  mask_data: np.ndarray,
                                  disk_mask: np.ndarray,
                                  canal_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Геометрическое выделение выпячиваний/грыжи как излишков диска относительно базовой формы (морфологическое открытие)
        в задней полуплоскости, определенной по направлению к каналу (если доступен).

        Возвращает (herniation_mask_3d, base_mask_3d).
        """
        try:
            zdim, ydim, xdim = disk_mask.shape
            candidate = np.zeros_like(disk_mask, dtype=bool)
            base_mask = np.zeros_like(disk_mask, dtype=bool)

            # Структурный элемент ~2.5 мм по (y, x)
            r_mm = 2.5
            ry = max(1, int(round(r_mm / max(self.voxel_spacing[1], 1e-6))))
            rx = max(1, int(round(r_mm / max(self.voxel_spacing[2], 1e-6))))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rx + 1, 2 * ry + 1))

            coords = np.argwhere(disk_mask)
            if coords.size == 0:
                return candidate, base_mask
            zc = int(np.round(coords[:, 0].mean()))
            z_from = max(0, zc - 2)
            z_to = min(zdim, zc + 3)

            for z in range(z_from, z_to):
                disk_slice = disk_mask[z].astype(np.uint8)
                if disk_slice.sum() == 0:
                    continue

                base_slice = cv2.morphologyEx(disk_slice, cv2.MORPH_OPEN, kernel)
                base_mask[z] = base_slice.astype(bool)

                posterior_half = np.ones_like(disk_slice, dtype=bool)
                canal_slice = canal_mask[z]
                if canal_slice.any():
                    dyx_disk = np.argwhere(disk_slice > 0)
                    dyx_canal = np.argwhere(canal_slice > 0)
                    if dyx_disk.size > 0 and dyx_canal.size > 0:
                        cy, cx = dyx_disk.mean(axis=0)
                        ky, kx = dyx_canal.mean(axis=0)
                        v = np.array([ky - cy, kx - cx], dtype=np.float32)
                        n = np.linalg.norm(v)
                        if n > 0:
                            v /= n
                            yy, xx = np.mgrid[0:ydim, 0:xdim]
                            posterior_half = ((yy - cy) * v[0] + (xx - cx) * v[1]) > 0

                cand_slice = (disk_slice.astype(bool) & (~base_mask[z]) & posterior_half)
                cand_slice = morphology.remove_small_objects(cand_slice, min_size=8)
                candidate[z] = cand_slice

            candidate = morphology.remove_small_objects(candidate, min_size=16)
            return candidate, base_mask
        except Exception as e:
            logger.error(f"Ошибка при геометрическом выделении грыжи: {e}")
            return np.zeros_like(disk_mask, dtype=bool), np.zeros_like(disk_mask, dtype=bool)
    
    def _calculate_max_protrusion(self, 
                                 disk_mask: np.ndarray,
                                 herniation_mask: np.ndarray,
                                 disk_boundary: np.ndarray) -> float:
        """
        Вычисляет максимальное выпячивание (мм) как минимальное расстояние от каждой точки грыжи до базовой границы диска
        с учётом анизотропии вокселей (перевод координат в мм).
        """
        try:
            if not np.any(herniation_mask):
                return 0.0

            boundary_coords = np.argwhere(disk_boundary)
            herniation_coords = np.argwhere(herniation_mask)
            if boundary_coords.size == 0 or herniation_coords.size == 0:
                return 0.0

            zs, ys, xs = self.voxel_spacing
            boundary_mm = boundary_coords.astype(np.float32)
            boundary_mm[:, 0] *= zs
            boundary_mm[:, 1] *= ys
            boundary_mm[:, 2] *= xs

            max_min = 0.0
            for p in herniation_coords:
                pmm = p.astype(np.float32)
                pmm[0] *= zs; pmm[1] *= ys; pmm[2] *= xs
                d = np.sqrt(np.sum((boundary_mm - pmm) ** 2, axis=1))
                min_d = float(np.min(d))
                if min_d > max_min:
                    max_min = min_d
            return max_min
        except Exception as e:
            logger.error(f"Ошибка при вычислении максимального выпячивания: {e}")
            return 0.0
    
    def _find_adjacent_vertebrae(self, 
                                mask_data: np.ndarray,
                                disk_label: int,
                                disk_mask: np.ndarray) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Находит соседние позвонки для измерения листеза.
        
        Args:
            mask_data: Данные сегментации
            disk_label: Метка диска
            disk_mask: Маска диска
            
        Returns:
            Кортеж (верхний_позвонок, нижний_позвонок)
        """
        try:
            # Получаем центроид диска
            disk_coords = np.argwhere(disk_mask)
            disk_centroid = disk_coords.mean(axis=0)
            
            # Ищем позвонки в окрестности диска
            unique_labels = np.unique(mask_data)
            vertebra_labels = [label for label in unique_labels if 10 <= label <= 50]  # Примерный диапазон меток позвонков
            
            upper_vertebra = None
            lower_vertebra = None
            min_upper_distance = float('inf')
            min_lower_distance = float('inf')
            
            for vertebra_label in vertebra_labels:
                vertebra_mask = (mask_data == vertebra_label)
                if not np.any(vertebra_mask):
                    continue
                
                vertebra_coords = np.argwhere(vertebra_mask)
                vertebra_centroid = vertebra_coords.mean(axis=0)
                
                # Определяем, выше или ниже диска находится позвонок
                # Предполагаем, что ось Y направлена сверху вниз
                y_diff = vertebra_centroid[1] - disk_centroid[1]
                distance = np.sqrt(np.sum((vertebra_centroid - disk_centroid) ** 2))
                
                if y_diff < -10 and distance < min_upper_distance:  # Выше диска
                    min_upper_distance = distance
                    upper_vertebra = {
                        'label': vertebra_label,
                        'mask': vertebra_mask,
                        'centroid': vertebra_centroid.tolist(),
                        'coords': vertebra_coords
                    }
                elif y_diff > 10 and distance < min_lower_distance:  # Ниже диска
                    min_lower_distance = distance
                    lower_vertebra = {
                        'label': vertebra_label,
                        'mask': vertebra_mask,
                        'centroid': vertebra_centroid.tolist(),
                        'coords': vertebra_coords
                    }
            
            return upper_vertebra, lower_vertebra
            
        except Exception as e:
            logger.error(f"Ошибка при поиске соседних позвонков: {e}")
            return None, None
    
    def _calculate_vertebral_displacement(self, 
                                        upper_vertebra: Dict,
                                        lower_vertebra: Dict) -> Tuple[float, float]:
        """
        Вычисляет смещение позвонков.
        
        Args:
            upper_vertebra: Данные верхнего позвонка
            lower_vertebra: Данные нижнего позвонка
            
        Returns:
            Кортеж (смещение_в_мм, смещение_в_процентах)
        """
        try:
            upper_centroid = np.array(upper_vertebra['centroid'])
            lower_centroid = np.array(lower_vertebra['centroid'])
            
            # Вычисляем смещение в сагиттальной плоскости (обычно по оси X или Z)
            # Пред��олагаем, что смещение измеряется по оси Z (anterior-posterior)
            displacement_voxels = abs(upper_centroid[2] - lower_centroid[2])
            displacement_mm = displacement_voxels * self.voxel_spacing[2]
            
            # Вычисляем размер нижнего позвонка для расчета процентного смещения
            lower_coords = lower_vertebra['coords']
            lower_size_z = (lower_coords[:, 2].max() - lower_coords[:, 2].min()) * self.voxel_spacing[2]
            
            displacement_percentage = (displacement_mm / lower_size_z) * 100 if lower_size_z > 0 else 0
            
            return displacement_mm, displacement_percentage
            
        except Exception as e:
            logger.error(f"Ошибка при вычислении смещения позвонков: {e}")
            return 0.0, 0.0
    
    def _classify_spondylolisthesis_grade(self, displacement_percentage: float) -> int:
        """
        Классифицирует степень спондилолистеза по Meyerding.
        
        Args:
            displacement_percentage: Процент смещения
            
        Returns:
            Степень листеза (0-4)
        """
        if displacement_percentage < 25:
            return 1
        elif displacement_percentage < 50:
            return 2
        elif displacement_percentage < 75:
            return 3
        elif displacement_percentage < 100:
            return 4
        else:
            return 4  # Максимальная степень
    
    def _determine_displacement_direction(self, 
                                        upper_vertebra: Dict,
                                        lower_vertebra: Dict) -> str:
        """
        Определяет направление смещения.
        
        Args:
            upper_vertebra: Данные верхнего позвонка
            lower_vertebra: Данные нижнего позвонка
            
        Returns:
            Направление смещения ('anterior', 'posterior', 'lateral')
        """
        try:
            upper_centroid = np.array(upper_vertebra['centroid'])
            lower_centroid = np.array(lower_vertebra['centroid'])
            
            # Смещение по оси Z (anterior-posterior)
            z_displacement = upper_centroid[2] - lower_centroid[2]
            
            # Смещение по оси X (lateral)
            x_displacement = abs(upper_centroid[0] - lower_centroid[0])
            
            if abs(z_displacement) > x_displacement:
                return 'anterior' if z_displacement > 0 else 'posterior'
            else:
                return 'lateral'
                
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
            Степень тяже��ти
        """
        severity_map = {
            1: 'mild',
            2: 'moderate',
            3: 'severe',
            4: 'very_severe'
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