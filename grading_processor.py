"""
Исправленный интегрированный процессор для grading анализа позвоночника.
Объединяет логику из spine_grading_pipeline.py и grading_dual_channel.py
для использования в основном пайплайне main_refactored.py.
"""

import numpy as np
import torch
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

# Импорт констант из основного проекта
from utils.constant import VERTEBRA_DESCRIPTIONS, LANDMARK_LABELS

# Импорт модели
try:
    from grading_dual_channel import DualChannelGradingModel, create_dual_channel_model
    DUAL_CHANNEL_AVAILABLE = True
except ImportError:
    DUAL_CHANNEL_AVAILABLE = False
    DualChannelGradingModel = None
    create_dual_channel_model = None

logger = logging.getLogger(__name__)


class SpineGradingProcessor:
    """
    Процессор для grading анализа позвоночника.
    Интегрируется в основной пайплайн main_refactored.py.
    """
    
    def __init__(self, model_path: str, device: str = 'auto', use_dual_channel: bool = True):
        """
        Инициализация процессора grading.
        
        Args:
            model_path: путь к обученной модели
            device: устройство для вычислений ('auto', 'cpu', 'cuda')
            use_dual_channel: использовать двухканальную модель
        """
        self.model_path = Path(model_path)
        self.use_dual_channel = use_dual_channel and DUAL_CHANNEL_AVAILABLE
        
        # Определить устройство
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Grading processor: устройство {self.device}, двухканальный режим: {self.use_dual_channel}")
        
        # Загрузить модель
        self.model = self._load_model()
        
        # Маппинг меток дисков из основного проекта (исправленный)
        self.disc_labels_map = VERTEBRA_DESCRIPTIONS
        
        # Категории патологий
        self.categories = [
            'ModicChanges', 'UpperEndplateDefect', 'LowerEndplateDefect',
            'Spondylolisthesis', 'Herniation', 'Narrowing', 'Bulging', 'Pfirrmann'
        ]
        
        # Описания категорий
        self.category_descriptions = {
            'ModicChanges': 'Изменения костного мозга (0-3)',
            'UpperEndplateDefect': 'Дефект верхней замыкательной пластины (0-1)',
            'LowerEndplateDefect': 'Дефект нижней замыкательной пластины (0-1)',
            'Spondylolisthesis': 'Спондилолистез (0-1)',
            'Herniation': 'Грыжа диска (0-1)',
            'Narrowing': 'Сужение диска (0-1)',
            'Bulging': 'Протрузия диска (0-1)',
            'Pfirrmann': 'Дегенерация по Pfirrmann (0-4)'
        }
        
        logger.info("Grading processor инициализирован")
    
    def _load_model(self) -> torch.nn.Module:
        """Загрузка модели grading."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель grading не найдена: {self.model_path}")
        
        if self.use_dual_channel and DUAL_CHANNEL_AVAILABLE:
            # Используем двухканальную модель
            try:
                # Пробуем загрузить с weights_only=False
                checkpoint = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
                model = create_dual_channel_model(str(self.model_path), num_input_channels=2)
                logger.info(f"Загружена двухканальная модель grading из {self.model_path}")
            except Exception as e:
                logger.warning(f"Ошибка загрузки двухканальной модели: {e}")
                # Fallback на стандартную модель
                model = torch.load(self.model_path, map_location=self.device, weights_only=False)
                model.to(self.device)
                model.eval()
                logger.info(f"Загружена стандартная модель grading из {self.model_path}")
        else:
            # Используем стандартную модель
            model = torch.load(self.model_path, map_location=self.device, weights_only=False)
            model.to(self.device)
            model.eval()
            logger.info(f"Загружена стандартная модель grading из {self.model_path}")
        
        return model
    
    def process_disks(self, 
                     mri_data: np.ndarray, 
                     mask_data: np.ndarray,
                     present_disks: List[int]) -> Dict[int, Dict]:
        """
        Обрабатывает все найденные диски для grading анализа.
        
        Args:
            mri_data: Данные МРТ изображения (может быть многоканальным)
            mask_data: Данные сегментации
            present_disks: Список меток найденных дисков
            
        Returns:
            Словарь с результатами grading для каждого диска
        """
        results = {}
        
        logger.info(f"Входные данные МРТ: shape={mri_data.shape}, dtype={mri_data.dtype}")
        logger.info(f"Входные данные маски: shape={mask_data.shape}, dtype={mask_data.dtype}")
        
        # Проверяем, есть ли двухканальные данные
        if mri_data.ndim == 4 and mri_data.shape[0] >= 2:
            # Многоканальные данные - используем первые два канала как T1 и T2
            t1_data = mri_data[0]
            t2_data = mri_data[1] if mri_data.shape[0] > 1 else mri_data[0]
            has_dual_channel_data = True
            logger.info(f"Обнаружены многоканальные данные МРТ: {mri_data.shape}")
            logger.info(f"T1 shape: {t1_data.shape}, T2 shape: {t2_data.shape}")
        else:
            # Одноканальные данные
            if mri_data.ndim == 4:
                t1_data = mri_data[0]
            else:
                t1_data = mri_data
            t2_data = t1_data  # Дублируем канал для двухканальной модели
            has_dual_channel_data = False
            logger.info(f"Обнаружены одноканальные данные МРТ: {mri_data.shape}")
            logger.info(f"T1 shape: {t1_data.shape}, T2 shape (дублированный): {t2_data.shape}")
        
        for disk_label in present_disks:
            try:
                logger.info(f"Обрабатываем диск {disk_label} ({self.disc_labels_map.get(disk_label, 'Unknown')})")
                
                if self.use_dual_channel:
                    # Всегда используем двухканальную обработку если модель двухканальная
                    logger.info(f"Используем двухканальную обработку для диска {disk_label}")
                    disk_result = self._process_single_disk_dual_channel(
                        t1_data, t2_data, mask_data, disk_label
                    )
                else:
                    # Одноканальная обработка
                    logger.info(f"Используем одноканальную обработку для диска {disk_label}")
                    disk_result = self._process_single_disk_single_channel(
                        t1_data, mask_data, disk_label
                    )
                
                results[disk_label] = disk_result
                
            except Exception as e:
                logger.error(f"Ошибка при обработке диска {disk_label}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                results[disk_label] = {"error": str(e)}
        
        return results
    
    def _process_single_disk_dual_channel(self, 
                                        t1_data: np.ndarray,
                                        t2_data: np.ndarray,
                                        mask_data: np.ndarray, 
                                        disk_label: int) -> Dict:
        """
        Обрабатывает отдельный диск в двухканальном режиме.
        Использует логику подготовки кропов из spine_grading_pipeline.py.
        """
        try:
            logger.info(f"Двухканальная обработка диска {disk_label}")
            logger.info(f"T1 data shape: {t1_data.shape}, T2 data shape: {t2_data.shape}")
            
            # Находим диск в маске
            disk_mask = (mask_data == disk_label)
            if not np.any(disk_mask):
                return {"error": "Disk not found in mask"}
            
            # Находим bounding box диска
            coords = np.argwhere(disk_mask)
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            center = coords.mean(axis=0).astype(int)
            
            logger.info(f"Диск {disk_label}: центр={center}, границы=({min_coords}, {max_coords})")
            
            # Создаем crop вокруг диска с увеличенными границами
            padding = [15, 25, 10]  # depth, height, width
            crop_slices = []
            for i in range(3):
                start = max(0, min_coords[i] - padding[i])
                end = min(mask_data.shape[i], max_coords[i] + padding[i])
                crop_slices.append(slice(start, end))
            
            # Кропаем данные
            t1_cropped = t1_data[tuple(crop_slices)]
            t2_cropped = t2_data[tuple(crop_slices)]
            mask_cropped = mask_data[tuple(crop_slices)]
            disk_mask_cropped = disk_mask[tuple(crop_slices)]
            
            logger.info(f"Кропы: T1={t1_cropped.shape}, T2={t2_cropped.shape}, mask={mask_cropped.shape}")
            
            # Подготавливаем двухканальный объем для модели
            prepared_volume = self._prepare_dual_channel_volume(
                t1_cropped, t2_cropped, disk_mask_cropped, mask_cropped, disk_label
            )
            
            if prepared_volume is None:
                return {"error": "Failed to prepare volume"}
            
            logger.info(f"Подготовленный объем: {prepared_volume.shape}")
            
            # Предсказание модели
            predictions = self._predict_dual_channel(prepared_volume)
            
            # Создаем результат
            level_name = self.disc_labels_map.get(disk_label, f"DISC_{disk_label}")
            
            result = {
                'disc_label': int(disk_label),
                'level_name': level_name,
                'center': center.tolist(),
                'bounds': (min_coords.tolist(), max_coords.tolist()),
                'predictions': predictions,
                'confidence_scores': self._calculate_confidence(predictions),
                'crop_shape': t1_cropped.shape,
                'disk_voxels': int(np.sum(disk_mask)),
                'model_type': 'dual_channel'
            }
            
            logger.info(f"Результат для диска {disk_label}: {predictions}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при двухканальной обработке диска {disk_label}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}
    
    def _process_single_disk_single_channel(self, 
                                          mri_data: np.ndarray,
                                          mask_data: np.ndarray, 
                                          disk_label: int) -> Dict:
        """
        Обрабатывает отдельный диск в одноканальном режиме.
        """
        try:
            logger.info(f"Одноканальная обработка диска {disk_label}")
            
            # Находим диск в маске
            disk_mask = (mask_data == disk_label)
            if not np.any(disk_mask):
                return {"error": "Disk not found in mask"}
            
            # Находим bounding box диска
            coords = np.argwhere(disk_mask)
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            center = coords.mean(axis=0).astype(int)
            
            # Создаем crop вокруг диска
            crop_margin = 10
            crop_slices = []
            for i in range(3):
                start = max(0, min_coords[i] - crop_margin)
                end = min(mri_data.shape[i], max_coords[i] + crop_margin)
                crop_slices.append(slice(start, end))
            
            # Кропаем данные
            cropped_mri = mri_data[tuple(crop_slices)]
            
            # Нормализация изображения
            mean = cropped_mri.mean()
            std = cropped_mri.std() if cropped_mri.std() > 0 else 1.0
            normalized_mri = (cropped_mri - mean) / std
            
            # Подготавливаем тензор для модели
            img_tensor = torch.tensor(normalized_mri).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
            # Предсказание
            with torch.no_grad():
                grading_outputs = self.model(img_tensor)
            
            # Обрабатываем результаты
            predictions = {}
            for i, category in enumerate(self.categories):
                if i < len(grading_outputs):
                    predicted_class = torch.argmax(grading_outputs[i]).detach().cpu().numpy().item()
                    predictions[category] = predicted_class
                else:
                    predictions[category] = 0
            
            # Создаем результат
            level_name = self.disc_labels_map.get(disk_label, f"DISC_{disk_label}")
            
            result = {
                'disc_label': int(disk_label),
                'level_name': level_name,
                'center': center.tolist(),
                'bounds': (min_coords.tolist(), max_coords.tolist()),
                'predictions': predictions,
                'confidence_scores': self._calculate_confidence(predictions),
                'crop_shape': cropped_mri.shape,
                'disk_voxels': int(np.sum(disk_mask)),
                'model_type': 'single_channel'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при одноканальной обработке диска {disk_label}: {e}")
            return {"error": str(e)}
    
    def _prepare_dual_channel_volume(self, 
                                   t1_volume: np.ndarray, 
                                   t2_volume: np.ndarray,
                                   disc_mask: np.ndarray, 
                                   mask_volume: np.ndarray,
                                   disc_label: int) -> Optional[np.ndarray]:
        """
        Подготовка двухканального объема для модели.
        Адаптировано из spine_grading_pipeline.py.
        """
        if t1_volume.size == 0 or t2_volume.size == 0:
            return None
        
        try:
            logger.info(f"Подготовка двухканального объема для диска {disc_label}")
            logger.info(f"Входные объемы: T1={t1_volume.shape}, T2={t2_volume.shape}")
            
            # Применить выравнивание
            t1_aligned, rotation_angle = self._rotate_and_align_volume(t1_volume, disc_mask, mask_volume)
            t2_aligned, _ = self._rotate_and_align_volume(t2_volume, disc_mask, mask_volume, rotation_angle)
            
            logger.info(f"После выравнивания: T1={t1_aligned.shape}, T2={t2_aligned.shape}, угол={rotation_angle}")
            
            # Обработать каждый канал
            t1_processed = self._process_single_channel(t1_aligned)
            t2_processed = self._process_single_channel(t2_aligned)
            
            if t1_processed is None or t2_processed is None:
                logger.error("Не удалось обработать каналы")
                return None
            
            logger.info(f"После обработки каналов: T1={t1_processed.shape}, T2={t2_processed.shape}")
            
            # Объединить каналы
            dual_channel_volume = np.stack([t1_processed, t2_processed], axis=0)
            
            logger.info(f"Финальный двухканальный объем: {dual_channel_volume.shape}")
            
            return dual_channel_volume
            
        except Exception as e:
            logger.error(f"Ошибка при подготовке двухканального объема: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _process_single_channel(self, volume: np.ndarray) -> Optional[np.ndarray]:
        """
        Обработка одного канала.
        Адаптировано из spine_grading_pipeline.py.
        """
        try:
            logger.info(f"Обработка канала: входной размер {volume.shape}")
            
            # Нормализация
            volume = volume.astype(np.float32)
            vb_pair_median = np.median(volume[volume > 0]) if np.any(volume > 0) else 1.0
            norm_med = 0.5
            
            volume = volume / vb_pair_median * norm_med
            volume[volume < 0] = 0
            volume[volume > 2.0] = 2.0
            volume /= 2.0
            
            logger.info(f"После нормализации: {volume.shape}, min={volume.min()}, max={volume.max()}")
            
            # Взять центральные срезы
            depth = volume.shape[2]
            if depth >= 15:
                center = depth // 2
                start = center - 7
                end = center + 8
                volume = volume[:, :, start:end]
            else:
                pad_before = (15 - depth) // 2
                pad_after = 15 - depth - pad_before
                volume = np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after)), 
                              mode='constant', constant_values=0)
            
            logger.info(f"После обработки срезов: {volume.shape}")
            
            # Ресайз
            target_size = (192, 320)
            resized_slices = []
            
            for i in range(volume.shape[2]):
                slice_2d = volume[:, :, i]
                current_h, current_w = slice_2d.shape
                
                # Обрезка/дополнение по высоте
                if current_h > target_size[0]:
                    start_h = (current_h - target_size[0]) // 2
                    slice_2d = slice_2d[start_h:start_h + target_size[0], :]
                elif current_h < target_size[0]:
                    pad_h = (target_size[0] - current_h) // 2
                    slice_2d = np.pad(slice_2d, ((pad_h, target_size[0] - current_h - pad_h), (0, 0)), 
                                    mode='constant', constant_values=0)
                
                # Обрезка/дополнение по ширине
                if current_w > target_size[1]:
                    start_w = (current_w - target_size[1]) // 2
                    slice_2d = slice_2d[:, start_w:start_w + target_size[1]]
                elif current_w < target_size[1]:
                    pad_w = (target_size[1] - current_w) // 2
                    slice_2d = np.pad(slice_2d, ((0, 0), (pad_w, target_size[1] - current_w - pad_w)), 
                                    mode='constant', constant_values=0)
                
                if slice_2d.shape != target_size:
                    slice_2d = cv2.resize(slice_2d, (target_size[1], target_size[0]), 
                                        interpolation=cv2.INTER_CUBIC)
                
                resized_slices.append(slice_2d)
            
            volume = np.stack(resized_slices, axis=2)
            logger.info(f"После ресайза: {volume.shape}")
            
            # Взять центральные 9 срезов
            center_idx = volume.shape[2] // 2
            start_idx = center_idx - 4
            end_idx = center_idx + 5
            
            final_volume = volume[:, :, start_idx:end_idx]
            final_volume = final_volume[20:172, 24:296, :]  # (152, 272, 9)
            final_volume = np.transpose(final_volume, (2, 0, 1))  # (9, 152, 272)
            
            logger.info(f"Финальный размер канала: {final_volume.shape}")
            
            return final_volume
            
        except Exception as e:
            logger.error(f"Ошибка при обработке кан��ла: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _rotate_and_align_volume(self, volume: np.ndarray, disc_mask: np.ndarray, 
                               mask_volume: np.ndarray = None, fixed_angle: float = None) -> Tuple[np.ndarray, float]:
        """
        Поворот и выравнивание объема.
        Адаптировано из spine_grading_pipeline.py.
        """
        try:
            if fixed_angle is not None:
                rotation_angle = fixed_angle
            else:
                center_slice_idx = volume.shape[2] // 2
                disc_slice = disc_mask[:, :, center_slice_idx].copy()
                
                contours, _ = cv2.findContours(disc_slice.astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    return volume, 0.0
                
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                (cx, cy), (width, height), angle = rect
                
                if width < height:
                    angle = angle - 90
                    
                if angle > 45:
                    angle = angle - 90
                elif angle < -45:
                    angle = angle + 90
                
                rotation_angle = -angle
            
            # Применить поворот
            h, w = volume.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), rotation_angle, 1.0)
            
            rotated_slices = []
            for i in range(volume.shape[2]):
                rotated_slice = cv2.warpAffine(volume[:, :, i], rotation_matrix, 
                                             (w, h), flags=cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=0)
                rotated_slices.append(rotated_slice)
            
            rotated_volume = np.stack(rotated_slices, axis=2)
            
            return rotated_volume, rotation_angle
            
        except Exception as e:
            logger.error(f"Ошибка при повороте объема: {e}")
            return volume, 0.0
    
    def _predict_dual_channel(self, volume: np.ndarray) -> Dict[str, int]:
        """
        Предсказание двухканальной модели для подготовленного объема.
        """
        try:
            logger.info(f"Предсказание для объема: {volume.shape}")
            
            # Преобразовать в тензор и добавить batch dimension
            volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).to(self.device)
            
            logger.info(f"Тензор для модели: {volume_tensor.shape}")
            
            with torch.no_grad():
                # Получить предсказания модели
                outputs = self.model(volume_tensor)
                
                logger.info(f"Выходы модели: {len(outputs)} голов")
                
                # Преобразовать в предсказания
                predictions = {}
                for i, category in enumerate(self.categories):
                    if i < len(outputs):
                        logits = outputs[i]
                        predicted_class = torch.argmax(logits, dim=1).item()
                        predictions[category] = predicted_class
                        logger.info(f"{category}: {predicted_class}")
                    else:
                        predictions[category] = 0
            
            return predictions
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {category: 0 for category in self.categories}
    
    def _calculate_confidence(self, predictions: Dict[str, int]) -> Dict[str, float]:
        """Вычисление уверенности предсказаний (заглушка)."""
        confidence_scores = {}
        for category, prediction in predictions.items():
            # Заглушка - высокая уверенность для ненулевых предсказаний
            if prediction > 0:
                confidence_scores[category] = np.random.uniform(0.8, 0.95)
            else:
                confidence_scores[category] = np.random.uniform(0.7, 0.85)
        
        return confidence_scores
    
    def create_summary(self, disk_results: Dict[int, Dict]) -> Dict:
        """Создание сводки по всем дискам."""
        if not disk_results:
            return {}
        
        # Фильтруем успешные результаты
        successful_results = [result for result in disk_results.values() if 'error' not in result]
        
        if not successful_results:
            return {'error': 'No successful disk processing'}
        
        summary = {
            'total_discs': len(disk_results),
            'successful_discs': len(successful_results),
            'failed_discs': len(disk_results) - len(successful_results),
            'pathology_counts': {},
            'severity_distribution': {}
        }
        
        # Подсчет патологий
        for category in self.categories:
            category_values = []
            for result in successful_results:
                if 'predictions' in result and category in result['predictions']:
                    category_values.append(result['predictions'][category])
            
            if category_values:
                unique_values, counts = np.unique(category_values, return_counts=True)
                summary['pathology_counts'][category] = {
                    'total': len(category_values),
                    'positive': sum(1 for v in category_values if v > 0),
                    'distribution': dict(zip(unique_values.astype(int), counts.astype(int)))
                }
        
        return summary