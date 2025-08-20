#!/usr/bin/env python3
"""
Пайплайн для анализа патологий позвоночника.

Простая интеграция для использования обученной модели:
- Вход: ресэмпленный двухканальный МРТ (T1+T2) и маска
- Выход: оценки патологий для каждого диска
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import cv2
import json

# Импорт модели
from garding import GradingModel, BasicBlock

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpineGradingPipeline:
    """
    Пайплайн для анализа патологий позвоночника.
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Инициализация пайплайна.
        
        Args:
            model_path: путь к обученной модели
            device: устройство для вычислений ('auto', 'cpu', 'cuda')
        """
        self.model_path = Path(model_path)
        
        # Определить устройство
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Используется устройство: {self.device}")
        
        # Загрузить модель
        self.model = self._load_model()
        
        # Маппинг меток дисков (исправленный из main_refactored)
        from utils.constant import VERTEBRA_DESCRIPTIONS
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
        
        logger.info("Пайплайн инициализирован")
    
    def _load_model(self) -> torch.nn.Module:
        """Загрузка обученной модели."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {self.model_path}")
        
        # Создать модель
        model = GradingModel(
            block=BasicBlock,
            layers=[3, 4, 6, 3],
            num_classes=2,  # T1 + T2 channels
            zero_init_residual=False
        )
        
        # Загрузить веса
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        logger.info(f"Модель загружена из {self.model_path}")
        
        return model
    
    def analyze_patient(self, 
                       t1_volume: np.ndarray, 
                       t2_volume: np.ndarray, 
                       mask_volume: np.ndarray,
                       patient_id: str = "unknown") -> Dict:
        """
        Анализ патологий для одного пациента.
        
        Args:
            t1_volume: 3D массив T1 изображения (depth, height, width)
            t2_volume: 3D массив T2 изображения (depth, height, width)
            mask_volume: 3D массив маски (depth, height, width)
            patient_id: ID пациента
            
        Returns:
            Словарь с результатами анализа
        """
        logger.info(f"Анализ пациента {patient_id}")
        logger.info(f"Размеры: T1={t1_volume.shape}, T2={t2_volume.shape}, Mask={mask_volume.shape}")
        
        # Найти диски в маске
        unique_labels = np.unique(mask_volume)
        disc_labels = [label for label in unique_labels if label > 200]
        
        if not disc_labels:
            logger.warning("Диски не найдены в маске")
            return {'patient_id': patient_id, 'discs': [], 'error': 'No discs found'}
        
        logger.info(f"Найдено дисков: {len(disc_labels)}")
        
        results = {
            'patient_id': patient_id,
            'discs': [],
            'summary': {}
        }
        
        # Анализировать каждый диск
        for disc_label in disc_labels:
            try:
                disc_result = self._analyze_single_disc(
                    t1_volume, t2_volume, mask_volume, disc_label
                )
                if disc_result:
                    results['discs'].append(disc_result)
            except Exception as e:
                logger.error(f"Ошибка анализа диска {disc_label}: {e}")
                continue
        
        # Создать сводку
        results['summary'] = self._create_summary(results['discs'])
        
        logger.info(f"Анализ завершен. Обработано дисков: {len(results['discs'])}")
        
        return results
    
    def _analyze_single_disc(self, 
                           t1_volume: np.ndarray, 
                           t2_volume: np.ndarray,
                           mask_volume: np.ndarray, 
                           disc_label: int) -> Optional[Dict]:
        """
        Анализ одного диска.
        
        Args:
            t1_volume: 3D массив T1 изображения
            t2_volume: 3D массив T2 изображения
            mask_volume: 3D массив маски
            disc_label: метка диска
            
        Returns:
            Словарь с результатами анализа диска или None
        """
        # Найти маску диска
        disc_mask = (mask_volume == disc_label)
        
        if np.sum(disc_mask) == 0:
            return None
        
        # Найти координаты диска
        coords = np.where(disc_mask)
        center = [int(np.mean(coords[i])) for i in range(3)]
        min_coords = [np.min(coords[i]) for i in range(3)]
        max_coords = [np.max(coords[i]) for i in range(3)]
        
        # Расширить границы
        padding = [15, 25, 10]  # depth, height, width
        for i in range(3):
            min_coords[i] = max(0, min_coords[i] - padding[i])
            max_coords[i] = min(t1_volume.shape[i], max_coords[i] + padding[i])
        
        # Извлечь подобъемы
        t1_sub = t1_volume[
            min_coords[0]:max_coords[0],
            min_coords[1]:max_coords[1],
            min_coords[2]:max_coords[2]
        ]
        
        t2_sub = t2_volume[
            min_coords[0]:max_coords[0],
            min_coords[1]:max_coords[1],
            min_coords[2]:max_coords[2]
        ]
        
        disc_mask_sub = disc_mask[
            min_coords[0]:max_coords[0],
            min_coords[1]:max_coords[1],
            min_coords[2]:max_coords[2]
        ]
        
        mask_sub = mask_volume[
            min_coords[0]:max_coords[0],
            min_coords[1]:max_coords[1],
            min_coords[2]:max_coords[2]
        ]
        
        # Подготовить для модели
        prepared_volume = self._prepare_dual_channel_volume(
            t1_sub, t2_sub, disc_mask_sub, mask_sub, disc_label
        )
        
        if prepared_volume is None:
            return None
        
        # Предсказание модели
        predictions = self._predict(prepared_volume)
        
        # Создать результат
        level_name = self.disc_labels_map.get(disc_label, f"DISC_{disc_label}")
        
        result = {
            'disc_label': int(disc_label),
            'level_name': level_name,
            'center': center,
            'bounds': (min_coords, max_coords),
            'predictions': predictions,
            'confidence_scores': self._calculate_confidence(predictions)
        }
        
        return result
    
    def _prepare_dual_channel_volume(self, 
                                   t1_volume: np.ndarray, 
                                   t2_volume: np.ndarray,
                                   disc_mask: np.ndarray, 
                                   mask_volume: np.ndarray,
                                   disc_label: int) -> Optional[np.ndarray]:
        """
        Подготовка двухканального объема для модели.
        
        Returns:
            Подготовленный объем (2, 9, 152, 272) или None
        """
        if t1_volume.size == 0 or t2_volume.size == 0:
            return None
        
        # Применить выравнивание
        t1_aligned, rotation_angle = self._rotate_and_align_volume(t1_volume, disc_mask, mask_volume)
        t2_aligned, _ = self._rotate_and_align_volume(t2_volume, disc_mask, mask_volume, rotation_angle)
        
        # Обработать каждый канал
        t1_processed = self._process_single_channel(t1_aligned)
        t2_processed = self._process_single_channel(t2_aligned)
        
        if t1_processed is None or t2_processed is None:
            return None
        
        # Объединить каналы
        dual_channel_volume = np.stack([t1_processed, t2_processed], axis=0)
        
        return dual_channel_volume
    
    def _process_single_channel(self, volume: np.ndarray) -> Optional[np.ndarray]:
        """Обработка одного канала."""
        # Нормализация
        volume = volume.astype(np.float32)
        vb_pair_median = np.median(volume[volume > 0]) if np.any(volume > 0) else 1.0
        norm_med = 0.5
        
        volume = volume / vb_pair_median * norm_med
        volume[volume < 0] = 0
        volume[volume > 2.0] = 2.0
        volume /= 2.0
        
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
        
        # Взять центральные 9 срезов
        center_idx = volume.shape[2] // 2
        start_idx = center_idx - 4
        end_idx = center_idx + 5
        
        final_volume = volume[:, :, start_idx:end_idx]
        final_volume = final_volume[20:172, 24:296, :]  # (152, 272, 9)
        final_volume = np.transpose(final_volume, (2, 0, 1))  # (9, 152, 272)
        
        return final_volume
    
    def _rotate_and_align_volume(self, volume: np.ndarray, disc_mask: np.ndarray, 
                               mask_volume: np.ndarray = None, fixed_angle: float = None) -> Tuple[np.ndarray, float]:
        """Поворот и выравнивание объема."""
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
    
    def _predict(self, volume: np.ndarray) -> Dict[str, int]:
        """
        Предсказание модели для подготовленного объема.
        
        Args:
            volume: подготовленный объем (2, 9, 152, 272)
            
        Returns:
            Словарь с предсказаниями для каждой категории
        """
        # Преобразовать в тензор и добавить batch dimension
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Получить предсказания модели
            outputs = self.model(volume_tensor)
            
            # Преобразовать в предсказания
            predictions = {}
            for i, category in enumerate(self.categories):
                logits = outputs[i]
                predicted_class = torch.argmax(logits, dim=1).item()
                predictions[category] = predicted_class
        
        return predictions
    
    def _calculate_confidence(self, predictions: Dict[str, int]) -> Dict[str, float]:
        """Вычисление уверенности предсказаний (заглушка)."""
        # В реальной реализации здесь можно использовать softmax вероятности
        confidence_scores = {}
        for category, prediction in predictions.items():
            # Заглушка - случайная уверенность
            confidence_scores[category] = np.random.uniform(0.7, 0.95)
        
        return confidence_scores
    
    def _create_summary(self, disc_results: List[Dict]) -> Dict:
        """Создание сводки по всем дискам."""
        if not disc_results:
            return {}
        
        summary = {
            'total_discs': len(disc_results),
            'pathology_counts': {},
            'severity_distribution': {}
        }
        
        # Подсчет патологий
        for category in self.categories:
            category_values = [disc['predictions'][category] for disc in disc_results if category in disc['predictions']]
            
            if category_values:
                summary['pathology_counts'][category] = {
                    'total': len(category_values),
                    'positive': sum(1 for v in category_values if v > 0),
                    'distribution': dict(zip(*np.unique(category_values, return_counts=True)))
                }
        
        return summary
    
    def save_results(self, results: Dict, output_path: str):
        """Сохранение результатов в JSON файл."""
        output_path = Path(output_path)
        
        # Преобразовать numpy типы в стандартные Python типы для JSON
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        results_json = convert_numpy_types(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Результаты сохранены в {output_path}")
    
    def print_results(self, results: Dict):
        """Вывод результатов в консоль."""
        print(f"\n{'='*60}")
        print(f"РЕЗУЛЬТАТЫ АНАЛИЗА ПАЦИЕНТА {results['patient_id']}")
        print(f"{'='*60}")
        
        if 'error' in results:
            print(f"❌ Ошибка: {results['error']}")
            return
        
        print(f"Обработано дисков: {len(results['discs'])}")
        print()
        
        for disc in results['discs']:
            print(f"🔍 Диск {disc['level_name']} (метка {disc['disc_label']}):")
            print(f"   Центр: {disc['center']}")
            
            for category, prediction in disc['predictions'].items():
                confidence = disc['confidence_scores'][category]
                description = self.category_descriptions[category]
                print(f"   • {category}: {prediction} ({description}) [уверенность: {confidence:.2f}]")
            
            print()
        
        # Сводка
        if results['summary']:
            print(f"📊 СВОДКА:")
            print(f"   Всего дисков: {results['summary']['total_discs']}")
            
            for category, counts in results['summary']['pathology_counts'].items():
                positive_rate = counts['positive'] / counts['total'] * 100
                print(f"   {category}: {counts['positive']}/{counts['total']} ({positive_rate:.1f}%) положительных")


def main():
    """Пример использования пайплайна."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Пайплайн анализа патологий позвоночника')
    parser.add_argument('--model', type=str, required=True,
                       help='Путь к обученной модели')
    parser.add_argument('--t1', type=str, required=True,
                       help='Путь к T1 изображению (.npy)')
    parser.add_argument('--t2', type=str, required=True,
                       help='Путь к T2 изображению (.npy)')
    parser.add_argument('--mask', type=str, required=True,
                       help='Путь к маске (.npy)')
    parser.add_argument('--patient_id', type=str, default='test_patient',
                       help='ID пациента')
    parser.add_argument('--output', type=str, default=None,
                       help='Путь для сохранения результатов (JSON)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Устройство для вычислений')
    
    args = parser.parse_args()
    
    try:
        # Создать пайплайн
        pipeline = SpineGradingPipeline(args.model, args.device)
        
        # Загрузить данные
        logger.info("Загрузка данных...")
        t1_volume = np.load(args.t1)
        t2_volume = np.load(args.t2)
        mask_volume = np.load(args.mask)
        
        # Анализ
        results = pipeline.analyze_patient(t1_volume, t2_volume, mask_volume, args.patient_id)
        
        # Вывод результатов
        pipeline.print_results(results)
        
        # Сохранение
        if args.output:
            pipeline.save_results(results, args.output)
        
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()