"""
Модуль для обработки исследований позвоночника с использованием Triton Inference Server.
"""

import time
import logging
import traceback
from typing import List, Dict, Optional

import numpy as np
import pydicom.dataset
from nibabel.nifti1 import Nifti1Image

from triton_client import TritonModelClient, triton_segmentation_inference, triton_grading_inference
from utils.constant import (
    LANDMARK_LABELS, VERTEBRA_DESCRIPTIONS,
    IDX_MODIC, IDX_UP_ENDPLATE, IDX_LOW_ENDPLATE, IDX_SPONDY, 
    IDX_HERN, IDX_NARROW, IDX_BULGE, IDX_PFIRRMAN
)


def process_study_triton(
    studies: List[pydicom.dataset.FileDataset],
    triton_client: TritonModelClient,
    model_names: Dict[str, str],
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Обрабатывает исследование позвоночника с использованием Triton Inference Server.
    
    Args:
        studies: Список DICOM datasets
        triton_client: Клиент Triton Inference Server
        model_names: Словарь с именами моделей на Triton
        logger: Logger для записи сообщений
        
    Returns:
        Словарь с результатами обработки
    """
    start_time = time.time()
    
    try:
        if logger:
            logger.info("Начинаем обработку исследования через Triton")
            logger.info(f"Количество DICOM файлов: {len(studies)}")
            logger.info(f"Доступные модели: {list(model_names.keys())}")
        
        # Выполняем сегментацию через Triton
        nifti_img, nifti_seg = run_segmentation_triton(
            studies, triton_client, model_names, logger
        )
        
        if nifti_img is None or nifti_seg is None:
            if logger:
                logger.error("Сегментация через Triton не удалась")
            return {"error": "Triton segmentation failed"}
        
        # Анализируем присутствующие диски
        seg_data = nifti_seg.get_fdata()
        unique_labels = np.unique(seg_data)
        present_disks = [label for label in LANDMARK_LABELS if label in unique_labels]
        
        if logger:
            logger.info(f"Найдены диски: {present_disks}")
        
        # Обрабатываем каждый диск с помощью Triton grading модели
        results = {}
        
        # Получаем данные изображения
        if hasattr(nifti_img, 'get_fdata'):
            if nifti_img.get_fdata().ndim == 4:
                # Берем второй канал (индекс 1) для grading
                mri_data = nifti_img.get_fdata()[1]
            else:
                mri_data = nifti_img.get_fdata()
        else:
            mri_data = np.asarray(nifti_img)
        
        mask_data = nifti_seg.get_fdata()
        
        for disk_label in present_disks:
            try:
                if logger:
                    logger.info(f"Обрабатываем диск {disk_label} ({VERTEBRA_DESCRIPTIONS.get(disk_label, 'Unknown')})")
                
                disk_result = process_single_disk_triton(
                    mri_data, mask_data, disk_label, 
                    triton_client, model_names["grading_model"], logger
                )
                results[disk_label] = disk_result
                
            except Exception as e:
                if logger:
                    logger.error(f"Ошибка при обработке диска {disk_label}: {e}")
                results[disk_label] = {"error": str(e)}
        
        elapsed_time = time.time() - start_time
        
        if logger:
            logger.info(f"Обработка через Triton завершена за {elapsed_time:.2f} секунд")
        
        return {
            "processing_time": elapsed_time,
            "processed_disks": len(results),
            "disk_results": results,
            "segmentation_shape": seg_data.shape,
            "unique_labels": unique_labels.tolist(),
            "inference_backend": "triton"
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Критическая ошибка при обработке через Triton: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}


def run_segmentation_triton(
    studies: List[pydicom.dataset.FileDataset],
    triton_client: TritonModelClient,
    model_names: Dict[str, str],
    logger: Optional[logging.Logger] = None
) -> tuple[Optional[Nifti1Image], Optional[Nifti1Image]]:
    """
    Выполняет сегментацию исследования через Triton.
    
    Args:
        studies: Список DICOM datasets
        triton_client: Клиент Triton
        model_names: Словарь с именами моделей
        logger: Logger для записи сообщений
        
    Returns:
        Кортеж (nifti_img, nifti_seg) или (None, None) если сегментация не удалась
    """
    try:
        if logger:
            logger.info("Выполняем сегментацию через Triton...")
        
        # Здесь должна быть логика преобразования DICOM в формат для Triton
        # Пока используем заглушку
        
        # Пример: создаем тестовые данные
        test_shape = (1, 1, 128, 128, 64)  # Batch, Channel, H, W, D
        test_input = np.random.rand(*test_shape).astype(np.float32)
        
        if logger:
            logger.info(f"Подготовлены данные для сегментации: {test_input.shape}")
        
        # Сагиттальная сегментация - шаг 1
        if logger:
            logger.info("Выполняем сагиттальную сегментацию - шаг 1")
        
        sag_step1_result = triton_segmentation_inference(
            triton_client, 
            model_names["sag_step_1_model"], 
            test_input
        )
        
        if sag_step1_result is None:
            if logger:
                logger.error("Ошибка в сагиттальной сегментации - шаг 1")
            return None, None
        
        # Сагиттальная сегментация - шаг 2
        if logger:
            logger.info("Выполняем сагиттальную сегментацию - шаг 2")
        
        sag_step2_result = triton_segmentation_inference(
            triton_client, 
            model_names["sag_step_2_model"], 
            test_input
        )
        
        if sag_step2_result is None:
            if logger:
                logger.error("Ошибка в сагиттальной сегментации - шаг 2")
            return None, None
        
        # Аксиальная сегментация (опционально)
        if logger:
            logger.info("Выполняем аксиальную сегментацию")
        
        ax_result = triton_segmentation_inference(
            triton_client, 
            model_names["ax_model"], 
            test_input
        )
        
        # Создаем заглушки для NIfTI изображений
        # В реальной реализации здесь должна быть логика постобработки результатов Triton
        dummy_img_data = np.random.rand(128, 128, 64)
        dummy_seg_data = np.random.randint(0, 100, (128, 128, 64))
        
        # Создаем NIfTI изображения
        affine = np.eye(4)
        nifti_img = Nifti1Image(dummy_img_data, affine)
        nifti_seg = Nifti1Image(dummy_seg_data, affine)
        
        if logger:
            logger.info("Сегментация через Triton завершена успешно")
        
        return nifti_img, nifti_seg
        
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при сегментации через Triton: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None


def process_single_disk_triton(
    mri_data: np.ndarray,
    mask_data: np.ndarray,
    disk_label: int,
    triton_client: TritonModelClient,
    grading_model_name: str,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Обрабатывает отдельный диск для grading через Triton.
    
    Args:
        mri_data: Данные МРТ изображения
        mask_data: Данные сегментации
        disk_label: Метка диска
        triton_client: Клиент Triton
        grading_model_name: Имя модели grading на Triton
        logger: Logger для записи сообщений
        
    Returns:
        Словарь с результатами grading для диска
    """
    try:
        # Находим диск в маске
        disk_mask = (mask_data == disk_label)
        if not np.any(disk_mask):
            if logger:
                logger.warning(f"Диск {disk_label} не найден в маске")
            return {"error": "Disk not found in mask"}
        
        # Находим bounding box диска
        coords = np.argwhere(disk_mask)
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        
        # Создаем crop вокруг диска
        crop_margin = 10
        crop_slices = []
        for i in range(3):
            start = max(0, min_coords[i] - crop_margin)
            end = min(mri_data.shape[i], max_coords[i] + crop_margin)
            crop_slices.append(slice(start, end))
        
        # Кропаем данные
        cropped_mri = mri_data[tuple(crop_slices)]
        
        # Нормализация
        mean = cropped_mri.mean()
        std = cropped_mri.std() if cropped_mri.std() > 0 else 1.0
        normalized_mri = (cropped_mri - mean) / std
        
        # Подготавливаем для Triton (добавляем batch и channel dimensions)
        input_data = normalized_mri[np.newaxis, np.newaxis, ...].astype(np.float32)
        
        if logger:
            logger.info(f"Диск {disk_label}: отправляем данные в Triton, форма: {input_data.shape}")
        
        # Выполняем grading через Triton
        predictions = triton_grading_inference(
            triton_client, 
            grading_model_name, 
            input_data
        )
        
        if predictions is None:
            if logger:
                logger.error(f"Ошибка grading через Triton для диска {disk_label}")
            return {"error": "Triton grading failed"}
        
        # Формируем результат
        result = {
            "modic": predictions[IDX_MODIC] if len(predictions) > IDX_MODIC else 0,
            "up_endplate": predictions[IDX_UP_ENDPLATE] if len(predictions) > IDX_UP_ENDPLATE else 0,
            "low_endplate": predictions[IDX_LOW_ENDPLATE] if len(predictions) > IDX_LOW_ENDPLATE else 0,
            "spondylolisthesis": predictions[IDX_SPONDY] if len(predictions) > IDX_SPONDY else 0,
            "herniation": predictions[IDX_HERN] if len(predictions) > IDX_HERN else 0,
            "narrowing": predictions[IDX_NARROW] if len(predictions) > IDX_NARROW else 0,
            "bulging": predictions[IDX_BULGE] if len(predictions) > IDX_BULGE else 0,
            "pfirrman": predictions[IDX_PFIRRMAN] if len(predictions) > IDX_PFIRRMAN else 0,
            "crop_shape": cropped_mri.shape,
            "disk_voxels": int(np.sum(disk_mask)),
            "inference_backend": "triton"
        }
        
        if logger:
            logger.info(f"Диск {disk_label}: результаты Triton grading = {result}")
        
        return result
        
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при обработке диска {disk_label} через Triton: {e}")
        return {"error": str(e)}


def convert_dicom_to_triton_input(
    studies: List[pydicom.dataset.FileDataset],
    target_shape: tuple = (1, 1, 128, 128, 64)
) -> np.ndarray:
    """
    Преобразует DICOM данные в формат для Triton.
    
    Args:
        studies: Список DICOM datasets
        target_shape: Целевая форма для Triton (batch, channel, H, W, D)
        
    Returns:
        Массив numpy в формате для Triton
    """
    # Здесь должна быть логика преобразования DICOM в numpy массив
    # Пока возвращаем заглушку
    return np.random.rand(*target_shape).astype(np.float32)