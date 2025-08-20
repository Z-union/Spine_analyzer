"""
Рефакторенный модуль для обработки исследований позвоночника.
Разделяет инициализацию моделей, получение данных и обработку исследований.
"""

import os
import time
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

import numpy as np
import torch
import pydicom.dataset
from nibabel.nifti1 import Nifti1Image
from nibabel.loadsave import save

from utils import (
    pad_nd_image, average4d, reorient_canonical, resample, DefaultPreprocessor,
    largest_component, iterative_label, transform_seg2image, extract_alternate,
    fill_canal, crop_image2seg
)
from model import sliding_window_inference, get_sliding_window_slicers
from dicom_io import load_study_dicoms
from utils.constant import (
    LANDMARK_LABELS, VERTEBRA_DESCRIPTIONS, COLORS,
    DEFAULT_CROP_MARGIN, DILATE_SIZE, EXTRACT_LABELS_RANGE,
    SAG_PATCH_SIZE, AX_PATCH_SIZE, CROP_SHAPE,
    IDX_MODIC, IDX_UP_ENDPLATE, IDX_LOW_ENDPLATE, IDX_SPONDY, 
    IDX_HERN, IDX_NARROW, IDX_BULGE, IDX_PFIRRMAN
)


def get_device():
    """Определяет доступное устройство для вычислений."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device('xpu')
    else:
        return torch.device('cpu')


def initialize_models(
    ax_model_path: str = 'model/weights/ax.pth',
    sag_step_1_model_path: str = 'model/weights/sag_step_1.pth',
    sag_step_2_model_path: str = 'model/weights/sag_step_2.pth',
    grading_model_path: str = 'model/weights/grading.pth',
    use_dual_channel_grading: bool = True
) -> Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module, object]:
    """
    Инициализирует все модели для обработки позвоночника.
    
    Args:
        ax_model_path: Путь к модели для аксиальной сегментации
        sag_step_1_model_path: Путь к модели для первого шага сагиттальной сегментации
        sag_step_2_model_path: Путь к модели для второго шага сагиттальной сегментации
        grading_model_path: Путь к модели для grading
        use_dual_channel_grading: Использовать двухканальную модель grading
        
    Returns:
        Кортеж из трех моделей сегментации и grading процессора
    """
    device = get_device()
    
    # Загружаем модели сегментации
    ax_model = torch.load(ax_model_path, map_location=device, weights_only=False)
    sag_step_1_model = torch.load(sag_step_1_model_path, map_location=device, weights_only=False)
    sag_step_2_model = torch.load(sag_step_2_model_path, map_location=device, weights_only=False)
    
    # Создаем grading процессор
    try:
        from grading_processor import SpineGradingProcessor
        grading_processor = SpineGradingProcessor(
            model_path=grading_model_path,
            device=str(device),
            use_dual_channel=use_dual_channel_grading
        )
        print(f"Инициализирован grading процессор (двухканальный: {use_dual_channel_grading})")
    except Exception as e:
        print(f"Ошибка при инициализации grading процессора: {e}")
        print("Используется заглушка")
        grading_processor = None
    
    # Переводим модели сегментации в режим оценки
    ax_model.eval()
    sag_step_1_model.eval()
    sag_step_2_model.eval()
    
    return ax_model, sag_step_1_model, sag_step_2_model, grading_processor


def get_study_from_folder(study_folder: str) -> List[pydicom.dataset.FileDataset]:
    """
    Получает исследование из папки с DICOM файлами.
    
    Args:
        study_folder: Путь к папке с DICOM файлами
        
    Returns:
        Список DICOM datasets
    """
    from dicom_io.io import is_dicom_file
    from pydicom import dcmread
    import glob
    
    dicom_paths = [
        f for f in glob.glob(os.path.join(study_folder, '**', '*'), recursive=True) 
        if os.path.isfile(f) and is_dicom_file(f)
    ]
    
    if not dicom_paths:
        raise ValueError(f"В папке {study_folder} не найдено DICOM-файлов")
    
    dicoms = [dcmread(f, stop_before_pixels=True) for f in dicom_paths]
    return dicoms


def get_study_from_orthanc(
    study_instance_uid: str,
    client
) -> List[pydicom.dataset.FileDataset]:
    """
    Получает исследование из Orthanc по Study Instance UID.
    
    Args:
        study_instance_uid: UID исследования
        client: DICOMwebClient для подключения к Orthanc
        
    Returns:
        Список DICOM datasets
    """
    try:
        dicom_files = client.retrieve_study(study_instance_uid)
        
        if dicom_files:
            print(f"DICOMweb success: Retrieved {len(dicom_files)} instances")
            return dicom_files
        else:
            raise Exception("No instances retrieved via DICOMweb")
            
    except Exception as e:
        print(f"DICOMweb failed: {e}")
        raise


def pick_best_scan(scan_tuple):
    """
    Выбирает лучший доступный скан из кортежа (T1, T2, STIR).
    Приоритет: T1 > T2 > STIR
    """
    if scan_tuple[0] is not None:  # T1
        return scan_tuple[0]
    if scan_tuple[1] is not None:  # T2
        return scan_tuple[1]
    if scan_tuple[2] is not None:  # STIR
        return scan_tuple[2]
    return None


def run_segmentation(
    studies: List[pydicom.dataset.FileDataset],
    ax_model: torch.nn.Module,
    sag_step_1_model: torch.nn.Module,
    sag_step_2_model: torch.nn.Module,
    logger: Optional[logging.Logger] = None
) -> Tuple[Optional[Nifti1Image], Optional[Nifti1Image]]:
    """
    Выполняет сегментацию исследования.
    
    Args:
        studies: Список DICOM datasets
        ax_model: Модель для аксиальной сегментации
        sag_step_1_model: Модель для первого шага сагиттальной сегментации
        sag_step_2_model: Модель для второго шага сагиттальной сегментации
        logger: Logger для записи сообщений
        
    Returns:
        Кортеж (nifti_img, nifti_seg) или (None, None) если сегментация не удалась
    """
    device = get_device()
    
    try:
        # Здесь должна быть логика преобразования studies в структуру для load_study_dicoms
        # Пока используем заглушку - нужно адаптировать под нов��ю структуру
        
        # Загрузка всех доступных проекций и контрастов
        # Временно используем старую логику, но адаптированную
        all_scans, _ = load_study_dicoms("", require_extensions=False)  # Заглушка
        sag_scans, ax_scans, _ = all_scans
        
        sag_scans = average4d(sag_scans)
        ax_scans = average4d(ax_scans)
        
        sag_scans = reorient_canonical(sag_scans)
        ax_scans = reorient_canonical(ax_scans)
        
        sag_scans = resample(sag_scans)
        ax_scans = resample(ax_scans)
        
        sag_scan = pick_best_scan(sag_scans)
        ax_scan = pick_best_scan(ax_scans)
        
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
            
            # save(sag_scan, 'sagittal_original.nii.gz')
            nifti_img = Nifti1Image(sag_scan.get_fdata()[np.newaxis, ...], sag_scan.affine, sag_scan.header)
            
            # Двухэтапная сегментация
            for step, model in [('step_1', sag_step_1_model), ('step_2', sag_step_2_model)]:
                data, seg, properties = preprocessor.run_case(nifti_img, transpose_forward=[0, 1, 2])
                img, slicer_revert_padding = pad_nd_image(data, SAG_PATCH_SIZE, 'constant', {"constant_values": 0}, True)
                slicers = get_sliding_window_slicers(img.shape[1:])
                
                num_segmentation_heads = 9 if step == 'step_1' else 11
                predicted_logits = sliding_window_inference(
                    data=img,
                    slicers=slicers,
                    model_or_triton=model,  # локальная модель PyTorch
                    patch_size=SAG_PATCH_SIZE,
                    num_heads=num_segmentation_heads,
                    batch_size=4,  # размер батча на скользящее окно
                    device=device,  # torch.device("cuda") или "cpu"
                    use_gaussian=True,  # оставляем Gaussian fusion
                    mode="3d"  # "2d", если у тебя 2D данные
                )
                
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
                predicted_logits = predicted_logits.detach().cpu().numpy()
                segmentation_reverted_cropping, predicted_probabilities = preprocessor.convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits)
                
                if not isinstance(segmentation_reverted_cropping, Nifti1Image):
                    seg_nifti = Nifti1Image(segmentation_reverted_cropping.astype(np.uint8), sag_scan.affine, sag_scan.header)
                else:
                    seg_nifti = segmentation_reverted_cropping
                
                nifti_seg = largest_component(seg_nifti, binarize=True, dilate=DILATE_SIZE)
                
                # Итеративная разметка
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
                nifti_img = pick_best_scan(sagittals)
                nifti_seg = transform_seg2image(nifti_img, nifti_seg)
                
                if step == 'step_1':
                    nifti_seg = extract_alternate(nifti_seg, labels=EXTRACT_LABELS_RANGE)
                    img_data = np.asanyarray(nifti_img.dataobj)
                    seg_data = np.asanyarray(nifti_seg.dataobj)
                    assert img_data.shape == seg_data.shape, f"Shapes do not match: {img_data.shape} vs {seg_data.shape}"
                    multi_channel = np.stack([img_data, seg_data], axis=0)
                    nifti_img = Nifti1Image(multi_channel, nifti_img.affine, nifti_img.header)
        
        # Сегментация аксиала, если есть
        if ax_scan is not None:
            if logger:
                logger.info("Выбран аксиальный скан для сегментации.")
            
            # save(ax_scan, 'axial_original.nii.gz')
            axial = Nifti1Image(ax_scan.get_fdata()[np.newaxis, ...], ax_scan.affine, ax_scan.header)
            
            data, seg, properties = preprocessor.run_case(axial, transpose_forward=[0, 1, 2])
            img, slicer_revert_padding = pad_nd_image(data, AX_PATCH_SIZE, 'constant', {"constant_values": 0}, True)
            slicers = get_sliding_window_slicers(img.shape[1:], patch_size=AX_PATCH_SIZE)

            predicted_logits = sliding_window_inference(
                data=img,
                slicers=slicers,
                model_or_triton=ax_model,  # твоя локальная PyTorch модель
                patch_size=AX_PATCH_SIZE,
                num_heads=5,
                batch_size=4,  # размер батча на скользящее окно
                device=device,  # torch.device("cuda") или "cpu"
                use_gaussian=False,  # выключаем Gaussian fusion
                mode="2d"  # 2D режим
            )
            
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
            predicted_logits = predicted_logits.detach().cpu().numpy()
            segmentation_reverted_cropping, predicted_probabilities = preprocessor.convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits)
            
            if not isinstance(segmentation_reverted_cropping, Nifti1Image):
                seg_nifti = Nifti1Image(segmentation_reverted_cropping.astype(np.uint8), ax_scan.affine, ax_scan.header)
            else:
                seg_nifti = segmentation_reverted_cropping
            
            processed_axial_data = np.asanyarray(data)
            processed_axial = Nifti1Image(processed_axial_data, ax_scan.affine, ax_scan.header)
            ax_seg_nifti = Nifti1Image(np.asanyarray(seg_nifti.dataobj), ax_scan.affine, ax_scan.header)
            
            # save(processed_axial, 'axial_processed.nii.gz')
            # save(ax_seg_nifti, 'axial_segmentation.nii.gz')
            #
            # if logger:
            #     logger.info("Обработанные аксиальные файлы сохранены: axial_processed.nii.gz, axial_segmentation.nii.gz")
            
            processed_axial = Nifti1Image(np.squeeze(processed_axial.get_fdata()), processed_axial.affine, processed_axial.header)
            ax_seg_nifti = transform_seg2image(processed_axial, ax_seg_nifti)
            processed_axial = crop_image2seg(processed_axial, ax_seg_nifti, margin=DEFAULT_CROP_MARGIN)
            ax_seg_nifti = transform_seg2image(processed_axial, ax_seg_nifti)
        
        # Возвращаем результаты (приоритет: сагиттал -> аксиал)
        if nifti_img is not None and nifti_seg is not None:
            return sagittals, nifti_seg
        elif ax_scan is not None:
            return processed_axial, ax_seg_nifti
        else:
            return None, None
            
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при сегментации: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return None, None


def process_study(
    studies: List[pydicom.dataset.FileDataset],
    ax_model: torch.nn.Module,
    sag_step_1_model: torch.nn.Module,
    sag_step_2_model: torch.nn.Module,
    grading_processor: object,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Обрабатывает исследование позвоночника.
    
    Args:
        studies: Список DICOM datasets
        ax_model: Модель для аксиальной сегментации
        sag_step_1_model: Модель для первого шага сагиттальной сегментации
        sag_step_2_model: Модель для второго шага сагиттальной сегментации
        grading_processor: Процессор для grading анализа
        logger: Logger для записи сообщений
        
    Returns:
        Слова��ь с результатами обработки
    """
    start_time = time.time()
    
    try:
        if logger:
            logger.info("Начинаем обработку исследования")
            logger.info(f"Количество DICOM файлов: {len(studies)}")
        
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
        if hasattr(nifti_img, 'get_fdata'):
            mri_data = nifti_img.get_fdata()
        else:
            mri_data = np.asarray(nifti_img)
        
        mask_data = nifti_seg.get_fdata()
        
        # Обрабатываем диски с помощью grading процессора
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
        
        elapsed_time = time.time() - start_time
        
        if logger:
            logger.info(f"Обработка завершена за {elapsed_time:.2f} секунд")
        
        return {
            "processing_time": elapsed_time,
            "processed_disks": len(disk_results),
            "disk_results": disk_results,
            "grading_summary": grading_summary,
            "segmentation_shape": seg_data.shape,
            "unique_labels": unique_labels.tolist(),
            "present_disks": present_disks
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Критическая ошибка при обработке исследования: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}


def process_single_disk(
    mri_data: np.ndarray,
    mask_data: np.ndarray,
    disk_label: int,
    grading_model: torch.nn.Module,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Обрабатывает отдельный диск для grading с поддержкой двухканального входа.
    
    Args:
        mri_data: Данные МРТ изображения
        mask_data: Данные сегментации
        disk_label: Метка диска
        grading_model: Модель для grading
        logger: Logger для записи сообщений
        
    Returns:
        Словарь с результатами grading для диска
    """
    try:
        device = get_device()
        
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
        cropped_mask = mask_data[tuple(crop_slices)]
        
        # Нормализация изображения
        mean = cropped_mri.mean()
        std = cropped_mri.std() if cropped_mri.std() > 0 else 1.0
        normalized_mri = (cropped_mri - mean) / std
        
        # Нормализация маски диска (приведение к 0-1)
        disk_mask_cropped = (cropped_mask == disk_label).astype(np.float32)
        if disk_mask_cropped.max() > 1:
            disk_mask_cropped = disk_mask_cropped / disk_mask_cropped.max()
        
        # Проверяем, поддерживает ли модель двухканальный вход
        try:
            # Пытаемся определить количество входных каналов модели
            if hasattr(grading_model, 'conv1'):
                input_channels = grading_model.conv1.in_channels
            else:
                input_channels = 1  # По умолчанию одноканальная
        except:
            input_channels = 1
        
        # Подготавливаем тензор для модели
        if input_channels == 2:
            # Двухканальный вход: [изображение, маска диска]
            dual_channel = np.stack([normalized_mri, disk_mask_cropped], axis=0)
            img_tensor = torch.tensor(dual_channel).unsqueeze(0).float().to(device)
            if logger:
                logger.debug(f"Используется двухканальный вход для диска {disk_label}")
        else:
            # Одноканальный вход: только изображение
            img_tensor = torch.tensor(normalized_mri).unsqueeze(0).unsqueeze(0).float().to(device)
            if logger:
                logger.debug(f"Используется одноканальный вход для диска {disk_label}")
        
        # Предсказание
        with torch.no_grad():
            grading_outputs = grading_model(img_tensor)
        
        # Обрабатываем результаты
        predictions = [torch.argmax(output).detach().cpu().numpy().item() for output in grading_outputs]
        
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
            "input_channels": input_channels,
            "model_type": "dual_channel" if input_channels == 2 else "single_channel"
        }
        
        if logger:
            logger.info(f"Диск {disk_label}: результаты grading = {result}")
        
        return result
        
    except Exception as e:
        if logger:
            logger.error(f"Ошибка при обработке диска {disk_label}: {e}")
        return {"error": str(e)}


def main():
    """
    Пример использования рефакторенного кода.
    """
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Инициализация моделей (один раз)
        logger.info("Инициализация моделей...")
        models = initialize_models()
        ax_model, sag_step_1_model, sag_step_2_model, grading_model = models
        logger.info("Модели загружены успешно")
        
        # Получение исследования
        study_folder = "ST000000"  # Пример папки
        logger.info(f"Загрузка исследования из {study_folder}")
        studies = get_study_from_folder(study_folder)
        logger.info(f"Загружено {len(studies)} DICOM файлов")
        
        # Обработка исслед��вания
        logger.info("Начинаем обработку исследования...")
        results = process_study(studies, ax_model, sag_step_1_model, sag_step_2_model, grading_model, logger)
        
        # Вывод результатов
        logger.info("Результаты обработки:")
        logger.info(f"Время обработки: {results.get('processing_time', 'N/A')} секунд")
        logger.info(f"Обработано дисков: {results.get('processed_disks', 'N/A')}")
        
        if 'disk_results' in results:
            for disk_label, disk_result in results['disk_results'].items():
                disk_name = VERTEBRA_DESCRIPTIONS.get(disk_label, f"Disk_{disk_label}")
                logger.info(f"{disk_name}: {disk_result}")
        
    except Exception as e:
        logger.error(f"Ошибка в main: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()