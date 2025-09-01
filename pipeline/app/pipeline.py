import logging
import os
import shutil
import zipfile
import tarfile
import tempfile
from typing import Any, Dict, Optional, Tuple, List
from nibabel.nifti1 import Nifti1Image
import numpy as np
import tritonclient.grpc as grpcclient
import pandas as pd
import cv2

from orthanc_client import fetch_study_temp
from config import settings

import tools
import iterative_label
import grading_processor
from preprocessor import DefaultPreprocessor, largest_component
from dicom_io import load_study_dicoms
from average4d import average4d
from reorient_canonical import reorient_canonical
from resample import resample
from predictor import sliding_window_inference
from pathology_measurements import PathologyMeasurements

logger = logging.getLogger(__name__)


def save_segmentation_overlay_multiclass(
    image_nifti: Nifti1Image,
    seg_nifti: Nifti1Image,
    output_dir: Optional[str] = None,
    slice_axis: int = 0,
    thickness: int = 1,
    colormap: dict = None
):
    """
    Сохраняет срезы изображения с наложенной мультиклассовой сегментацией в виде контуров.

    Parameters
    ----------
    image_nifti : nib.Nifti1Image
        NIfTI изображение.
    seg_nifti : nib.Nifti1Image
        NIfTI сегментация с целыми метками классов.
    output_dir : str
        Папка для сохранения PNG срезов.
    slice_axis : int, optional
        Ось срезов (по умолчанию 0).
    thickness : int, optional
        Толщина контура.
    colormap : dict, optional
        Словарь {label: (B, G, R)} для цветов. Если None — генерируются случайные цвета.
    """
    if output_dir is None:
        os.makedirs(output_dir, exist_ok=True)

    results = list()

    image_data = np.asanyarray(image_nifti.dataobj)
    seg_data = np.asanyarray(seg_nifti.dataobj).astype(np.uint8)

    num_slices = image_data.shape[slice_axis]

    # Автоматическая генерация цветов для классов
    labels = np.unique(seg_data)
    labels = labels[labels != 0]  # исключаем фон
    if colormap is None:
        np.random.seed(42)
        colormap = {label: tuple(np.random.randint(0, 256, size=3).tolist()) for label in labels}

    for i in range(num_slices):
        # Выбираем срез
        if slice_axis == 0:
            img_slice = image_data[i, :, :]
            seg_slice = seg_data[i, :, :]
        elif slice_axis == 1:
            img_slice = image_data[:, i, :]
            seg_slice = seg_data[:, i, :]
        else:
            img_slice = image_data[:, :, i]
            seg_slice = seg_data[:, :, i]

        # Нормализуем изображение для отображения
        img_norm = ((img_slice - np.min(img_slice)) / (np.ptp(img_slice) + 1e-8) * 255).astype(np.uint8)
        img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

        # Рисуем контуры для каждого класса
        for label, color in colormap.items():
            mask = (seg_slice == label).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_color, contours, -1, color, thickness)
            results.append(img_color)
        # Сохраняем срез
        if output_dir is None:
            cv2.imwrite(os.path.join(output_dir, f"slice_{i:03d}.png"), img_color)

        return results


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


def _prepare_study_dir(path: str) -> Tuple[str, Optional[str]]:
    """
    Normalize input path to a directory containing DICOM files.
    Returns (study_dir, temp_dir_to_cleanup). If a temp dir is created for extraction,
    it is returned as both study_dir and temp_dir_to_cleanup.
    """
    if os.path.isdir(path):
        return path, None

    if os.path.isfile(path):
        # ZIP produced by Orthanc
        if zipfile.is_zipfile(path):
            tmpdir = tempfile.mkdtemp(prefix="orthanc_study_")
            with zipfile.ZipFile(path) as zf:
                zf.extractall(tmpdir)
            return tmpdir, tmpdir
        # tar.gz or tgz
        if path.lower().endswith((".tar.gz", ".tgz")):
            tmpdir = tempfile.mkdtemp(prefix="orthanc_study_")
            with tarfile.open(path, "r:gz") as tf:
                tf.extractall(tmpdir)
            return tmpdir, tmpdir
        # Fallback: a single file path (e.g., a single .dcm) -> use its parent dir
        parent = os.path.dirname(path)
        if parent:
            return parent, None

    raise ValueError(f"Unsupported input for study path: {path}. Provide a directory or a ZIP/TAR.GZ archive.")

def _pick_best_scan(scan_tuple):
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


def process_study_path(path: str, client: grpcclient.InferenceServerClient, study_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Здесь должна быть твоя доменная обработка исследования.

    Вход:
      - path: путь к скачанному архиву/папке с исследованием (временный ресурс)
      - study_id: опционально для удобства логирования

    Что делать внутри (примерный план):
      - распаковать архив (если это архив) во временную папку
      - найти нужные DICOM-файлы
      - выполнить предобработку/инференс/постобработку
      - результаты сохранить (БД/S3/файлы/публикация в Kafka и т.п.)

    Важно: очистку временных ресурсов НЕ делать тут — это делает обёртка
    run_pipeline_for_study(). Эта функция должна только обрабатывать и
    возвращать результат (dict), пригодный для ответа REST или логирования в Kafka.
    """
    logger.info(f"Processing study {study_id or ''} from path: {path}")

    study_dir = None
    temp_dir = None
    try:
        is_file = os.path.isfile(path)
        size = os.path.getsize(path) if is_file else None
        study_dir, temp_dir = _prepare_study_dir(path)
        scans, correspondence = load_study_dicoms(study_dir)

        sag_scans, ax_scans, _ = scans

        sag_scans = average4d(sag_scans)
        ax_scans = average4d(ax_scans)

        sag_scans = reorient_canonical(sag_scans)
        ax_scans = reorient_canonical(ax_scans)

        sag_scans = resample(sag_scans)
        ax_scans = resample(ax_scans)

        sag_scan = _pick_best_scan(sag_scans)
        ax_scan = _pick_best_scan(ax_scans)

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
            for step, model in [('step_1', 'seg_sag_stage_1'), ('step_2', 'seg_sag_stage_2')]:
                data, seg, properties = preprocessor.run_case(nifti_img, transpose_forward=[0, 1, 2])
                img, slicer_revert_padding = tools.pad_nd_image(data, settings.SAG_PATCH_SIZE, 'constant', {"constant_values": 0},
                                                          True)
                slicers = tools.get_sliding_window_slicers(img.shape[1:])
                num_segmentation_heads = 9 if step == 'step_1' else 11
                predicted_logits = sliding_window_inference(
                    data=img,
                    slicers=slicers,
                    patch_size=settings.SAG_PATCH_SIZE,
                    num_heads=num_segmentation_heads,
                    batch_size=4,
                    use_gaussian=True,
                    mode="3d",
                    triton_client=client,
                    triton_model_name=model,
                    triton_input_name='input',
                    triton_output_name='output',
                )
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
                segmentation_reverted_cropping, predicted_probabilities = preprocessor.convert_predicted_logits_to_segmentation_with_correct_shape(
                    predicted_logits)

                if not isinstance(segmentation_reverted_cropping, Nifti1Image):
                    seg_nifti = Nifti1Image(segmentation_reverted_cropping.astype(np.uint8), sag_scan.affine,
                                            sag_scan.header)
                else:
                    seg_nifti = segmentation_reverted_cropping

                nifti_seg = largest_component(seg_nifti, binarize=True, dilate=settings.DILATE_SIZE)

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

                nifti_seg = iterative_label.iterative_label(**iterative_label_params)
                nifti_seg = iterative_label.fill_canal(nifti_seg, canal_label=2, cord_label=1)
                nifti_seg = iterative_label.transform_seg2image(sag_scan, nifti_seg)

                sagittals = [iterative_label.crop_image2seg(sag, nifti_seg, margin=10) if sag is not None else None for
                             sag in sag_scans]
                nifti_img = _pick_best_scan(sagittals)
                nifti_seg = iterative_label.transform_seg2image(nifti_img, nifti_seg)


                if step == 'step_1':
                    nifti_seg = iterative_label.extract_alternate(nifti_seg, labels=settings.EXTRACT_LABELS_RANGE)
                    img_data = np.asanyarray(nifti_img.dataobj)
                    seg_data = np.asanyarray(nifti_seg.dataobj)
                    assert img_data.shape == seg_data.shape, f"Shapes do not match: {img_data.shape} vs {seg_data.shape}"
                    multi_channel = np.stack([img_data, seg_data], axis=0)
                    nifti_img = Nifti1Image(multi_channel, nifti_img.affine, nifti_img.header)

        nifti_seg = iterative_label.transform_seg2image(_pick_best_scan(sag_scans), nifti_seg)
        unique_labels = np.unique(nifti_seg.get_fdata())
        present_disks = [label for label in settings.LANDMARK_LABELS if label in unique_labels]

        img_data = [img.get_fdata() if hasattr(img, 'get_fdata') else None for img in sag_scans]
        img_data = img_data[:-1]
        example = next(arr for arr in img_data if arr is not None)
        img_data = [arr if arr is not None else np.zeros_like(example) for arr in img_data]
        seg_data = np.asanyarray(nifti_seg.dataobj)

        processor = grading_processor.SpineGradingProcessor(client=client)

        disk_results = processor.process_disks(img_data, seg_data, present_disks)
        grading_summary = processor.create_summary(disk_results)

        pathology_measurements = {}
        if disk_results and 'error' not in grading_summary:
            if logger:
                logger.info("Запускаем измерения патологий...")

            try:
                pathology_measurements = measure_all_pathologies(
                    img_data, seg_data, disk_results)

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
                            logger.info(
                                f"Листез {level_name}: смещение={displacement:.1f}мм ({percentage:.1f}%), степень={grade}")

                    logger.info(f"Итого: грыж={herniation_count}, листезов={spondylolisthesis_count}")

                rows = []

                for disk_label, result in disk_results.items():
                    row = {'disk_label': disk_label}

                    # Добавляем категории (Modic, Pfirrmann, ...)
                    if 'predictions' in result:
                        for category, value in result['predictions'].items():
                            row[category] = value

                    # Добавляем измерения патологий
                    measurements = pathology_measurements.get(disk_label, {})

                    # Грыжа
                    hernia = measurements.get('Disc herniation', {})
                    row['hernia_detected'] = hernia.get('detected', False)
                    row['hernia_volume_mm3'] = hernia.get('volume_mm3', None)
                    row['hernia_max_protrusion_mm'] = hernia.get('max_protrusion_mm', None)

                    # Листез
                    spondy = measurements.get('Spondylolisthesis', {})
                    row['spondy_detected'] = spondy.get('detected', False)
                    row['spondy_displacement_mm'] = spondy.get('displacement_mm', None)
                    row['spondy_displacement_percentage'] = spondy.get('displacement_percentage', None)
                    row['spondy_grade'] = spondy.get('grade', None)

                    rows.append(row)

                df = pd.DataFrame(rows)

            except Exception as e:
                if logger:
                    logger.error(f"Ошибка при измерении патологий: {e}")
                pathology_measurements = {"error": str(e)}


        present_counts = {
            "sagittal": sum(s is not None for s in scans[0]),
            "axial": sum(s is not None for s in scans[1]),
            "coronal": sum(s is not None for s in scans[2]),
        }

        segmentations = save_segmentation_overlay_multiclass(_pick_best_scan(sag_scans), nifti_seg)
        return {
            "input_type": "file" if is_file else "dir",
            "archive_extracted_to": study_dir if temp_dir else None,
            "size_bytes": size,
            "series_detected": present_counts,
            "correspondence_pairs": len(correspondence),
            "results": df.to_json(orient="records", indent=2),
            "segmentations": segmentations
        }
    finally:
        # Clean up only the extraction dir we created here (not the downloaded tmp archive)
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def run_pipeline_for_study(study_id: str) -> Dict[str, Any]:
    """
    Оркестрация пайплайна для одного исследования:
      1) Скачать исследование из Orthanc во временный ресурс
      2) Передать путь в доменную обработку process_study_path()
      3) Корректно удалить временные файлы/папки независимо от исхода

    Возвращает dict, готовый к отдаче наружу либо к публикации.
    """
    tmp_path: Optional[str] = None
    try:
        client = grpcclient.InferenceServerClient(url=settings.TRITON_URL, verbose=False)
        tmp_path = fetch_study_temp(study_id)
        logger.info(f"Study {study_id} downloaded to temp: {tmp_path}")

        result = process_study_path(tmp_path, study_id=study_id, client=client)
        return {
            "status": "ok",
            "study_id": study_id,
            "pipeline_result": result,
        }
    except Exception:
        logger.exception(f"Pipeline failed for study {study_id}")
        # Пробрасываем исключение, чтобы HTTP вернул 500, а Kafka увидела ошибку в логах
        raise
    finally:
        if tmp_path:
            try:
                if os.path.isdir(tmp_path):
                    shutil.rmtree(tmp_path, ignore_errors=True)
                elif os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                logger.info(f"Cleaned up temp path for study {study_id}: {tmp_path}")
            except Exception as cleanup_err:
                logger.warning(f"Failed to cleanup temp path {tmp_path}: {cleanup_err}")

if __name__ == "__main__":
    process_study_path(r'F:\WorkSpace\позвонки\abaev\ST000000.zip', client=grpcclient.InferenceServerClient(url=settings.TRITON_URL, verbose=False))