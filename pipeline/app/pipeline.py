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

from .orthanc_client import fetch_study_temp
from .config import settings

from . import tools
from . import iterative_label
from . import grading_processor
from .preprocessor import DefaultPreprocessor, largest_component
from .dicom_io import load_study_dicoms
from .average4d import average4d
from .reorient_canonical import reorient_canonical
from .resample import resample
from .predictor import sliding_window_inference
from .pathology_measurements import PathologyMeasurements
from .dicom_reports import send_reports_to_orthanc

# Используем единый логгер из main
logger = logging.getLogger("dicom-pipeline")


def save_segmentation_overlay_multiclass(
    image_nifti: Nifti1Image,
    seg_nifti: Nifti1Image,
    output_dir: Optional[str] = None,
    slice_axis: int = 0,
    thickness: int = 1,
    colormap: dict = None
):
    """
    Save image slices with multiclass segmentation overlay as contours.
    
    Parameters
    ----------
    image_nifti : nib.Nifti1Image
        NIfTI image
    seg_nifti : nib.Nifti1Image
        NIfTI segmentation with integer class labels
    output_dir : str, optional
        Directory to save PNG slices (if None, only returns images)
    slice_axis : int, optional
        Slice axis (default 0)
    thickness : int, optional
        Contour thickness
    colormap : dict, optional
        Dictionary {label: (B, G, R)} for colors. If None - random colors generated
        
    Returns
    -------
    list
        List of overlay images as numpy arrays
    """
    results = []
    
    try:
        image_data = np.asanyarray(image_nifti.dataobj)
        seg_data = np.asanyarray(seg_nifti.dataobj).astype(np.uint8)
        
        num_slices = image_data.shape[slice_axis]
        logger.debug(f"Creating segmentation overlays for {num_slices} slices")
        
        # Generate colors for classes
        labels = np.unique(seg_data)
        labels = labels[labels != 0]  # exclude background
        if colormap is None:
            np.random.seed(42)
            colormap = {label: tuple(np.random.randint(0, 256, size=3).tolist()) for label in labels}
        
        logger.debug(f"Found {len(labels)} unique labels for overlay: {labels.tolist()}")
        
        # Create output directory if specified
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            logger.debug(f"Saving overlays to directory: {output_dir}")
        
        for i in range(num_slices):
            # Select slice
            if slice_axis == 0:
                img_slice = image_data[i, :, :]
                seg_slice = seg_data[i, :, :]
            elif slice_axis == 1:
                img_slice = image_data[:, i, :]
                seg_slice = seg_data[:, i, :]
            else:
                img_slice = image_data[:, :, i]
                seg_slice = seg_data[:, :, i]
            
            # Skip empty slices
            if np.sum(seg_slice) == 0:
                continue
            
            # Normalize image for display
            img_norm = ((img_slice - np.min(img_slice)) / (np.ptp(img_slice) + 1e-8) * 255).astype(np.uint8)
            img_color = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
            
            # Draw contours for each class
            for label, color in colormap.items():
                mask = (seg_slice == label).astype(np.uint8)
                if np.sum(mask) == 0:
                    continue
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img_color, contours, -1, color, thickness)
            
            results.append(img_color)
            
            # Save slice if output directory specified
            if output_dir is not None:
                output_path = os.path.join(output_dir, f"slice_{i:03d}.png")
                cv2.imwrite(output_path, img_color)
        
        logger.info(f"Created {len(results)} segmentation overlay images")
        return results
        
    except Exception as e:
        logger.error(f"Failed to create segmentation overlays: {str(e)}")
        return []


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
    Process study from path (archive or directory).
    
    Args:
        path: Path to downloaded archive/folder with study
        study_id: Optional study ID for logging
        client: Triton inference client
        
    Returns:
        Dictionary with processing results
    """
    logger.info(f"Starting processing for study_id={study_id}, path={path}")
    logger.debug(f"Path exists: {os.path.exists(path)}, is_file: {os.path.isfile(path)}, size: {os.path.getsize(path) if os.path.exists(path) else 'N/A'}")

    study_dir = None
    temp_dir = None
    try:
        is_file = os.path.isfile(path)
        size = os.path.getsize(path) if is_file else None
        logger.debug(f"Input type: {'file' if is_file else 'directory'}, size: {size} bytes")
        
        # Step 1: Prepare study directory
        logger.info("Step 1: Preparing study directory")
        try:
            study_dir, temp_dir = _prepare_study_dir(path)
            logger.info(f"Study directory prepared: {study_dir}, temp_dir: {temp_dir}")
        except Exception as e:
            logger.error(f"Failed to prepare study directory: {str(e)}")
            raise
        
        # Step 2: Load DICOM files
        logger.info("Step 2: Loading DICOM files")
        try:
            scans, correspondence = load_study_dicoms(study_dir)
            sag_scans, ax_scans, cor_scans = scans
            logger.info(f"DICOM files loaded - sagittal: {len([s for s in sag_scans if s is not None])}, "
                       f"axial: {len([s for s in ax_scans if s is not None])}, "
                       f"coronal: {len([s for s in cor_scans if s is not None])}, "
                       f"correspondence pairs: {len(correspondence)}")
        except Exception as e:
            logger.error(f"Failed to load DICOM files: {str(e)}")
            raise

        # Step 3: Preprocessing - Average 4D
        logger.info("Step 3: Averaging 4D scans")
        try:
            sag_scans = average4d(sag_scans)
            ax_scans = average4d(ax_scans)
            logger.debug("4D averaging completed")
        except Exception as e:
            logger.error(f"Failed during 4D averaging: {str(e)}")
            raise

        # Step 4: Reorient to canonical
        logger.info("Step 4: Reorienting to canonical orientation")
        try:
            sag_scans = reorient_canonical(sag_scans)
            ax_scans = reorient_canonical(ax_scans)
            logger.debug("Reorientation completed")
        except Exception as e:
            logger.error(f"Failed during reorientation: {str(e)}")
            raise

        # Step 5: Resample
        logger.info("Step 5: Resampling scans")
        try:
            sag_scans = resample(sag_scans)
            ax_scans = resample(ax_scans)
            logger.debug("Resampling completed")
        except Exception as e:
            logger.error(f"Failed during resampling: {str(e)}")
            raise

        # Step 6: Select best scan
        logger.info("Step 6: Selecting best scan for segmentation")
        sag_scan = _pick_best_scan(sag_scans)
        ax_scan = _pick_best_scan(ax_scans)
        
        logger.info(f"Best scan selection - sagittal: {'found' if sag_scan is not None else 'not found'}, "
                   f"axial: {'found' if ax_scan is not None else 'not found'}")

        if sag_scan is None and ax_scan is None:
            logger.error("No sagittal or axial scans available for segmentation")
            return {"error": "No suitable scans found for segmentation"}

        preprocessor = DefaultPreprocessor()
        nifti_img = None
        nifti_seg = None

        # Step 7: Segmentation
        if sag_scan is not None:
            logger.info("Step 7: Starting sagittal scan segmentation")
            
            try:
                nifti_img = Nifti1Image(sag_scan.get_fdata()[np.newaxis, ...], sag_scan.affine, sag_scan.header)
                logger.debug(f"Created NIfTI image with shape: {nifti_img.shape}")
            except Exception as e:
                logger.error(f"Failed to create NIfTI image: {str(e)}")
                raise

            # Two-stage segmentation
            for step_num, (step, model) in enumerate([('step_1', 'seg_sag_stage_1'), ('step_2', 'seg_sag_stage_2')], 1):
                logger.info(f"Step 7.{step_num}: Running segmentation {step} with model {model}")
                
                try:
                    data, seg, properties = preprocessor.run_case(nifti_img, transpose_forward=[0, 1, 2])
                    logger.debug(f"Preprocessor output shape: {data.shape}")
                except Exception as e:
                    logger.error(f"Preprocessor failed at {step}: {str(e)}")
                    raise
                
                try:
                    img, slicer_revert_padding = tools.pad_nd_image(data, settings.SAG_PATCH_SIZE, 'constant', 
                                                                   {"constant_values": 0}, True)
                    slicers = tools.get_sliding_window_slicers(img.shape[1:])
                    num_segmentation_heads = 9 if step == 'step_1' else 11
                    logger.debug(f"Padded image shape: {img.shape}, num_heads: {num_segmentation_heads}")
                except Exception as e:
                    logger.error(f"Failed to prepare image for inference at {step}: {str(e)}")
                    raise
                
                logger.info(f"Running sliding window inference for {step}")
                try:
                    # Use model-specific batch sizes from environment variables
                    if model == 'seg_sag_stage_1':
                        batch_size = settings.SEG_SAG_STAGE_1_BATCH_SIZE
                    elif model == 'seg_sag_stage_2':
                        batch_size = settings.SEG_SAG_STAGE_2_BATCH_SIZE
                    else:
                        batch_size = settings.INFERENCE_BATCH_SIZE
                    
                    logger.debug(f"Using batch_size={batch_size} for model {model}")
                    
                    predicted_logits = sliding_window_inference(
                        data=img,
                        slicers=slicers,
                        patch_size=settings.SAG_PATCH_SIZE,
                        num_heads=num_segmentation_heads,
                        batch_size=batch_size,
                        use_gaussian=True,
                        mode="3d",
                        triton_client=client,
                        triton_model_name=model,
                        triton_input_name='input',
                        triton_output_name='output',
                    )
                    logger.debug(f"Inference completed for {step}, output shape: {predicted_logits.shape}")
                except Exception as e:
                    logger.error(f"Sliding window inference failed at {step}: {str(e)}")
                    raise
                
                try:
                    predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
                    segmentation_reverted_cropping, predicted_probabilities = preprocessor.convert_predicted_logits_to_segmentation_with_correct_shape(
                        predicted_logits)
                    logger.debug(f"Segmentation post-processing completed for {step}")
                except Exception as e:
                    logger.error(f"Post-processing failed at {step}: {str(e)}")
                    raise

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
            logger.info("Starting pathology measurements")

            try:
                pathology_measurements = measure_all_pathologies(
                    img_data, seg_data, disk_results)

                logger.info(f"Pathology measurements completed. Measured disks: {len(pathology_measurements)}")

                # Output brief statistics on measurements
                herniation_count = 0
                spondylolisthesis_count = 0

                for disk_label, measurements in pathology_measurements.items():
                    if measurements.get('Disc herniation', {}).get('detected', False):
                        herniation_count += 1
                        volume = measurements['Disc herniation']['volume_mm3']
                        protrusion = measurements['Disc herniation']['max_protrusion_mm']
                        level_name = measurements.get('level_name', f'Disk_{disk_label}')
                        logger.info(f"Herniation at {level_name}: volume={volume:.1f}mm3, protrusion={protrusion:.1f}mm")

                    if measurements.get('Spondylolisthesis', {}).get('detected', False):
                        spondylolisthesis_count += 1
                        displacement = measurements['Spondylolisthesis']['displacement_mm']
                        percentage = measurements['Spondylolisthesis']['displacement_percentage']
                        grade = measurements['Spondylolisthesis']['grade']
                        level_name = measurements.get('level_name', f'Disk_{disk_label}')
                        logger.info(
                            f"Spondylolisthesis at {level_name}: displacement={displacement:.1f}mm ({percentage:.1f}%), grade={grade}")

                logger.info(f"Total pathologies found: herniations={herniation_count}, spondylolisthesis={spondylolisthesis_count}")

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
            "segmentations": segmentations,
            "disk_results": disk_results,
            "pathology_measurements": pathology_measurements
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
      3) Отправить SR и SC отчеты обратно в Orthanc
      4) Корректно удалить временные файлы/папки независимо от исхода

    Возвращает dict, готовый к отдаче наружу либо к публикации.
    """
    tmp_path: Optional[str] = None
    try:
        client = grpcclient.InferenceServerClient(url=settings.TRITON_URL, verbose=False)
        tmp_path = fetch_study_temp(study_id)
        logger.info(f"Study {study_id} downloaded to temp: {tmp_path}")
        # Обработка исследования
        result = process_study_path(tmp_path, study_id=study_id, client=client)
        
        # Отправка отчетов в Orthanc
        report_upload_result = {"sr_uploaded": False, "sc_uploaded": False}
        if result:
            # Извлекаем данные для отчетов напрямую из результата
            grading_results = result.get("disk_results", {})
            pathology_measurements = result.get("pathology_measurements", {})
            segmentation_images = result.get("segmentations", [])
            
            logger.info(f"segmentation_images: {segmentation_images}")
            # Отправляем отчеты в Orthanc
            try:
                report_upload_result = send_reports_to_orthanc(
                    study_id=study_id,
                    grading_results=grading_results,
                    pathology_measurements=pathology_measurements,
                    segmentation_images=segmentation_images
                )
                logger.info(f"Reports uploaded to Orthanc for study {study_id}: {report_upload_result}")
            except Exception as e:
                logger.error(f"Failed to upload reports to Orthanc: {e}")
                report_upload_result["errors"] = [str(e)]
        
        # TODO: Отправка уведомления о завершении обработки через Kafka
        # Здесь должен быть код для публикации сообщения в Kafka топик
        # с информацией о завершении обработки и результатах
        # Например:
        # kafka_producer.send(
        #     topic='spine-analysis-completed',
        #     value={
        #         'study_id': study_id,
        #         'status': 'completed',
        #         'sr_uploaded': report_upload_result.get('sr_uploaded', False),
        #         'sc_uploaded': report_upload_result.get('sc_uploaded', False),
        #         'sc_count': report_upload_result.get('sc_count', 0),
        #         'timestamp': datetime.now().isoformat()
        #     }
        # )
        
        return {
            "status": "ok",
            "study_id": study_id,
            "pipeline_result": result,
            "report_upload": report_upload_result
        }, segmentation_images
    
    except Exception:
        logger.exception(f"Pipeline failed for study {study_id}")
        
        # TODO: Отправка уведомления об ошибке через Kafka
        # kafka_producer.send(
        #     topic='spine-analysis-failed',
        #     value={
        #         'study_id': study_id,
        #         'status': 'failed',
        #         'error': str(e),
        #         'timestamp': datetime.now().isoformat()
        #     }
        # )
        
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