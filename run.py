import argparse
import textwrap
from pathlib import Path

import nibabel
import numpy as np
import torch
from nibabel.nifti1 import Nifti1Image
from nibabel.loadsave import save
from scipy.ndimage import affine_transform

from utils import pad_nd_image, average4d, reorient_canonical, resample, DefaultPreprocessor, largest_component, iterative_label, transform_seg2image, extract_alternate, fill_canal, crop_image2seg, recalculate_correspondence
from model import internal_predict_sliding_window_return_logits, internal_get_sliding_window_slicers, GradingModel, BasicBlock, Bottleneck
from dicom_io import load_dicoms_from_folder, load_study_dicoms

# --- Константы ---
DEFAULT_CROP_MARGIN = 10
DEFAULT_CROP_SHAPE_PAD = (16, 16, 0)  # (D, H, W) добавка к crop_shape
LANDMARK_LABELS = [63, 64, 65, 66, 67, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 91, 92, 93, 94, 95, 96, 100]


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


def get_crop_shape(median_bbox: np.ndarray, pad=DEFAULT_CROP_SHAPE_PAD) -> tuple:
    """
    Возвращает форму кропа с учётом паддинга.
    """
    crop_shape = tuple((stop - start) + p for (start, stop), p in zip(median_bbox, pad))
    return crop_shape


def get_centered_slices(bbox_center: list, crop_shape: tuple, data_shape: tuple) -> tuple:
    """
    Возвращает tuple из срезов, центрированных по bbox_center, с формой crop_shape.
    """
    slices = []
    for dim in range(3):
        start = bbox_center[dim] - (crop_shape[dim] // 2)
        stop = start + crop_shape[dim]
        start = max(0, start)
        stop = min(data_shape[dim], stop)
        slices.append(slice(start, stop))
    return tuple(slices)


def crop_and_pad(data: np.ndarray, slices: tuple, crop_shape: tuple) -> np.ndarray:
    """
    Кроп и паддинг массива data по срезам slices до формы crop_shape.
    """
    cropped = data[slices]
    return pad_or_crop_to_shape(cropped, crop_shape)


def process_disk(mri_data: np.ndarray, mask_data: np.ndarray, disk_label: int, crop_shape: tuple, nifti_img: Nifti1Image, nifti_seg: Nifti1Image, output_dir: Path, model):
    """
    Кроп, выравнивание и сохранение ROI для одного диска.
    """
    disk_mask = (mask_data == disk_label)
    if not np.any(disk_mask):
        return
    bbox = find_bounding_box(mask_data, disk_label)
    if bbox is None:
        return
    bbox_center = [((sl.start + sl.stop) // 2) for sl in bbox]
    slices = get_centered_slices(bbox_center, crop_shape, mask_data.shape)
    mri_crop = crop_and_pad(mri_data, slices, crop_shape)
    mask_crop = crop_and_pad(mask_data, slices, crop_shape)
    angle_deg = compute_principal_angle(mask_crop == disk_label)
    rotated_mri, rotated_mask = rotate_volume_and_mask(mri_crop, mask_crop, angle_deg)

    mean = rotated_mri.mean()
    std = rotated_mri.std() if rotated_mri.std() > 0 else 1.0
    img = (rotated_mri - mean) / std
    img = img[np.newaxis, ...]
    img = torch.tensor(img).unsqueeze(0).float().to('cuda')
    model.to('cuda')
    model.eval()
    with torch.no_grad():
        garding = model(img)
    return [torch.argmax(result).detach().cpu().numpy() for result in garding]



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
    parser.add_argument('--input_sag', type=Path, help='The input DICOM folder containing the sagittal images.', default=Path(r'F:\WorkSpace\Z-Union\test MRI\T2_listez_3'))
    parser.add_argument('--input_ax', type=Path, help='The input DICOM folder containing the axial images.', default=Path(r'F:\WorkSpace\Z-Union\test MRI\AX T2'))
    parser.add_argument('--output', type=Path, help='The output folder where the segmentation results will be saved.', default=Path(r'./results'))
    return parser.parse_args()


def run_segmentation_pipeline(args) -> tuple:
    """
    Основной пайплайн сегментации: загрузка, препроцессинг, сегментация, постобработка.
    Возвращает итоговые nifti_img, nifti_seg.
    """
    med_data = load_dicoms_from_folder(args, require_extensions=True)
    med_data = average4d(med_data)
    med_data = reorient_canonical(med_data)
    med_data = resample(med_data)
    # Получить актуальное correspondence после преобразований
    sag_img, ax_img = med_data[0], med_data[1]
    nibabel.save(sag_img, 'sag.nii.gz')
    nibabel.save(ax_img, 'ax.nii.gz')

    new_correspondence = recalculate_correspondence(sag_img, ax_img)

    nifti_img = Nifti1Image(med_data[0].dataobj[np.newaxis, ...], med_data[0].affine, med_data[0].header)
    preprocessor = DefaultPreprocessor()
    for step in ['step_1', 'step_2']:
        data, seg, properties  = preprocessor.run_case(nifti_img, transpose_forward=[0, 1, 2])
        img, slicer_revert_padding = pad_nd_image(data, (128, 96, 96), 'constant', {"constant_values": 0}, True)
        slicers = internal_get_sliding_window_slicers(img.shape[1:])
        model = torch.load(rf'model/weights/sag_{step}.pth', weights_only=False)
        num_segmentation_heads = 9 if step == 'step_1' else 11
        predicted_logits = internal_predict_sliding_window_return_logits(
            img, slicers, model, patch_size=(128, 96, 96), results_device='cuda', num_segmentation_heads=num_segmentation_heads)
        predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        predicted_logits = predicted_logits.detach().cpu().numpy()
        segmentation_reverted_cropping, predicted_probabilities = preprocessor.convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits)
        nifti_seg = largest_component(segmentation_reverted_cropping, binarize=True, dilate=5)
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
        nifti_seg = transform_seg2image(med_data[0], nifti_seg)
        nifti_img = crop_image2seg(med_data[0], nifti_seg, margin=DEFAULT_CROP_MARGIN)
        nifti_seg = transform_seg2image(nifti_img, nifti_seg)
        if step == 'step_1':
            nifti_seg = extract_alternate(nifti_seg, labels=list(range(63, 101)))
            img_data = np.asanyarray(nifti_img.dataobj)
            seg_data = np.asanyarray(nifti_seg.dataobj)
            assert img_data.shape == seg_data.shape, f"Shapes do not match: {img_data.shape} vs {seg_data.shape}"
            multi_channel = np.stack([img_data, seg_data], axis=0)
            nifti_img = Nifti1Image(multi_channel, nifti_img.affine, nifti_img.header)


    axial = Nifti1Image(med_data[1].dataobj[np.newaxis, ...], med_data[1].affine, med_data[1].header)

    data, seg, properties = preprocessor.run_case(axial, transpose_forward=[0, 1, 2])
    img, slicer_revert_padding = pad_nd_image(data, (320, 320), 'constant', {"constant_values": 0}, True)
    slicers = internal_get_sliding_window_slicers(img.shape[1:], patch_size=(320, 320))
    model = torch.load(r'model/weights/ax.pth', weights_only=False)
    predicted_logits = internal_predict_sliding_window_return_logits(
        img, slicers, model, patch_size=(320, 320), results_device='cuda',
        num_segmentation_heads=4, mode='2d')
    predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
    predicted_logits = predicted_logits.detach().cpu().numpy()
    segmentation_reverted_cropping, predicted_probabilities = preprocessor.convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_logits)
    nibabel.save(segmentation_reverted_cropping, 'ax.nii.gz')

    return nifti_img, nifti_seg


def main():
    """
    Основная точка входа: парсинг аргументов, запуск пайплайна, обработка и сохранение ROI для каждого диска.
    """
    args = parse_args()
    output_dir = args.output
    output_dir.mkdir(exist_ok=True, parents=True)
    nifti_img, nifti_seg = run_segmentation_pipeline(args)
    save(nifti_img, 'img.nii.gz')
    save(nifti_seg, 'seg.nii.gz')
    median_bbox = np.array([[136, 168], [104, 150], [16, 70]])
    median_bbox = median_bbox.astype(int)
    crop_shape = get_crop_shape(median_bbox)
    mri_data_sag = nifti_img.get_fdata().transpose(2, 1, 0)
    mask_data_sag = nifti_seg.get_fdata().transpose(2, 1, 0)

    model = torch.load('model/weights/grading.pth', weights_only=False)

    first_pass = False
    result = []
    for disk_label in LANDMARK_LABELS:
        garding = process_disk(mri_data_sag, mask_data_sag, disk_label, crop_shape, nifti_img, nifti_seg, output_dir, model=model)
        if garding is None:
            continue
        if not first_pass and garding is not None:
            first_pass = True
            continue

        result.append(garding)



    pass


if __name__ == '__main__':
    main()