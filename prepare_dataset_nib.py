import os
import numpy as np
import nibabel as nib
import pandas as pd
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
from scipy.ndimage import affine_transform
from utils_bbox_pca import find_bounding_box, compute_principal_angle

# Пути к данным
IMAGES_DIR = r"F:\\WorkSpace\\Z-Union\\chess\\10159290\\images"
MASKS_DIR = r"F:\\WorkSpace\\Z-Union\\chess\\10159290\\masks"
GRADINGS_CSV = r"F:\\WorkSpace\\Z-Union\\chess\\10159290\\radiological_gradings.csv"
OUTPUT_DIR = r"F:\\WorkSpace\\Z-Union\\chess\\10159290\\dataset_cuts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Загрузка оценок
if os.path.exists(GRADINGS_CSV):
    gradings = pd.read_csv(GRADINGS_CSV)
else:
    gradings = None

def reorient_canonical_sitk(image_sitk):
    '''
    Получаем SimpleITK.Image, внутри обрабатываем через nibabel для канонической ориентации,
    возвращаем обратно SimpleITK.Image.
    '''
    # 1. Переводим из sitk в numpy
    image_np = sitk.GetArrayFromImage(image_sitk)  # numpy array (Z, Y, X)

    # 2. Берем геометрию из sitk
    spacing = image_sitk.GetSpacing()
    origin = image_sitk.GetOrigin()
    direction = image_sitk.GetDirection()

    # 3. Переводим affine sitk -> nibabel affine
    # Внимание: sitk хранит direction в виде 1D tuple, нужно преобразовать в матрицу 3x3
    direction_matrix = np.array(direction).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = direction_matrix * np.array(spacing).reshape(1, 3)
    affine[:3, 3] = origin

    # 4. Создаем nibabel образ
    image_nifti = nib.Nifti1Image(image_np, affine)

    # 5. Определяем исходный dtype
    image_data_dtype = image_np.dtype

    # 6. Проверка диапазона значений и рескейл (точно как в твоей функции)
    if np.issubdtype(image_data_dtype, np.integer):
        image_data = image_np.astype(np.float64)
        image_min, image_max = image_data.min(), image_data.max()
        dtype_min, dtype_max = np.iinfo(image_data_dtype).min, np.iinfo(image_data_dtype).max
        if (image_min < dtype_min) or (image_max > dtype_max):
            data_rescaled = image_data * (dtype_max - dtype_min) / (image_max - image_min)
            image_data = data_rescaled - (data_rescaled.min() - dtype_min)
            image_nifti = nib.Nifti1Image(image_data.astype(image_data_dtype), affine)

    # 7. Приведение к канонической ориентации через nibabel
    image_nifti_canonical = nib.as_closest_canonical(image_nifti)

    # 8. Перевод обратно из nibabel -> numpy -> sitk
    image_np_canonical = np.asanyarray(image_nifti_canonical.dataobj)
    affine_canonical = image_nifti_canonical.affine

    # 9. Извлекаем новую геометрию для sitk
    new_spacing = np.linalg.norm(affine_canonical[:3, :3], axis=0)
    new_direction = (affine_canonical[:3, :3] / new_spacing).flatten()
    new_origin = affine_canonical[:3, 3]

    # 10. Создаем обратно SimpleITK.Image
    image_sitk_canonical = sitk.GetImageFromArray(image_np_canonical)
    image_sitk_canonical.SetSpacing(tuple(new_spacing))
    image_sitk_canonical.SetDirection(tuple(new_direction))
    image_sitk_canonical.SetOrigin(tuple(new_origin))

    return image_sitk_canonical

def pad_or_crop_hw(volume, target_hw, is_mask=False):
    h, w = volume.shape[1], volume.shape[2]
    new_h, new_w = target_hw
    result = np.zeros((volume.shape[0], new_h, new_w), dtype=volume.dtype)
    offset_h = (new_h - h) // 2
    offset_w = (new_w - w) // 2
    h_slice_src = slice(max(0, -offset_h), min(h, new_h - offset_h))
    w_slice_src = slice(max(0, -offset_w), min(w, new_w - offset_w))
    h_slice_dst = slice(max(0, offset_h), max(0, offset_h) + (h_slice_src.stop - h_slice_src.start))
    w_slice_dst = slice(max(0, offset_w), max(0, offset_w) + (w_slice_src.stop - w_slice_src.start))
    result[:, h_slice_dst, w_slice_dst] = volume[:, h_slice_src, w_slice_src]
    return result

def rotate_volume_and_mask(volume, mask, angle_deg):
    angle_rad = -np.radians(angle_deg)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0],
        [0,                0,                1]
    ])
    center = 0.5 * np.array(volume.shape)
    offset = center - rotation_matrix @ center
    rotated_volume = affine_transform(volume, rotation_matrix, offset=offset, order=1)
    rotated_mask = affine_transform(mask, rotation_matrix, offset=offset, order=0)
    return rotated_volume, rotated_mask

def resample_to_spacing(image, new_spacing=(1.0, 1.0, 1.0), is_mask=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)

def process_images():
    # Загрузка медианного bbox
    median_bbox_path = os.path.join(OUTPUT_DIR, 'median_bbox.npy')
    if not os.path.exists(median_bbox_path):
        raise RuntimeError("median_bbox.npy не найден. Сначала выполните 'collect_bboxes'.")
    median_bbox = np.load(median_bbox_path)  # shape (3, 2)
    median_bbox = median_bbox.astype(int)
    crop_shape = [stop - start for start, stop in median_bbox]
    crop_shape[0] += 16
    crop_shape[1] += 16

    image_files = sorted(glob(os.path.join(IMAGES_DIR, '*.mha')))
    for img_path in tqdm(image_files, desc="Processing images (median bbox)"):
        img_name = os.path.basename(img_path)
        patient_id = img_name.split('_')[0]
        mask_path = os.path.join(MASKS_DIR, img_name)
        if not os.path.exists(mask_path):
            continue
        try:
            img_itk = reorient_canonical_sitk(sitk.ReadImage(img_path))
            mask_itk = reorient_canonical_sitk(sitk.ReadImage(mask_path))
            img_itk = resample_to_spacing(img_itk, new_spacing=(1.0, 1.0, 1.0), is_mask=False)
            mask_itk = resample_to_spacing(mask_itk, new_spacing=(1.0, 1.0, 1.0), is_mask=True)
            mri_data = sitk.GetArrayFromImage(img_itk)
            mask_data = sitk.GetArrayFromImage(mask_itk).astype(np.uint8)
            # Получаем affine из SimpleITK
            origin = np.array(img_itk.GetOrigin())
            spacing = np.array(img_itk.GetSpacing())
            direction = np.array(img_itk.GetDirection()).reshape(3, 3)
            affine = np.eye(4)
            affine[:3, :3] = direction * spacing[np.newaxis, :]
            affine[:3, 3] = origin
        except Exception as e:
            print(f"[WARNING] Failed to read or process {img_path} or its mask: {e}")
            continue
        for disk_label in range(201, 250):
            disk_mask = (mask_data == disk_label)
            if not np.any(disk_mask):
                continue
            bbox = find_bounding_box(mask_data, disk_label)
            if bbox is None:
                continue
            # Центр bbox
            bbox_center = [((sl.start + sl.stop) // 2) for sl in bbox]
            # Центр медианного bbox
            # median_center = [((start + stop) // 2) for start, stop in median_bbox]
            # Смещение для выравнивания центра
            # offset = [bc - mc for bc, mc in zip(bbox_center, median_center)]
            # Новый bbox с формой медианного bbox, центрированный относительно центра найденного bbox
            slices = []
            for dim in range(3):
                start = bbox_center[dim] - (crop_shape[dim] // 2)
                stop = start + crop_shape[dim]
                # Ограничение по границам изображения
                start = max(0, start)
                stop = min(mask_data.shape[dim], stop)
                slices.append(slice(start, stop))
            mri_crop = mri_data[tuple(slices)]
            mask_crop = mask_data[tuple(slices)]
            # Привести к точному размеру медианного bbox (D, H, W)
            def pad_or_crop_to_shape(arr, target_shape):
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
            mri_crop = pad_or_crop_to_shape(mri_crop, crop_shape)
            mask_crop = pad_or_crop_to_shape(mask_crop, crop_shape)
            angle_deg = compute_principal_angle(mask_crop == disk_label)
            rotated_mri, rotated_mask = rotate_volume_and_mask(mri_crop, mask_crop, angle_deg)
            out_img_path = os.path.join(OUTPUT_DIR, f'{patient_id}_disk{disk_label}_img.nii.gz')
            out_mask_path = os.path.join(OUTPUT_DIR, f'{patient_id}_disk{disk_label}_mask.nii.gz')
            new_mri = nib.Nifti1Image(rotated_mri, affine)
            new_mask = nib.Nifti1Image(rotated_mask, affine)




            nib.save(new_mri, out_img_path)
            nib.save(new_mask, out_mask_path)
            if gradings is not None:
                grading_row = gradings[(gradings['Patient'] == int(patient_id)) & (gradings['IVD label'] == (disk_label - 200))]
                if not grading_row.empty:
                    # Сохраняем только нужные метки
                    grading_cols = [
                        'Modic',
                        'UP endplate',
                        'LOW endplate',
                        'Spondylolisthesis',
                        'Disc herniation',
                        'Disc narrowing',
                        'Disc bulging',
                        'Pfirrman grade',
                    ]
                    grading_row = grading_row[grading_cols]
                    grading_row.to_csv(os.path.join(OUTPUT_DIR, f'{patient_id}_disk{disk_label}_grading.csv'), index=False)

def collect_bboxes():
    image_files = sorted(glob(os.path.join(IMAGES_DIR, '*.mha')))
    all_bboxes = []
    for img_path in tqdm(image_files, desc="Collecting bboxes"):
        img_name = os.path.basename(img_path)
        patient_id = img_name.split('_')[0]
        mask_path = os.path.join(MASKS_DIR, img_name)
        if not os.path.exists(mask_path):
            continue
        try:
            # img_itk = reorient_canonical_sitk(sitk.ReadImage(img_path))
            mask_itk = reorient_canonical_sitk(sitk.ReadImage(mask_path))
            # img_itk = resample_to_spacing(img_itk, new_spacing=(1.0, 1.0, 1.0), is_mask=False)
            mask_itk = resample_to_spacing(mask_itk, new_spacing=(1.0, 1.0, 1.0), is_mask=True)
            mask_data = sitk.GetArrayFromImage(mask_itk).astype(np.uint8)
        except Exception as e:
            print(f"[WARNING] Failed to read or process {img_path} or its mask: {e}")
            continue
        for disk_label in range(201, 250):
            if not np.any(mask_data == disk_label):
                continue
            bbox = find_bounding_box(mask_data, disk_label)
            if bbox is None:
                continue
            final_padding = 5
            padded_bbox = []
            for dim, sl in enumerate(bbox):
                start = max(0, sl.start - final_padding)
                stop = min(mask_data.shape[dim], sl.stop + final_padding)
                padded_bbox.append((start, stop))
            all_bboxes.append(padded_bbox)
    # Сохраняем все bbox и медианный bbox
    all_bboxes_np = np.array(all_bboxes)
    median_bbox = np.median(all_bboxes_np, axis=0).astype(int)
    median_bbox += median_bbox % 2
    np.save(os.path.join(OUTPUT_DIR, 'all_bboxes.npy'), all_bboxes_np)
    np.save(os.path.join(OUTPUT_DIR, 'median_bbox.npy'), median_bbox)
    print(f"Saved median bbox: {median_bbox}")

if __name__ == "__main__":
    # import sys
    # if len(sys.argv) > 1 and sys.argv[1] == 'collect_bboxes':
    #     collect_bboxes()
    # else:
    #     process_images()

    collect_bboxes()

    process_images()

