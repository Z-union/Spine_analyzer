import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import nibabel as nib
from sklearn.decomposition import PCA
from tqdm import tqdm

# Параметры путей
IMAGES_DIR = r"F:\WorkSpace\Z-Union\chess\10159290\images"
MASKS_DIR = r"F:\WorkSpace\Z-Union\chess\10159290\masks"
GRADINGS_CSV = r"F:\WorkSpace\Z-Union\chess\10159290\radiological_gradings.csv"
OUTPUT_DIR = r"F:\WorkSpace\Z-Union\chess\10159290\dataset_cuts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_bounding_box(mask):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    minc = coords.min(axis=0)
    maxc = coords.max(axis=0)
    return minc, maxc

def crop_with_bbox(img, bbox):
    minc, maxc = bbox
    slices = tuple(slice(minc[d], maxc[d]+1) for d in range(len(minc)))
    return img[slices]

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

def get_pca_transform_central(coords, central_percent=0.6):
    # PCA по центральной части диска
    pca = PCA(n_components=3)
    pca.fit(coords)
    main_axis = np.argmax(pca.explained_variance_)
    # Проекция координат на главную ось
    projections = coords @ pca.components_[main_axis]
    min_proj, max_proj = projections.min(), projections.max()
    center = (min_proj + max_proj) / 2
    half_width = (max_proj - min_proj) * central_percent / 2
    mask_central = (projections >= center - half_width) & (projections <= center + half_width)
    coords_central = coords[mask_central]
    # Повторно PCA по центральной части
    pca_central = PCA(n_components=3)
    pca_central.fit(coords_central)
    R = pca_central.components_
    # Жёстко: главная ось -> X (axis=2)
    R_new = np.stack([R[2], R[1], R[0]], axis=0)
    if np.linalg.det(R_new) < 0:
        R_new[2, :] *= -1
    return R_new, pca_central.mean_

def apply_affine_to_image_and_mask(img, mask, R, mean, is_mask=False):
    img_itk = sitk.GetImageFromArray(img)
    mask_itk = sitk.GetImageFromArray(mask)
    spacing = (1.0, 1.0, 1.0)
    img_itk.SetSpacing(spacing)
    mask_itk.SetSpacing(spacing)
    center = tuple(mean[::-1])
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(R.flatten())
    affine.SetCenter(center)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img_itk)
    resampler.SetTransform(affine.GetInverse())
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(img_itk.GetSize())
    resampler.SetOutputOrigin(img_itk.GetOrigin())
    resampler.SetOutputDirection(img_itk.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    img_rot = resampler.Execute(img_itk)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    mask_rot = resampler.Execute(mask_itk)
    return sitk.GetArrayFromImage(img_rot), sitk.GetArrayFromImage(mask_rot)

def get_half_bounding_box(mask, half='lower', axis=0):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    minc = coords.min(axis=0)
    maxc = coords.max(axis=0)
    mid = (minc[axis] + maxc[axis]) // 2
    if half == 'lower':
        maxc[axis] = mid
    else:
        minc[axis] = mid + 1
    return minc, maxc


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

def main():
    gradings = pd.read_csv(GRADINGS_CSV)
    for img_name in tqdm(os.listdir(IMAGES_DIR)):
        if not img_name.endswith('.mha'):
            continue
        patient_id = img_name.split('_')[0]
        img_path = os.path.join(IMAGES_DIR, img_name)
        mask_path = os.path.join(MASKS_DIR, img_name)
        if not os.path.exists(mask_path):
            continue
        # img_itk = reorient_canonical_sitk(sitk.ReadImage(img_path))
        # mask_itk = reorient_canonical_sitk(sitk.ReadImage(mask_path))

        img_itk = sitk.ReadImage(img_path)
        mask_itk = sitk.ReadImage(mask_path)

        img_itk = resample_to_spacing(img_itk, new_spacing=(1.0, 1.0, 1.0), is_mask=False)
        mask_itk = resample_to_spacing(mask_itk, new_spacing=(1.0, 1.0, 1.0), is_mask=True)
        img = sitk.GetArrayFromImage(img_itk)
        mask = sitk.GetArrayFromImage(mask_itk)

        for disk_label in range(201, 250):
            disk_mask = (mask == disk_label)
            if not np.any(disk_mask):
                continue
            coords = np.argwhere(disk_mask)
            # PCA только по центральной части диска
            R, mean = get_pca_transform_central(coords, central_percent=0.6)
            img_rot, mask_rot = apply_affine_to_image_and_mask(img, mask, R, mean)
            # bounding box только по текущему диску
            disk_mask_rot = (mask_rot == disk_label)
            bbox_disk = get_bounding_box(disk_mask_rot)
            if bbox_disk is None:
                continue
            # Половины позвонков
            vertebra_up_mask_rot = (mask_rot == (disk_label - 200))
            vertebra_down_mask_rot = (mask_rot == (disk_label - 199))
            bbox_up_half = get_half_bounding_box(vertebra_up_mask_rot, half='lower', axis=0)
            bbox_down_half = get_half_bounding_box(vertebra_down_mask_rot, half='upper', axis=0)
            # Собираем итоговый bbox
            bboxes = [bbox for bbox in [bbox_up_half, bbox_disk, bbox_down_half] if bbox is not None]
            minc = np.min([b[0] for b in bboxes], axis=0)
            maxc = np.max([b[1] for b in bboxes], axis=0)
            bbox_total = (minc, maxc)
            img_crop = crop_with_bbox(img_rot, bbox_total)
            mask_crop = crop_with_bbox(mask_rot, bbox_total)
            out_img_path = os.path.join(OUTPUT_DIR, f'{patient_id}_disk{disk_label}_img.nii.gz')
            out_mask_path = os.path.join(OUTPUT_DIR, f'{patient_id}_disk{disk_label}_mask.nii.gz')
            img_crop_itk = sitk.GetImageFromArray(img_crop)
            mask_crop_itk = sitk.GetImageFromArray(mask_crop)
            img_crop_itk.SetSpacing((1.0, 1.0, 1.0))
            mask_crop_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(img_crop_itk, out_img_path)
            sitk.WriteImage(mask_crop_itk, out_mask_path)
            grading_row = gradings[(gradings['Patient'] == int(patient_id)) & (gradings['IVD label'] == (disk_label - 200))]
            if not grading_row.empty:
                grading_row.to_csv(os.path.join(OUTPUT_DIR, f'{patient_id}_disk{disk_label}_grading.csv'), index=False)

if __name__ == '__main__':
    main()
