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
    image_np_canonical = np.asanyarray(image_nifti.dataobj)
    affine_canonical = image_nifti.affine

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
    # Собираем список всех patient_id
    all_img_names = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.mha')]
    patient_ids = set([name.split('_')[0] for name in all_img_names])
    for patient_id in tqdm(sorted(patient_ids)):
        # Собираем пути к каналам
        channel_files = {'t1': None, 't2': None, 't2_SPACE': None}
        for ch in channel_files.keys():
            fname = f'{patient_id}_{ch}.mha'
            fpath = os.path.join(IMAGES_DIR, fname)
            if os.path.exists(fpath):
                channel_files[ch] = fpath
        # Сегментация (берём по t1, t2 или t2_SPACE — что есть)
        mask_path = None
        for ch in ['t1', 't2', 't2_SPACE']:
            fname = f'{patient_id}_{ch}.mha'
            mpath = os.path.join(MASKS_DIR, fname)
            if os.path.exists(mpath):
                mask_path = mpath
                break
        if mask_path is None:
            continue
        # Читаем маску
        mask_itk = sitk.ReadImage(mask_path)
        mask_itk = resample_to_spacing(mask_itk, new_spacing=(1.0, 1.0, 1.0), is_mask=True)
        mask = sitk.GetArrayFromImage(mask_itk)
        for disk_label in range(201, 250):
            disk_mask = (mask == disk_label)
            if not np.any(disk_mask):
                continue
            coords = np.argwhere(disk_mask)
            R, mean = get_pca_transform_central(coords, central_percent=0.6)
            # --- Жёсткая фиксация направления осей ---
            vertebra_up_coords = np.argwhere(mask == (disk_label - 200))
            vertebra_down_coords = np.argwhere(mask == (disk_label - 199))
            if vertebra_up_coords.size > 0 and vertebra_down_coords.size > 0:
                z_up = vertebra_up_coords[:, 0].mean()
                z_down = vertebra_down_coords[:, 0].mean()
                if z_down < z_up:
                    R[2, :] *= -1  # Инвертируем ось Z (нижний всегда ниже)
            canal_coords = np.argwhere(mask == 2)
            disk_coords = np.argwhere(mask == disk_label)
            if canal_coords.size > 0 and disk_coords.size > 0:
                x_canal = canal_coords[:, 2].mean()
                x_disk = disk_coords[:, 2].mean()
                if x_canal < x_disk:
                    R[0, :] *= -1  # Инвертируем ось X (канал всегда справа)
            # ---
            # Для каждого канала: читаем, ресемплим, вращаем, crop
            img_channels = []
            shape_ref = None
            bbox_disk = None
            for ch in ['t1', 't2', 't2_SPACE']:
                img_path = channel_files[ch]
                if img_path is not None:
                    img_itk = sitk.ReadImage(img_path)
                    img_itk = resample_to_spacing(img_itk, new_spacing=(1.0, 1.0, 1.0), is_mask=False)
                    img = sitk.GetArrayFromImage(img_itk)
                    img_rot, _ = apply_affine_to_image_and_mask(img, mask, R, mean)
                    if bbox_disk is None:
                        disk_mask_rot = (mask == disk_label)
                        bbox_disk = get_bounding_box(disk_mask_rot)
                    img_crop = crop_with_bbox(img_rot, bbox_disk)
                    if shape_ref is None:
                        shape_ref = img_crop.shape
                    else:
                        # Приводим к shape_ref (pad/crop)
                        pad = [(0, max(0, shape_ref[d] - img_crop.shape[d])) for d in range(3)]
                        img_crop = np.pad(img_crop, pad, mode='constant')
                        img_crop = img_crop[tuple(slice(0, shape_ref[d]) for d in range(3))]
                    img_channels.append(img_crop)
                else:
                    # Нет канала — заполняем нулями
                    if shape_ref is None:
                        # Нужно shape_ref — пропускаем, обработаем после первого канала
                        img_channels.append(None)
                    else:
                        img_channels.append(np.zeros(shape_ref, dtype=np.float32))
            # Если shape_ref появился только после первого канала, заполняем пропущенные
            if shape_ref is None:
                # Не удалось определить форму — пропускаем этот диск
                continue
            for i in range(3):
                if img_channels[i] is None:
                    img_channels[i] = np.zeros(shape_ref, dtype=np.float32)
            # Собираем channels-first
            img_3ch = np.stack(img_channels, axis=0)
            # Аналогично crop для маски
            mask_rot = apply_affine_to_image_and_mask(mask, mask, R, mean, is_mask=True)[1]
            mask_crop = crop_with_bbox(mask_rot, bbox_disk)
            out_img_path = os.path.join(OUTPUT_DIR, f'{patient_id}_disk{disk_label}_img.nii.gz')
            out_mask_path = os.path.join(OUTPUT_DIR, f'{patient_id}_disk{disk_label}_mask.nii.gz')
            img_crop_itk = sitk.GetImageFromArray(img_3ch)
            mask_crop_itk = sitk.GetImageFromArray(mask_crop)
            img_crop_itk.SetSpacing((1.0, 1.0, 1.0))
            mask_crop_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(img_crop_itk, out_img_path)
            sitk.WriteImage(mask_crop_itk, out_mask_path)
            # --- Сохраняем первый канал отдельно для теста ---
            first_channel = img_3ch[0]
            out_first_channel_path = os.path.join(OUTPUT_DIR, f'{patient_id}_disk{disk_label}_img_first_channel.nii.gz')
            first_channel_itk = sitk.GetImageFromArray(first_channel)
            first_channel_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(first_channel_itk, out_first_channel_path)
            # ---
            grading_row = gradings[(gradings['Patient'] == int(patient_id)) & (gradings['IVD label'] == (disk_label - 200))]
            if not grading_row.empty:
                grading_row.to_csv(os.path.join(OUTPUT_DIR, f'{patient_id}_disk{disk_label}_grading.csv'), index=False)

if __name__ == '__main__':
    main()
