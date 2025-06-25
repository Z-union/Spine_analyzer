from nibabel.nifti1 import Nifti1Image
import nibabel as nib
import numpy as np
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation, inv_ornt_aff

def as_closest_canonical(img):
    ornt = io_orientation(img.affine)
    can_ornt = axcodes2ornt(('R', 'A', 'S'))
    transform = ornt_transform(ornt, can_ornt)
    new_data = apply_orientation(img.get_fdata(), transform)
    new_affine = img.affine @ inv_ornt_aff(transform, img.shape)
    return Nifti1Image(new_data, new_affine, img.header)

def reorient_canonical(med_data: list) -> list:
    """
    Переориентирует изображения к ближайшей канонической (стандартной) ориентации.
    Корректно приводит типы данных и affine, сохраняет масштабирование.

    :param med_data: Список nibabel.Nifti1Image для переориентации
    :return: Список переориентированных изображений (nibabel.Nifti1Image)
    """
    res = []
    for image in med_data:

        # Get image dtype from the image data (preferred over header dtype to avoid data loss)
        image_data_dtype = getattr(np, np.asanyarray(image.dataobj).dtype.name)

        # Rescale the image to the output dtype range if necessary
        if "int" in np.dtype(image_data_dtype).name:
            image_data = np.asanyarray(image.dataobj).astype(np.float64)
            image_min, image_max = image_data.min(), image_data.max()
            dtype_min, dtype_max = np.iinfo(image_data_dtype).min, np.iinfo(image_data_dtype).max
            if (image_min < dtype_min) or (dtype_max < image_max):
                data_rescaled = image_data * (dtype_max - dtype_min) / (image_max - image_min)
                image_data = data_rescaled - (data_rescaled.min() - dtype_min)
                image = Nifti1Image(image_data.astype(image_data_dtype), image.affine, image.header)

        # Transform the image to the closest canonical orientation
        output_image = as_closest_canonical(image)

        # Ensure correct image dtype, affine and header
        output_image = Nifti1Image(
            np.asanyarray(output_image.dataobj).astype(image_data_dtype),
            output_image.affine, output_image.header
        )
        output_image.set_data_dtype(image_data_dtype)
        output_image.set_qform(output_image.affine)
        output_image.set_sform(output_image.affine)

        res.append(output_image)
    return res

def recalculate_correspondence(sag_img, ax_img):
    """
    Пересчитывает correspondence для преобразованных Nifti1Image (например, после reorient_canonical/resample).
    Сопоставляет срезы по ближайшей Z-координате в физическом пространстве.

    :param sag_img: Nifti1Image сагиттальный
    :param ax_img: Nifti1Image аксиальный
    :return: correspondence: список кортежей (i, j, sag_z, ax_z, distance)
    """
    def get_slice_centers(nifti_img):
        shape = nifti_img.shape
        affine = nifti_img.affine
        centers = []
        for i in range(shape[2]):  # предполагаем, что Z — срезы
            idx = np.array([shape[0] // 2, shape[1] // 2, i, 1])
            xyz = affine @ idx
            centers.append(xyz)
        return np.array(centers)

    sag_centers = get_slice_centers(sag_img)
    ax_centers = get_slice_centers(ax_img)

    correspondence = []
    for i, sag_z in enumerate(sag_centers[:, 2]):
        j = np.argmin(np.abs(ax_centers[:, 2] - sag_z))
        distance = np.abs(ax_centers[j, 2] - sag_z)
        correspondence.append((i, j, sag_z, ax_centers[j, 2], distance))
    return correspondence