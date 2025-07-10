import nibabel
import nibabel as nib
import numpy as np


def average4d(med_data: list) -> list:
    """
    Обёртка для усреднения 4D-медицинских изображений по последнему измерению.
    Преобразует каждый объект в Nifti, корректно приводит типы данных и сохраняет масштабирование.

    :param med_data: Список объектов с методом to_nifti()
    :return: Список усреднённых 3D-изображений (nibabel.Nifti1Image)
    """
    res = []
    for image in med_data:
        if image is None:
            res.append(None)
            continue
        image = image.to_nifti()
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
                image = nib.Nifti1Image(image_data.astype(image_data_dtype), image.affine, image.header)

        output_image = _average4d(image)

        # Ensure correct image dtype, affine and header
        output_image = nib.Nifti1Image(
            np.asanyarray(output_image.dataobj).astype(image_data_dtype),
            output_image.affine, output_image.header
        )
        output_image.set_data_dtype(image_data_dtype)
        output_image.set_qform(output_image.affine)
        output_image.set_sform(output_image.affine)
        # Сохраняем pixel scaling (scl_slope и scl_inter) из исходного изображения
        output_image.header['scl_slope'] = image.header.get('scl_slope', 1.0)
        output_image.header['scl_inter'] = image.header.get('scl_inter', 0.0)
        res.append(output_image)


    return res

def _average4d(image: 'nibabel.nifti1.Nifti1Image') -> 'nibabel.nifti1.Nifti1Image':
    """
    Усредняет последнее измерение 4D-изображения, получая 3D-объём.

    :param image: Входное 4D-изображение (nibabel.Nifti1Image)
    :return: 3D-изображение (nibabel.Nifti1Image)
    """
    output_image_data = np.asanyarray(image.dataobj).astype(np.float64)

    # Average the last dimension
    if len(output_image_data.shape) == 4:
        output_image_data = np.mean(output_image_data, axis=-1)

    output_image = nib.Nifti1Image(output_image_data, image.affine, image.header)

    return output_image