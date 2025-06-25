import nibabel
import nibabel as nib
import numpy as np
import torchio as tio

def resample(med_data: list, mm: tuple = (1.0, 1.0, 1.0)) -> list:
    """
    Обёртка для ресемплирования медицинских изображений до заданного размера вокселя (мм).
    Корректно приводит типы данных и affine, сохраняет масштабирование.

    :param med_data: Список nibabel.Nifti1Image для ресемплирования
    :param mm: Желаемый размер вокселя (по умолчанию 1x1x1 мм)
    :return: Список ресемплированных изображений (nibabel.Nifti1Image)
    """
    res = []
    for image in med_data:
        # Get image dtype from the image data (preferred over header dtype to avoid data loss)
        image_data_dtype = getattr(np, np.asanyarray(image.dataobj).dtype.name)


        if "int" in np.dtype(image_data_dtype).name:
            image_data = np.asanyarray(image.dataobj).astype(np.float64)
            image_min, image_max = image_data.min(), image_data.max()
            dtype_min, dtype_max = np.iinfo(image_data_dtype).min, np.iinfo(image_data_dtype).max
            if (image_min < dtype_min) or (dtype_max < image_max):
                data_rescaled = image_data * (dtype_max - dtype_min) / (image_max - image_min)
                image_data = data_rescaled - (data_rescaled.min() - dtype_min)
                image = nib.Nifti1Image(image_data.astype(image_data_dtype), image.affine, image.header)

        output_image = _resample(image, mm)

        output_image = nib.Nifti1Image(
            np.asanyarray(output_image.dataobj).astype(image_data_dtype),
            output_image.affine, output_image.header
        )
        output_image.set_data_dtype(image_data_dtype)
        output_image.set_qform(output_image.affine)
        output_image.set_sform(output_image.affine)

        res.append(output_image)

    return res

def _resample(image: 'nibabel.Nifti1Image', mm: tuple = (1.0, 1.0, 1.0)) -> 'nibabel.Nifti1Image':
    """
    Ресемплирует изображение до целевого размера вокселя (мм) с помощью torchio.

    :param image: Входное изображение (nibabel.Nifti1Image)
    :param mm: Желаемый размер вокселя (по умолчанию 1x1x1 мм)
    :return: Ресемплированное изображение (nibabel.Nifti1Image)
    """
    image_data = np.asanyarray(image.dataobj).astype(np.float64)

    # Create result
    subject = tio.Resample(mm)(tio.Subject(
        image=tio.ScalarImage(tensor=image_data[None, ...], affine=image.affine),
    ))
    output_image_data = subject.image.data.numpy()[0, ...].astype(np.float64)

    output_image = nib.Nifti1Image(output_image_data, subject.image.affine, image.header)
    output_image.set_qform(output_image.affine)

    return output_image