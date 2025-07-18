import os


import nibabel
import nibabel as nib
import numpy as np
import torchio as tio

from tqdm import tqdm

def resample(med_data: list, mm: tuple = (1.0, 1.0, 1.0), is_mask: bool = False) -> list:
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

        output_image = _resample(image, mm, is_mask)

        output_image = nib.Nifti1Image(
            np.asanyarray(output_image.dataobj).astype(image_data_dtype),
            output_image.affine, output_image.header
        )
        output_image.set_data_dtype(image_data_dtype)
        output_image.set_qform(output_image.affine)
        output_image.set_sform(output_image.affine)

        res.append(output_image)

    return res

def _resample(image: 'nibabel.Nifti1Image', mm: tuple = (1.0, 1.0, 1.0), is_mask: bool = False) -> 'nibabel.Nifti1Image':
    """
    Ресемплирует изображение до целевого размера вокселя (мм) с помощью torchio.

    :param image: Входное изображение (nibabel.Nifti1Image)
    :param mm: Желаемый размер вокселя (по умолчанию 1x1x1 мм)
    :return: Ресемплированное изображение (nibabel.Nifti1Image)
    """
    image_data = np.asanyarray(image.dataobj).astype(np.float64)

    # Create result
    if not is_mask:
        subject = tio.Resample(mm)(tio.Subject(
            image=tio.ScalarImage(tensor=image_data[None, ...], affine=image.affine),
        ))
    else:
        subject = tio.Resample(mm)(tio.Subject(
            image=tio.LabelMap(tensor=image_data[None, ...], affine=image.affine),
        ))
    output_image_data = subject.image.data.numpy()[0, ...].astype(np.float64)

    output_image = nib.Nifti1Image(output_image_data, subject.image.affine, image.header)
    output_image.set_qform(output_image.affine)

    return output_image

def get_nii_files_recursive(folder):
    nii_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                nii_files.append(os.path.join(root, f))
    return nii_files

def batch_resample(input_dir, output_dir, voxel_size=(1.0, 1.0, 1.0), is_mask=False):
    files = get_nii_files_recursive(input_dir)
    for in_path in tqdm(files):
        rel_path = os.path.relpath(in_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            img = nib.load(in_path)
            resampled_img = resample([img], mm=voxel_size, is_mask=is_mask)[0]
            nib.save(resampled_img, out_path)
            print(f"Resampled: {rel_path}")
        except Exception as e:
            print(f"Error processing {rel_path}: {e}")

if __name__ == "__main__":
    images_dir = r"C:\Users\Gleb\PycharmProjects\totalspineseg\nnUNet_raw\Dataset001_SpineAX\imagesTr"
    labels_dir = r"C:\Users\Gleb\PycharmProjects\totalspineseg\nnUNet_raw\Dataset001_SpineAX\labelsTr"
    images_out = r"C:\Users\Gleb\PycharmProjects\totalspineseg\nnUNet_raw\Dataset003_SpineAX\imagesTr"
    labels_out = r"C:\Users\Gleb\PycharmProjects\totalspineseg\nnUNet_raw\Dataset003_SpineAX\labelsTr"
    voxel_size = (1.0, 1.0, 1.0)

    batch_resample(images_dir, images_out, voxel_size, is_mask=False)
    batch_resample(labels_dir, labels_out, voxel_size, is_mask=True)