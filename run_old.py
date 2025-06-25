import argparse
import textwrap
from pathlib import Path
from dicom_io import load_dicoms_from_folder
from utils import *
import nibabel as nib
import torch
import numpy as np
from utils.preprocessor import preprocess

def main():
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

    parser.add_argument(
        '--input_sag', type=Path,
        help='The input DICOM folder containing the sagittal images.',
        default=Path(r'F:\WorkSpace\Z-Union\test MRI\SAG T2')
    )
    parser.add_argument(
        '--input_ax', type=Path,
        help='The input DICOM folder containing the axial images.',
        default=Path(r'F:\WorkSpace\Z-Union\test MRI\AX T2')
    )
    parser.add_argument(
        '--output', type=Path,
        help='The output folder where the segmentation results will be saved.',
        default=Path(r'./results')
    )

    args = parser.parse_args()

    med_data = load_dicoms_from_folder(args, require_extensions=True)
    med_data = reorient_spinal_scan_to_canonical(med_data)
    img, label = resample(image=med_data[0].volume, label=None, spacing=med_data[0].pixel_spacing)

    img = np.rot90(img.transpose(2, 0, 1), axes=(1, 2))

    nifti_img = nib.Nifti1Image(img, np.eye(4))

    nib.save(nifti_img, f'{0}_scan_resampled.nii.gz')

    mean = img.mean()
    std = img.std()
    img = (img - mean) / (std + 1e-8)

    # orig_shape = img.shape
    # patches = preprocess(img, np.array(list(med_data.pixel_spacing) + [med_data.slice_thickness]), (1, 1, 1), [128, 96, 96])



    img, slicer_revert_padding = pad_nd_image(img, [128, 96, 96], 'constant', {"constant_values": 0}, True, 2)
    img = img[np.newaxis, ...]
    slicers = internal_get_sliding_window_slicers(img.shape[1:])
    model_1 = torch.load('models/step_1.pth', weights_only=False)
    res = internal_predict_sliding_window_return_logits(img, slicers, model_1)
    # res = res[(slice(None), *slicer_revert_padding)]
    # (slice(0, 1, None), slice(0, 310, None), slice(0, 310, None), slice(15, 80, None))
    # res = model_1(torch.tensor(img))

    seg = np.argmax(res, axis=0).astype(np.uint8)
    nifti_img = nib.Nifti1Image(seg, np.eye(4))

    nib.save(nifti_img, f'{1}_scan_resampled.nii.gz')
    predicted_logits = res[(slice(None), *slicer_revert_padding[1:])]

    # for idx, data in enumerate(med_data):
    #     nib.save(data.to_nifti(), f'{idx}_scan_resampled.nii.gz')

    pass

if __name__ == '__main__':
    main()