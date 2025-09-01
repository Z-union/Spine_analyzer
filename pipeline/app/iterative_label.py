import scipy.ndimage as ndi
import numpy as np
import nibabel as nib
import torchio as tio
import matplotlib.pyplot as plt
import os


def iterative_label(
        seg: 'nib.nifti1.Nifti1Image',
        loc: 'nib.nifti1.Nifti1Image' = None,
        selected_disc_landmarks: list = [],
        disc_labels: list = [],
        disc_landmark_labels: list = [],
        disc_landmark_output_labels: list = [],
        disc_output_step: int = 1,
        vertebrae_labels: list = [],
        vertebrae_landmark_output_labels: list = [],
        vertebrae_output_step: int = 1,
        vertebrae_extra_labels: list = [],
        region_max_sizes: list = [5, 12, 6, 1],
        region_default_sizes: list = [5, 12, 5, 1],
        loc_disc_labels: list = [],
        canal_labels: list = [],
        canal_output_label: int = 0,
        cord_labels: list = [],
        cord_output_label: int = 0,
        sacrum_labels: list = [],
        sacrum_output_label: int = 0,
        map_input_dict: dict = {},
        dilation_size: int = 1,
        disc_default_superior_output: int = 0,
    ) -> 'nib.nifti1.Nifti1Image':
    """
    Итеративная разметка позвонков, межпозвоночных дисков, спинного мозга и канала по исходной сегментации.
    Алгоритм поэтапно выделяет компоненты, сортирует их, сопоставляет с анатомическими лендмарками и формирует итоговую разметку.

    :param seg: nib.nifti1.Nifti1Image — исходная сегментация
    :param loc: nib.nifti1.Nifti1Image — локализатор (опционально)
    :param selected_disc_landmarks: список меток дисков-ориентиров
    :param disc_labels: список меток дисков
    :param disc_landmark_labels: все возможные метки дисков-ориентиров
    :param disc_landmark_output_labels: выходные метки для дисков-ориентиров
    :param disc_output_step: шаг между метками дисков в выходе
    :param vertebrae_labels: список меток позвонков
    :param vertebrae_landmark_output_labels: выходные метки для позвонков-ориентиров
    :param vertebrae_output_step: шаг между метками позвонков в выходе
    :param vertebrae_extra_labels: дополнительные метки позвонков
    :param region_max_sizes: максимальное число дисков/позвонков в каждом регионе
    :param region_default_sizes: дефолтное число дисков/позвонков в каждом регионе
    :param loc_disc_labels: метки локализатора для поиска первого диска
    :param canal_labels: метки канала
    :param canal_output_label: выходная метка канала
    :param cord_labels: метки спинного мозга
    :param cord_output_label: выходная метка спинного мозга
    :param sacrum_labels: метки крестца
    :param sacrum_output_label: выходная метка крестца
    :param map_input_dict: словарь сопоставления входных и выходных меток
    :param dilation_size: размер дилатации перед поиском компонент
    :param disc_default_superior_output: дефолтная метка верхнего диска
    :return: nib.nifti1.Nifti1Image — сегментация с разметкой позвонков, дисков, канала и спинного мозга
    """
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8).transpose(2, 1, 0)

    affine = seg.affine

    # Инвертируем X и Y оси
    affine[0, :3] *= -1
    affine[1, :3] *= -1

    # Инвертируем смещения по X и Y
    affine[0, 3] *= -1
    affine[1, 3] *= -1

    seg = nib.Nifti1Image(seg_data, affine, seg.header)
    seg.set_qform(affine)
    seg.set_sform(affine)

    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    output_seg_data = np.zeros_like(seg_data)

    # Get the canal centerline indices to use for sorting the discs and vertebrae based on the prjection on the canal centerline
    canal_centerline_indices = _get_canal_centerline_indices(seg_data, canal_labels + cord_labels)

    # Get the mask of the voxels anterior to the canal, this helps in sorting the vertebrae considering only the vertebrae body
    mask_aterior_to_canal = _get_mask_aterior_to_canal(seg_data, canal_labels + cord_labels)

    # Get sorted connected components superio-inferior (SI) for the disc labels
    disc_mask_labeled, disc_num_labels, disc_sorted_labels, disc_sorted_z_indices = _get_si_sorted_components(
        seg,
        disc_labels,
        canal_centerline_indices,
        mask_aterior_to_canal,
        dilation_size,
        combine_labels=True,
    )

    # Get sorted connected components superio-inferior (SI) for the vertebrae labels
    vert_mask_labeled, vert_num_labels, vert_sorted_labels, vert_sorted_z_indices = _get_si_sorted_components(
        seg,
        vertebrae_labels,
        canal_centerline_indices,
        mask_aterior_to_canal,
        dilation_size,
    )

    # Combine sequential vertebrae labels based on some conditions
    vert_mask_labeled, vert_num_labels, vert_sorted_labels, vert_sorted_z_indices = _merge_vertebrae_with_same_label(
        seg,
        vertebrae_labels,
        vert_mask_labeled,
        vert_num_labels,
        vert_sorted_labels,
        vert_sorted_z_indices,
        canal_centerline_indices,
        mask_aterior_to_canal,
    )

    # Combine extra labels with adjacent vertebrae labels
    vert_mask_labeled, vert_num_labels, vert_sorted_labels, vert_sorted_z_indices = _merge_extra_labels_with_adjacent_vertebrae(
        seg,
        vert_mask_labeled,
        vert_num_labels,
        vert_sorted_labels,
        vert_sorted_z_indices,
        vertebrae_extra_labels,
        canal_centerline_indices,
        mask_aterior_to_canal,
    )

    # Get the landmark disc labels and output labels - {label in sorted labels: output label}
    # TODO Currently only the first 2 landmark from selected_disc_landmarks is used, to get all landmarks see TODO in the function
    map_disc_sorted_labels_landmark2output = _get_landmark_output_labels(
        seg,
        loc,
        disc_mask_labeled,
        disc_sorted_labels,
        selected_disc_landmarks,
        disc_landmark_labels,
        disc_landmark_output_labels,
        loc_disc_labels,
        disc_default_superior_output,
    )

    # Build a list containing all possible labels for the disc ordered superio-inferior
    all_possible_disc_output_labels = []
    for l, s in zip(disc_landmark_output_labels, region_max_sizes):
        for i in range(s):
            all_possible_disc_output_labels.append(l + i * disc_output_step)

    # Make a list containing all possible labels for the disc ordered superio-inferior with the default region sizes
    all_default_disc_output_labels = []
    for l, s in zip(disc_landmark_output_labels, region_default_sizes):
        for i in range(s):
            all_default_disc_output_labels.append(l + i * disc_output_step)

    # Make a dict mapping the sorted disc labels to the output labels
    map_disc_sorted_labels_2output = {}

    # We loop over all the landmarks starting from the most superior
    for l_disc in [_ for _ in disc_sorted_labels if _ in map_disc_sorted_labels_landmark2output]:

        # If this is the most superior landmark, we have to adjust the start indices to start from the most superior label in the image
        if len(map_disc_sorted_labels_2output) == 0:
            # Get the index of the current landmark in the sorted disc labels
            start_l = disc_sorted_labels.index(l_disc)

            # Get the index of the current landmark in the list of all default disc output labels
            start_o_def = all_default_disc_output_labels.index(map_disc_sorted_labels_landmark2output[l_disc])

            # Adjust the start indices
            start_l, start_o_def = max(0, start_l - start_o_def), max(0, start_o_def - start_l)

            # Map the sorted disc labels to the output labels
            for l, o in zip(disc_sorted_labels[start_l:], all_default_disc_output_labels[start_o_def:]):
                map_disc_sorted_labels_2output[l] = o

        # Get the index of the current landmark in the sorted disc labels
        start_l = disc_sorted_labels.index(l_disc)

        # Get the index of the current landmark in the list of all possible disc output labels
        start_o = all_possible_disc_output_labels.index(map_disc_sorted_labels_landmark2output[l_disc])

        # Map the sorted disc labels to the output labels
        # This will ovveride the mapping from the previous landmarks for all labels inferior to the current landmark
        for l, o in zip(disc_sorted_labels[start_l:], all_possible_disc_output_labels[start_o:]):
            map_disc_sorted_labels_2output[l] = o

    # Label the discs with the output labels superio-inferior
    for l, o in map_disc_sorted_labels_2output.items():
        output_seg_data[disc_mask_labeled == l] = o

    if vert_num_labels > 0:
        # Build a list containing all possible labels for the vertebrae ordered superio-inferior
        # We start with the C1 and C2 labels as the first landmark is the C3 vertebrae
        all_possible_vertebrae_output_labels = [
            vertebrae_landmark_output_labels[0] - 2 * vertebrae_output_step, # C1
            vertebrae_landmark_output_labels[0] - vertebrae_output_step # C2
        ]
        for l, s in zip(vertebrae_landmark_output_labels, region_max_sizes):
            for i in range(s):
                all_possible_vertebrae_output_labels.append(l + i * vertebrae_output_step)

        # Make a list containing all possible labels for the vertebrae ordered superio-inferior with the default region sizes
        all_default_vertebrae_output_labels = [
            vertebrae_landmark_output_labels[0] - 2 * vertebrae_output_step, # C1
            vertebrae_landmark_output_labels[0] - vertebrae_output_step # C2
        ]
        for l, s in zip(vertebrae_landmark_output_labels, region_default_sizes):
            for i in range(s):
                all_default_vertebrae_output_labels.append(l + i * vertebrae_output_step)

        # Sort the combined disc+vert labels by their z-index
        sorted_labels = vert_sorted_labels + disc_sorted_labels
        sorted_z_indices = vert_sorted_z_indices + disc_sorted_z_indices
        is_vert = [True] * len(vert_sorted_labels) + [False] * len(disc_sorted_labels)

        # Sort the labels by their z-index (reversed to go from superior to inferior)
        sorted_z_indices, sorted_labels, is_vert = zip(*sorted(zip(sorted_z_indices, sorted_labels, is_vert))[::-1])
        sorted_z_indices, sorted_labels, is_vert = list(sorted_z_indices), list(sorted_labels), list(is_vert)

        # Look for two discs with no vertebrae between them, if found, look if there is there is 2 vertebrae without a disc between them just next to the discs and switch the second disk with the first vertebrae, if not found, look if there is there is 2 vertebrae without a disc between them just above to the discs and switch the first disk with the second vertebrae.
        # This is useful in cases where only the spinous processes is segmented in the last vertebrae and the disc is not segmented
        for idx in range(len(sorted_labels) - 1):
            # Cehck if this is two discs without a vertebrae between them
            if not is_vert[idx] and not is_vert[idx + 1]:
                # Check if there is two vertebrae without a disc between them just next to the discs
                if idx < len(sorted_labels) - 3 and is_vert[idx + 2] and is_vert[idx + 3]:
                    sorted_labels[idx + 1], sorted_labels[idx + 2] = sorted_labels[idx + 2], sorted_labels[idx + 1]
                    sorted_z_indices[idx + 1], sorted_z_indices[idx + 2] = sorted_z_indices[idx + 2], sorted_z_indices[idx + 1]
                    is_vert[idx + 1], is_vert[idx + 2] = is_vert[idx + 2], is_vert[idx + 1]

                # Check if there is two vertebrae without a disc between them just above to the discs
                elif idx > 1 and is_vert[idx - 1] and is_vert[idx - 2]:
                    sorted_labels[idx], sorted_labels[idx - 1] = sorted_labels[idx - 1], sorted_labels[idx]
                    sorted_z_indices[idx], sorted_z_indices[idx - 1] = sorted_z_indices[idx - 1], sorted_z_indices[idx]
                    is_vert[idx], is_vert[idx - 1] = is_vert[idx - 1], is_vert[idx]

        # If there is a disc at the top and 2 vertebrae below it, switch the disc with the first vertebrae
        if len(sorted_labels) > 2 and not is_vert[0] and is_vert[1] and is_vert[2]:
            sorted_labels[0], sorted_labels[1] = sorted_labels[1], sorted_labels[0]
            sorted_z_indices[0], sorted_z_indices[1] = sorted_z_indices[1], sorted_z_indices[0]
            is_vert[0], is_vert[1] = is_vert[1], is_vert[0]

        # If there is a disc at the bottom and 2 vertebrae above it, switch the disc with the second vertebrae
        if len(sorted_labels) > 2 and not is_vert[-1] and is_vert[-2] and is_vert[-3]:
            sorted_labels[-1], sorted_labels[-2] = sorted_labels[-2], sorted_labels[-1]
            sorted_z_indices[-1], sorted_z_indices[-2] = sorted_z_indices[-2], sorted_z_indices[-1]
            is_vert[-1], is_vert[-2] = is_vert[-2], is_vert[-1]

        # Make a dict mapping disc to vertebrae labels
        disc_output_labels_2vert = dict(zip(all_possible_disc_output_labels, all_possible_vertebrae_output_labels[2:]))

        # Make a dict mapping the sorted vertebrae labels to the output labels
        map_vert_sorted_labels_2output = {}

        l_vert_output = 0
        # We loop over all the labels starting from the most superior, and we map the vertebrae labels to the output labels
        for idx, curr_l, curr_is_vert in zip(range(len(sorted_labels)), sorted_labels, is_vert):
            if not curr_is_vert: # This is a disc
                # If the current disc is not in the map, continue
                if curr_l not in map_disc_sorted_labels_2output:
                    continue

                # Get the output label for the disc and vertebrae
                l_disc_output = map_disc_sorted_labels_2output[curr_l]
                l_vert_output = disc_output_labels_2vert[l_disc_output]

                if idx > 0 and len(map_vert_sorted_labels_2output) == 0: # This is the first disc
                    # Get the index of the current vertebrae in the default vertebrae output labels list
                    i = all_default_vertebrae_output_labels.index(l_vert_output)

                    # Get the labels of the vertebrae superior to the current disc
                    prev_vert_ls = [l for l, _is_v in zip(sorted_labels[idx - 1::-1], is_vert[idx - 1::-1]) if _is_v]

                    # Map all the vertebrae superior to the current disc to the default vertebrae output labels
                    for l, o in zip(prev_vert_ls, all_default_vertebrae_output_labels[i - 1::-1]):
                        map_vert_sorted_labels_2output[l] = o

            elif l_vert_output > 0: # This is a vertebrae
                # If this is the last vertebrae and no disc btween it and the prev vertebrae, map it to the next vertebrae output label
                # This is useful in cases where only the spinous processes is segmented in the last vertebrae and the disc is not segmented
                if idx == len(sorted_labels) - 1 and idx > 0 and is_vert[idx - 1] and l_vert_output != all_possible_vertebrae_output_labels[-1]:
                    map_vert_sorted_labels_2output[curr_l] = all_possible_vertebrae_output_labels[all_possible_vertebrae_output_labels.index(l_vert_output) + 1]
                else:
                    map_vert_sorted_labels_2output[curr_l] = l_vert_output

        # Label the vertebrae with the output labels superio-inferior
        for l, o in map_vert_sorted_labels_2output.items():
            output_seg_data[vert_mask_labeled == l] = o

    # Map Spinal Canal to the output label
    if canal_labels is not None and len(canal_labels) > 0 and canal_output_label > 0:
        output_seg_data[np.isin(seg_data, canal_labels)] = canal_output_label

    # Map Spinal Cord to the output label
    if cord_labels is not None and len(cord_labels) > 0 and cord_output_label > 0:
        output_seg_data[np.isin(seg_data, cord_labels)] = cord_output_label

    # Map Sacrum to the output label
    if sacrum_labels is not None and len(sacrum_labels) > 0 and sacrum_output_label > 0:
        output_seg_data[np.isin(seg_data, sacrum_labels)] = sacrum_output_label

    # Use the map to map input labels to the final output
    # This is useful to map the input C1 to the output.
    for orig, new in map_input_dict.items():
        output_seg_data[seg_data == int(orig)] = int(new)

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg

def _get_canal_centerline_indices(
        seg_data,
        canal_labels=[],
    ):
    '''
    Get the indices of the canal centerline.
    '''
    # If no canal labels is found in the segmentation, raise an error
    if not np.any(np.isin(canal_labels, seg_data)):
        raise ValueError(f"No canal found in the segmentation (canal labels: {canal_labels})")

    # Get array of indices for x, y, and z axes
    indices = np.indices(seg_data.shape)

    # Create a mask of the canal
    mask_canal = np.isin(seg_data, canal_labels)

    # Create a mask the canal centerline by finding the middle voxels in x and y axes for each z index
    mask_min_x_indices = np.min(indices[0], where=mask_canal, initial=np.iinfo(indices.dtype).max, keepdims=True, axis=(0, 1))
    mask_max_x_indices = np.max(indices[0], where=mask_canal, initial=np.iinfo(indices.dtype).min, keepdims=True, axis=(0, 1))
    mask_mid_x = indices[0] == ((mask_min_x_indices + mask_max_x_indices) // 2)
    mask_min_y_indices = np.min(indices[1], where=mask_canal, initial=np.iinfo(indices.dtype).max, keepdims=True, axis=(0, 1))
    mask_max_y_indices = np.max(indices[1], where=mask_canal, initial=np.iinfo(indices.dtype).min, keepdims=True, axis=(0, 1))
    mask_mid_y = indices[1] == ((mask_min_y_indices + mask_max_y_indices) // 2)
    mask_canal_centerline = mask_canal * mask_mid_x * mask_mid_y

    # Get the indices of the canal centerline
    return np.array(np.nonzero(mask_canal_centerline)).T

def _sort_labels_si(
        mask_labeled,
        labels,
        canal_centerline_indices,
        mask_aterior_to_canal=None,
    ):
    '''
    Sort the labels by their z-index (reversed to go from superior to inferior).
    '''
    # Get the indices of the center of mass for each label
    labels_indices = np.array(ndi.center_of_mass(np.isin(mask_labeled, labels), mask_labeled, labels))

    # Get the distance of each label indices from the canal centerline
    labels_distances_from_centerline = np.linalg.norm(labels_indices[:, None, :] - canal_centerline_indices[None, ...],axis=2)

    # Get the z-index of the closest canal centerline voxel for each label
    labels_z_indices = canal_centerline_indices[np.argmin(labels_distances_from_centerline, axis=1), -1]

    # If mask_aterior_to_canal is provided, calculate the center of mass in this mask if the label is inside the mask
    if mask_aterior_to_canal is not None:
        # Save the existing labels z-index in a dict
        labels_z_indices_dict = dict(zip(labels, labels_z_indices))

        # Get the part that is anterior to the canal od mask_labeled
        mask_labeled_aterior_to_canal = mask_aterior_to_canal * mask_labeled

        # Get the labels that contain voxels anterior to the canal
        labels_masked = np.isin(labels, mask_labeled_aterior_to_canal)

        # Get the indices of the center of mass for each label
        labels_masked_indices = np.array(ndi.center_of_mass(np.isin(mask_labeled_aterior_to_canal, labels_masked), mask_labeled_aterior_to_canal, labels_masked))

        # Get the distance of each label indices for each voxel in the canal centerline
        labels_masked_distances_from_centerline = np.linalg.norm(labels_masked_indices[:, None, :] - canal_centerline_indices[None, :],axis=2)

        # Get the z-index of the closest canal centerline voxel for each label
        labels_masked_z_indices = canal_centerline_indices[np.argmin(labels_masked_distances_from_centerline, axis=1), -1]

        # Update the dict with the new z-index of the labels anterior to the canal
        for l, z in zip(labels_masked, labels_masked_z_indices):
            labels_z_indices_dict[l] = z

        # Update the labels_z_indices from the dict
        labels_z_indices = [labels_z_indices_dict[l] for l in labels]

    # Sort the labels by their z-index (reversed to go from superior to inferior)
    return zip(*sorted(zip(labels_z_indices, labels))[::-1])

def _get_si_sorted_components(
        seg,
        labels,
        canal_centerline_indices,
        mask_aterior_to_canal=None,
        dilation_size=1,
        combine_labels=False,
    ):
    '''
    Get sorted connected components superio-inferior (SI) for the given labels in the segmentation.
    '''
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    binary_dilation_structure = ndi.iterate_structure(ndi.generate_binary_structure(3, 1), dilation_size)

    # Skip if no labels are provided
    if len(labels) == 0:
        return None, 0, [], []

    if combine_labels:
        # For discs, combine all labels before label continue voxels since the discs not touching each other
        _labels = [labels]
    else:
        _labels = [[_] for _ in labels]

    # Init labeled segmentation
    mask_labeled, num_labels = np.zeros_like(seg_data, dtype=np.uint32), 0

    # For each label, find connected voxels and label them into separate labels
    for l in _labels:
        mask = np.isin(seg_data, l)

        # Dilate the mask to combine small disconnected regions
        mask_dilated = ndi.binary_dilation(mask, binary_dilation_structure)

        # Label the connected voxels in the dilated mask into separate labels
        tmp_mask_labeled, tmp_num_labels = ndi.label(mask_dilated.astype(np.uint32), np.ones((3, 3, 3)))

        # Undo dilation
        tmp_mask_labeled *= mask

        # Add current labels to the labeled segmentation
        if tmp_num_labels > 0:
            mask_labeled[tmp_mask_labeled != 0] = tmp_mask_labeled[tmp_mask_labeled != 0] + num_labels
            num_labels += tmp_num_labels

    # If no label found, raise error
    if num_labels == 0:
        raise ValueError(f"Some label must be in the segmentation (labels: {labels})")

    # Reduce size of mask_labeled
    if mask_labeled.max() < np.iinfo(np.uint8).max:
        mask_labeled = mask_labeled.astype(np.uint8)
    elif mask_labeled.max() < np.iinfo(np.uint16).max:
        mask_labeled = mask_labeled.astype(np.uint16)

    # Sort the labels by their z-index (reversed to go from superior to inferior)
    sorted_z_indices, sorted_labels = _sort_labels_si(
        mask_labeled, range(1,num_labels+1), canal_centerline_indices, mask_aterior_to_canal
    )
    return mask_labeled, num_labels, list(sorted_labels), list(sorted_z_indices)

def _get_mask_aterior_to_canal(
        seg_data,
        canal_labels=[],
    ):
    '''
    Get the mask of the voxels anterior to the canal.
    '''
    # Get array of indices for x, y, and z axes
    indices = np.indices(seg_data.shape)

    # Create a mask of the canal
    mask_canal = np.isin(seg_data, canal_labels)

    # Create a mask the canal centerline by finding the middle voxels in x and y axes for each z index
    mask_min_y_indices = np.min(indices[1], where=mask_canal, initial=np.iinfo(indices.dtype).max, keepdims=True, axis=(0, 1))
    mask_max_y_indices = np.max(indices[1], where=mask_canal, initial=np.iinfo(indices.dtype).min, keepdims=True, axis=(0, 1))
    mask_mid_y_indices = (mask_min_y_indices + mask_max_y_indices) // 2

    return indices[1] > mask_mid_y_indices

def _merge_vertebrae_with_same_label(
        seg,
        labels,
        mask_labeled,
        num_labels,
        sorted_labels,
        sorted_z_indices,
        canal_centerline_indices,
        mask_aterior_to_canal=None,
    ):
    '''
    Combine sequential vertebrae labels if they have the same value in the original segmentation.
    This is useful when parts of the vertebrae are not touching in the segmentation but have the same odd/even value.
    '''
    if num_labels == 0 or len(labels) <= 1:
        return mask_labeled, num_labels, sorted_labels, sorted_z_indices

    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    new_sorted_labels = []

    # Store the previous label and the original label of the previous label
    prev_l, prev_orig_label = 0, 0

    # Loop over the sorted labels
    for l in sorted_labels:
        # Get the original label of the current label
        curr_orig_label = seg_data[mask_labeled == l].flat[0]

        # Combine the current label with the previous label if they have the same original label
        if curr_orig_label == prev_orig_label:
            # Combine the current label with the previous label
            mask_labeled[mask_labeled == l] = prev_l
            num_labels -= 1

        else:
            # Add the current label to the new sorted labels
            new_sorted_labels.append(l)
            prev_l, prev_orig_label = l, curr_orig_label

    sorted_labels = new_sorted_labels

    # Reduce size of mask_labeled
    if mask_labeled.max() < np.iinfo(np.uint8).max:
        mask_labeled = mask_labeled.astype(np.uint8)
    elif mask_labeled.max() < np.iinfo(np.uint16).max:
        mask_labeled = mask_labeled.astype(np.uint16)

    # Sort the labels by their z-index (reversed to go from superior to inferior)
    sorted_z_indices, sorted_labels = _sort_labels_si(
        mask_labeled, sorted_labels, canal_centerline_indices, mask_aterior_to_canal
    )

    return mask_labeled, num_labels, list(sorted_labels), list(sorted_z_indices)

def _merge_extra_labels_with_adjacent_vertebrae(
        seg,
        mask_labeled,
        num_labels,
        sorted_labels,
        sorted_z_indices,
        extra_labels,
        canal_centerline_indices,
        mask_aterior_to_canal,
    ):
    '''
    Combine extra labels with adjacent vertebrae labels.
    This is useful to combine segmentations of vertebrae introduced during region-based training, where the model sometimes outputs a general vertebrae label instead of the specific odd or even vertebrae.
    The process adjusts these remnants by merging them with the closest odd or even vertebrae to ensure correct segmentation.
    '''
    if num_labels == 0 or len(extra_labels) == 0:
        return mask_labeled, num_labels, sorted_labels, sorted_z_indices

    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    mask_extra = np.isin(seg_data, extra_labels)

    # Loop over vertebral labels (from inferior because the transverse process make it steal from above)
    for i in range(num_labels - 1, -1, -1):
        # Mkae mask for the current vertebrae with filling the holes and dilating it
        mask = _fill(mask_labeled == sorted_labels[i])
        mask = ndi.binary_dilation(mask, ndi.iterate_structure(ndi.generate_binary_structure(3, 1), 1))

        # Add the intersection of the mask with the extra labels to the current verebrae
        mask_labeled[mask_extra * mask] = sorted_labels[i]

    # Sort the labels by their z-index (reversed to go from superior to inferior)
    sorted_z_indices, sorted_labels = _sort_labels_si(
        mask_labeled, sorted_labels, canal_centerline_indices, mask_aterior_to_canal
    )

    return mask_labeled, num_labels, list(sorted_labels), list(sorted_z_indices)

def _get_landmark_output_labels(
        seg,
        loc,
        mask_labeled,
        sorted_labels,
        selected_landmarks,
        landmark_labels,
        landmark_output_labels,
        loc_labels,
        default_superior_output,
    ):
    '''
    Get dict mapping labels from sorted_labels to the output labels based on the landmarks in the segmentation or localizer.
    '''
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    loc_data = loc and np.asanyarray(loc.dataobj).round().astype(np.uint8)

    map_landmark_labels = dict(zip(landmark_labels, landmark_output_labels))

    # If localizer is provided, transform it to the segmentation space
    if loc_data is not None:
        loc_data = tio.Resample(
            tio.ScalarImage(tensor=seg_data[None, ...], affine=seg.affine)
        )(
            tio.LabelMap(tensor=loc_data[None, ...], affine=loc.affine)
        ).data.numpy()[0, ...].astype(np.uint8)

    # Init dict to store the output labels for the landmarks
    map_landmark_outputs = {}

    mask_seg_data_landmarks = np.isin(seg_data, landmark_labels)

    # First we try to set initial output labels from the localizer
    if loc_data is not None:
        # Make mask for the intersection of the localizer labels and the labels in the segmentation
        mask = np.isin(loc_data, loc_labels) * np.isin(mask_labeled, sorted_labels)
        mask_labeled_masked = mask * mask_labeled
        loc_data_masked = mask * loc_data

        # First we try to look for the landmarks in the localizer
        for output_label in np.array(landmark_output_labels)[np.isin(landmark_output_labels, loc_data_masked)].tolist():
            # Map the label with the most voxels in the localizer landmark to the output label
            map_landmark_outputs[np.argmax(np.bincount(mask_labeled_masked[loc_data_masked == output_label].flat))] = output_label

        if len(map_landmark_outputs) == 0:
            # Get the first label from sorted_labels that is in the localizer specified labels
            first_sorted_labels_in_loc = next(np.array(sorted_labels)[np.isin(sorted_labels, mask_labeled_masked)].flat, 0)

            if first_sorted_labels_in_loc > 0:
                # Get the output label for first_sorted_labels_in_loc, the label from the localizer that has the most voxels in it
                map_landmark_outputs[first_sorted_labels_in_loc] = np.argmax(np.bincount(loc_data_masked[mask_labeled_masked == first_sorted_labels_in_loc].flat))

    # If no output label found from the localizer, try to set the output labels from landmarks in the segmentation
    if len(map_landmark_outputs) == 0:
        for l in selected_landmarks:
            ############################################################################################################
            # TODO Remove this reake when we trust all the landmarks to get all landmarks instead of the first 2
            if len(map_landmark_outputs) > 0 and selected_landmarks.index(l) > 1:
                break
            ############################################################################################################
            if l in map_landmark_labels and l in seg_data:
                mask_labeled_l = np.argmax(np.bincount(mask_labeled[seg_data == l].flat))
                # We map only if the landmark cover the majority of the voxels in the mask_labeled label
                if np.argmax(np.bincount(seg_data[mask_seg_data_landmarks & (mask_labeled == mask_labeled_l)].flat)) == l:
                    map_landmark_outputs[mask_labeled_l] = map_landmark_labels[l]

    # If no init label found, set the default superior label
    if len(map_landmark_outputs) == 0 and default_superior_output > 0:
        map_landmark_outputs[sorted_labels[0]] = default_superior_output

    # If no init label found, print error
    if len(map_landmark_outputs) == 0:
        if loc_data is not None:
            raise ValueError(
                f"At least one of the landmarks must be in the segmentation or localizer (landmarks: {selected_landmarks}. "
                f"Check {loc_labels}), make sure the localizer is in the same space as the segmentation"
            )
        raise ValueError(f"At least one of the landmarks must be in the segmentation or localizer (landmarks: {selected_landmarks})")

    return map_landmark_outputs

def _fill(mask):
    '''
    Fill holes in a binary mask

    Parameters
    ----------
    mask : np.ndarray
        Binary mask

    Returns
    -------
    np.ndarray
        Binary mask with holes filled
    '''
    # Get array of indices for x, y, and z axes
    indices = np.indices(mask.shape)

    mask_min_x = np.min(np.where(mask, indices[0], np.inf), axis=0)[np.newaxis, ...]
    mask_max_x = np.max(np.where(mask, indices[0], -np.inf), axis=0)[np.newaxis, ...]
    mask_min_y = np.min(np.where(mask, indices[1], np.inf), axis=1)[:, np.newaxis, :]
    mask_max_y = np.max(np.where(mask, indices[1], -np.inf), axis=1)[:, np.newaxis, :]
    mask_min_z = np.min(np.where(mask, indices[2], np.inf), axis=2)[:, :, np.newaxis]
    mask_max_z = np.max(np.where(mask, indices[2], -np.inf), axis=2)[:, :, np.newaxis]

    return \
        ((mask_min_x <= indices[0]) & (indices[0] <= mask_max_x)) | \
        ((mask_min_y <= indices[1]) & (indices[1] <= mask_max_y)) | \
        ((mask_min_z <= indices[2]) & (indices[2] <= mask_max_z))


def extract_levels(
        seg,
        canal_labels=[],
        disc_labels=[],
        c1_label=0,
        c2_label=0,
):
    '''
    Extract vertebrae levels from Spinal Canal and Discs.

    The function extracts the vertebrae levels from the input segmentation by finding the closest voxel in the canal anteriorline to the middle of each disc.
    The superior voxels of the top vertebrae is set to 1 and the middle voxels between C2-C3 and the superior voxels are set to 2.
    The generated labeling convention follows the one from SCT (https://spinalcordtoolbox.com/stable/user_section/tutorials/vertebral-labeling/labeling-conventions.html)

    Parameters
    ----------
    seg : nib.Nifti1Image
        The input segmentation.
    canal_labels : list
        The canal labels.
    disc_labels : list
        The disc labels starting at C2C3 ordered from superior to inferior.
    c1_label : int
        The label for C1 vertebra in the segmentation, if provided it will be used to determine if C1 is in the segmentation.

    Returns
    -------
    nib.Nifti1Image
        The output segmentation with the vertebrae levels.
    '''
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)
    affine = seg.affine
    # Сначала заменяем все -0.0 на 0.0
    affine[affine == -0.0] = 0.0
    # Затем возвращаем -0.0 в нужные места
    affine[0, 2] = -0.0
    affine[1, 2] = -0.0

    seg = nib.Nifti1Image(seg_data, affine, seg.header)
    seg.set_qform(affine)
    seg.set_sform(affine)

    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    output_seg_data = np.zeros_like(seg_data)

    # Get array of indices for x, y, and z axes
    indices = np.indices(seg_data.shape)

    # Create a mask of the canal
    mask_canal = np.isin(seg_data, canal_labels)

    # If cancl is empty raise an error
    if not np.any(mask_canal):
        raise ValueError(f"No canal labels found in the segmentation.")

    # Create a canal anteriorline shifted toward the posterior tip by finding the middle voxels in x and the maximum voxels in y for each z index
    mask_min_x_indices = np.min(indices[0], where=mask_canal, initial=np.iinfo(indices.dtype).max, keepdims=True,
                                axis=(0, 1))
    mask_max_x_indices = np.max(indices[0], where=mask_canal, initial=np.iinfo(indices.dtype).min, keepdims=True,
                                axis=(0, 1))
    mask_mid_x = indices[0] == ((mask_min_x_indices + mask_max_x_indices) // 2)
    mask_max_y_indices = np.max(indices[1], where=mask_canal, initial=np.iinfo(indices.dtype).min, keepdims=True,
                                axis=(0, 1))
    mask_max_y = indices[1] == mask_max_y_indices
    mask_canal_anteriorline = mask_canal * mask_mid_x * mask_max_y

    # Get the indices of the canal anteriorline
    canal_anteriorline_indices = np.array(np.nonzero(mask_canal_anteriorline)).T

    # Get the labels of the discs in the segmentation
    disc_labels_in_seg = np.array(disc_labels)[np.isin(disc_labels, seg_data)]

    # If no disc labels found in the segmentation raise an error
    if len(disc_labels_in_seg) == 0:
        raise ValueError(f"No disc labels found in the segmentation.")

    # Get the labels of the discs and the output labels
    first_disk_idx = disc_labels.index(disc_labels_in_seg[0])
    out_labels = list(range(3 + first_disk_idx, 3 + first_disk_idx + len(disc_labels_in_seg)))

    # Filter the discs that are in the segmentation
    map_labels = dict(zip(disc_labels_in_seg, out_labels))

    # Create mask of the discs
    mask_discs = np.isin(seg_data, disc_labels_in_seg)

    # Get list of indices for all the discs voxels
    discs_indices = np.nonzero(mask_discs)

    # Get the matching labels for the discs indices
    discs_indices_labels = seg_data[discs_indices]

    # Make the discs_indices 2D array
    discs_indices = np.array(discs_indices).T

    # Calculate the distance of each disc voxel to each canal anteriorline voxel
    discs_distances_from_all_anteriorline = np.linalg.norm(
        discs_indices[:, None, :] - canal_anteriorline_indices[None, ...], axis=2)

    # Find the minimum distance for each disc voxel and the corresponding canal anteriorline index
    discs_distance_from_anteriorline = np.min(discs_distances_from_all_anteriorline, axis=1)
    discs_anteriorline_indices = canal_anteriorline_indices[np.argmin(discs_distances_from_all_anteriorline, axis=1)]

    # Find the closest voxel to the canal anteriorline for each disc label (posterior tip)
    disc_labels_anteriorline_indices = [discs_anteriorline_indices[discs_indices_labels == label][
                                            np.argmin(discs_distance_from_anteriorline[discs_indices_labels == label])]
                                        for label in disc_labels_in_seg]

    # Set the output labels to the closest voxel to the canal anteriorline for each disc
    for idx, label in zip(disc_labels_anteriorline_indices, disc_labels_in_seg):
        output_seg_data[tuple(idx)] = map_labels[label]

    # If C2-C3 and C1 are in the segmentation, set 1 and 2
    if 3 in output_seg_data and c2_label != 0 and c1_label != 0 and all(np.isin([c1_label, c2_label], seg_data)):
        # Place 1 at the top of C2 if C1 is visible in the image
        # Find the location of the C2-C3 disc
        c2c3_index = np.unravel_index(np.argmax(output_seg_data == 3), seg_data.shape)

        # Find the maximum coordinate of the vertebra C1
        c1_coords = np.where(seg_data == c1_label)
        c1_z_max_index = np.max(c1_coords[2])

        # Extract coordinate of the vertebrae
        # The coordinate of 1 needs to be in the same slice as 3 but below the max index of C1
        vert_coords = np.where(seg_data[c2c3_index[0], :, :c1_z_max_index] == c2_label)

        # Check if not empty
        if len(vert_coords[1]) > 0:
            # Find top pixel of the vertebrae
            argmax_z = np.argmax(vert_coords[1])
            top_vert_voxel = tuple([c2c3_index[0]] + [vert_coords[i][argmax_z] for i in range(2)])

            # Set 1 to the superior voxels and project onto the anterior line
            top_vert_distances_from_all_anteriorline = np.linalg.norm(
                top_vert_voxel - canal_anteriorline_indices[None, ...], axis=2)
            top_vert_index_anteriorline = canal_anteriorline_indices[
                np.argmin(top_vert_distances_from_all_anteriorline, axis=1)]
            output_seg_data[tuple(top_vert_index_anteriorline[0])] = 1

            # Set 2 to the middle voxels between C2-C3 and the superior voxels
            c1c2_index = tuple([(top_vert_voxel[i] + c2c3_index[i]) // 2 for i in range(3)])

            # Project 2 on the anterior line
            c1c2_distances_from_all_anteriorline = np.linalg.norm(c1c2_index - canal_anteriorline_indices[None, ...],
                                                                  axis=2)
            c1c2_index_anteriorline = canal_anteriorline_indices[
                np.argmin(c1c2_distances_from_all_anteriorline, axis=1)]
            output_seg_data[tuple(c1c2_index_anteriorline[0])] = 2

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg

def transform_seg2image(
        image,
        seg,
        save_dir: str = None,
        interpolation='nearest',
    ):
    """
    Transform the segmentation to the image space to have the same origin, spacing, direction and shape as the image.
    Optionally saves middle slices of image and segmentation for visual check.

    Parameters
    ----------
    image : nib.Nifti1Image
        Image.
    seg : nib.Nifti1Image
        Segmentation.
    save_dir : str, optional
        Directory to save slices, defaults to None.
    interpolation : str, optional
        Interpolation method, can be 'nearest', 'linear' or 'label', defaults to 'nearest'.

    Returns
    -------
    nib.Nifti1Image
        Output segmentation.
    """
    # Check if the input image is 4D and take the first image from the last axis for resampling
    if len(np.asanyarray(image.dataobj).shape) == 4:
        image = image.slicer[..., 0]

    image_data = np.asanyarray(image.dataobj).astype(np.float64)
    image_affine = image.affine.copy()
    seg_data = np.asanyarray(seg.dataobj)
    if interpolation == 'linear':
        seg_data = seg_data.astype(np.float32)
    else:
        seg_data = seg_data.round().astype(np.uint8)
    seg_affine = seg.affine.copy()

    # Dilations size
    dilation_size = int(np.ceil(np.max(np.array(image.header.get_zooms()) / np.array(seg.header.get_zooms()))))
    pad_width = int(dilation_size * int(np.ceil(np.max(np.array(seg.header.get_zooms()) / np.array(image.header.get_zooms())))))

    if interpolation == 'label':
        image_data = np.pad(image_data, pad_width)
        image_affine[:3, 3] -= (image_affine[:3, :3] @ ([pad_width] * 3))
        seg_data = np.pad(seg_data, pad_width)
        seg_affine[:3, 3] -= (seg_affine[:3, :3] @ ([pad_width] * 3))
        seg_data = ndi.grey_dilation(seg_data, footprint=ndi.iterate_structure(ndi.generate_binary_structure(3, 3), dilation_size))

    # TorchIO images
    tio_img = tio.ScalarImage(tensor=image_data[None, ...], affine=image_affine)
    if interpolation == 'linear':
        tio_seg = tio.ScalarImage(tensor=seg_data[None, ...], affine=seg_affine)
    else:
        tio_seg = tio.LabelMap(tensor=seg_data[None, ...], affine=seg_affine)

    # Resample
    tio_output_seg = tio.Resample(tio_img)(tio_seg)
    output_seg_data = tio_output_seg.data.numpy()[0, ...]
    if interpolation != 'linear':
        output_seg_data = output_seg_data.round().astype(np.uint8)

    if interpolation == 'label':
        com_output_seg_data = np.zeros_like(output_seg_data)
        labels = [_ for _ in np.unique(output_seg_data) if _ != 0]
        com = ndi.center_of_mass(output_seg_data != 0, output_seg_data, labels)
        for label, idx in zip(labels, com):
            idx = np.round(idx).astype(int)
            idx = np.maximum(np.minimum(idx, np.array(com_output_seg_data.shape) - pad_width - 1), [pad_width] * 3)
            com_output_seg_data[tuple(idx)] = label
        output_seg_data = com_output_seg_data[pad_width:-pad_width, pad_width:-pad_width, pad_width:-pad_width]

    output_seg = nib.Nifti1Image(output_seg_data, image.affine, seg.header)

    # Save slices if save_dir is specified
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        mid_slices = [s // 2 for s in output_seg_data.shape]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.ravel()
        # Axial
        axes[0].imshow(image_data[:, :, mid_slices[2]], cmap='gray')
        axes[0].set_title('Image Axial')
        axes[1].imshow(output_seg_data[:, :, mid_slices[2]], cmap='tab20')
        axes[1].set_title('Seg Axial')
        # Sagittal
        axes[2].imshow(image_data[mid_slices[0], :, :], cmap='gray')
        axes[2].set_title('Image Sagittal')
        axes[3].imshow(output_seg_data[mid_slices[0], :, :], cmap='tab20')
        axes[3].set_title('Seg Sagittal')
        # Coronal
        axes[4].imshow(image_data[:, mid_slices[1], :], cmap='gray')
        axes[4].set_title('Image Coronal')
        axes[5].imshow(output_seg_data[:, mid_slices[1], :], cmap='tab20')
        axes[5].set_title('Seg Coronal')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'mid_slices.png'))
        plt.close(fig)

    return output_seg

def extract_alternate(
        seg,
        labels=[],
        prioratize_labels=[],
    ):
    '''
    This function extracts binary masks that include every other intervertebral discs (IVD).
    It loops through the segmentation labels from superior to inferior, selecting alternating discs.
    To choose the first IVD to include, it uses the first disc in the image that matches the labels provided in the prioratize_labels argument, if supplied.
    If prioratize_labels is not provided, it starts from the first disc in the image.
    For inference purposes, this prioritization is not needed, as the goal is simply to include every other disc in the mask, without concern for which disc is selected first.

    Parameters
    ----------
    seg : nib.Nifti1Image
        The input segmentation.
    labels : list of int
        The labels to extract alternate elements from.
    prioratize_labels : list of int
        Specify labels that will be prioratized in the output, the first label in the list will be included in the output.

    Returns
    -------
    nib.Nifti1Image
        The output segmentation with the vertebrae levels.
    '''
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    output_seg_data = np.zeros_like(seg_data)

    # Get the labels in the segmentation
    labels = np.array(labels)[np.isin(labels, seg_data)]

    # Get the labels to prioratize in the output that are in the segmentation and in the labels
    prioratize_labels = np.array(prioratize_labels)[np.isin(prioratize_labels, labels)]

    selected_labels = labels[::2]

    if len(prioratize_labels) > 0 and prioratize_labels[0] not in selected_labels:
        selected_labels = labels[1::2]

    output_seg_data[np.isin(seg_data, selected_labels)] = 1

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg


def fill_canal(
        seg,
        canal_label = 1,
        cord_label = 0,
        largest_canal = False,
        largest_cord = False,
    ):
    '''
    Fill holes in the spinal canal, this will put the spinal canal label in all the voxels (labeled as a background) between the spinal canal and the spinal cord.

    Parameters
    ----------
    seg : nib.Nifti1Image
        Segmentation image.
    canal_label : int, optional
        Label used for Spinal Canal, defaults to 1.
    cord_label : int, optional
        Label used for spinal cord, defaults to 0.
    largest_canal : bool, optional
        Take the largest spinal canal component, defaults to False.
    largest_cord : bool, optional
        Take the largest spinal cord component, defaults to False.

    Returns
    -------
    nib.Nifti1Image
        Output segmentation image with filled spinal canal.
    '''
    output_seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    # Take the largest spinal cord component
    if cord_label and largest_cord and cord_label in output_seg_data:
        cord_mask = output_seg_data == cord_label
        cord_mask_largest = _largest_component(cord_mask)
        output_seg_data[cord_mask & ~cord_mask_largest] = canal_label

    if canal_label in output_seg_data:

        canal_labels = [cord_label, canal_label] if cord_label else [canal_label]

        # Take the largest spinal canal component
        if largest_canal:
            canal_mask = np.isin(output_seg_data, canal_labels)
            canal_mask_largest = _largest_component(canal_mask)
            output_seg_data[canal_mask & ~canal_mask_largest] = 0

        # Create an array of x indices
        x_indices = np.broadcast_to(np.arange(output_seg_data.shape[0])[..., np.newaxis, np.newaxis], output_seg_data.shape)
        # Create an array of y indices
        y_indices = np.broadcast_to(np.arange(output_seg_data.shape[1])[..., np.newaxis], output_seg_data.shape)

        canal_mask = np.isin(output_seg_data, canal_labels)
        canal_mask_min_x = np.min(np.where(canal_mask, x_indices, np.inf), axis=0)[np.newaxis, ...]
        canal_mask_max_x = np.max(np.where(canal_mask, x_indices, -np.inf), axis=0)[np.newaxis, ...]
        canal_mask_min_y = np.min(np.where(canal_mask, y_indices, np.inf), axis=1)[:, np.newaxis, :]
        canal_mask_max_y = np.max(np.where(canal_mask, y_indices, -np.inf), axis=1)[:, np.newaxis, :]
        canal_mask = \
            (canal_mask_min_x <= x_indices) & \
                (x_indices <= canal_mask_max_x) & \
                (canal_mask_min_y <= y_indices) & \
                (y_indices <= canal_mask_max_y)
        canal_mask = canal_mask & (output_seg_data != cord_label) if cord_label else canal_mask
        output_seg_data[canal_mask] = canal_label

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)

    return output_seg

def _largest_component(mask):
    if mask.sum() == 0:
        return mask
    mask_labeled, num_labels = ndi.label(mask, np.ones((3, 3, 3)))
    # Find the label of the largest component
    label_sizes = np.bincount(mask_labeled.ravel())[1:]  # Skip 0 label size
    largest_label = label_sizes.argmax() + 1  # +1 because bincount labels start at 0
    return mask_labeled == largest_label

def crop_image2seg(
        image,
        seg,
        margin = 0,
    ):
    '''
    Crop the image to the non-zero region of the segmentation with a margin.

    Parameters
    ----------
    image: nib.Nifti1Image
        The image to crop.
    seg: nib.Nifti1Image
        The segmentation to use for cropping.
    margin: int
        Margin to add to the cropped region in voxels, defaults to 0 - no margin.

    Returns
    -------
    nib.Nifti1Image
        The cropped image.
    '''
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    # Get bounding box of the segmentation and crop the image
    x, y, z = np.nonzero(seg_data)

    if len(x) > 0 and len(y) > 0 and len(z) > 0:
        # Calculate the bounding box
        min_x, max_x = x.min(), x.max()
        min_y, max_y = y.min(), y.max()
        min_z, max_z = z.min(), z.max()

        # Add margin to the bounding box ensuring it does not exceed the image dimensions
        min_x = max(min_x - margin, 0)
        max_x = min(max_x + margin, seg_data.shape[0] - 1)
        min_y = max(min_y - margin, 0)
        max_y = min(max_y + margin, seg_data.shape[1] - 1)
        min_z = max(min_z - margin, 0)
        max_z = min(max_z + margin, seg_data.shape[2] - 1)

        image = image.slicer[
            min_x: max_x + 1,
            min_y: max_y + 1,
            min_z: max_z + 1
        ]

    return image