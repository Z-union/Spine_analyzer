import numpy as np

def find_bounding_box(mask, target_class):
    indices = np.argwhere(mask == target_class)
    if indices.size == 0:
        return None
    min_coords = indices.min(axis=0)
    max_coords = indices.max(axis=0) + 1
    return tuple(slice(start, end) for start, end in zip(min_coords, max_coords))

def compute_principal_angle(mask_disk):
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
