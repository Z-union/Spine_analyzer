import numpy as np
from scipy.ndimage import gaussian_filter

from typing import Union, List, Tuple
import torch

def internal_predict_sliding_window_return_logits(
        data: np.ndarray,
        slicers,
        network,
        use_gaussian: bool = True,
        patch_size: Tuple[int, ...] = None,
        num_segmentation_heads: int = 9,
        results_device = 'cpu',
        mode: str = '3d',
    ) -> torch.Tensor:
    """
    Версия NumPy: скользящее окно + (опционально) Gaussian-weighted fusion.
    data: np.ndarray, форму которого предполагаем (C_in, D, H, W) или (C_in, H, W) для 2D.
    slicers: список кортежей slice-объектов,
             каждый sl ≃ (slice(...), slice(...), slice(...), slice(...)) (или без первой размерности, если 2D).
    use_gaussian: True, если хотим применять гауссово-диффузный вес на патчы.
    patch_size: соответствующий размер (D_patch, H_patch, W_patch) для compute_gaussian.
    allow_tqdm, verbose: флаги вывода прогресса.
    Возвращает: predicted_logits — np.ndarray формы (C_out, D, H, W) (или (C_out, H, W)),
                 где C_out = self.label_manager.num_segmentation_heads.
    """

    data = torch.from_numpy(data)
    data = data.to(results_device)
    predicted_logits = n_predictions = prediction = gaussian = workon = None

    try:
        predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]),
                                       dtype=torch.half,
                                       device=results_device)
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

        if use_gaussian:
            gaussian = compute_gaussian(patch_size, sigma_scale=1. / 8,
                                        value_scaling_factor=10,
                                        device=results_device)
        else:
            gaussian = 1.0

        for sl in slicers:
            workon = data[sl][None]
            if mode == '2d':
                workon = workon.squeeze(2)
            workon = workon.to(results_device)

            prediction = _internal_maybe_mirror_and_predict(workon, network.to(results_device))[0].to(results_device)

            if use_gaussian:
                prediction = prediction * gaussian
            predicted_logits[sl] += prediction
            n_predictions[sl[1:]] += gaussian

        predicted_logits /= n_predictions

        if torch.any(torch.isinf(predicted_logits)):
            raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                               'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                               'predicted_logits to fp32')

    except Exception as e:
        del predicted_logits, n_predictions, prediction, gaussian, workon
        raise e
    return predicted_logits

@torch.inference_mode()
def _internal_maybe_mirror_and_predict(x: torch.Tensor, network) -> torch.Tensor:
    """
    TTA-увеличение через зеркалирование для NumPy.
    x: np.ndarray формы (C_in, D, H, W) или (C_in, H, W).
    Возвращает: prediction: np.ndarray формы (C_out, D, H, W) или (C_out, H, W).
    Предполагаем, что self.network принимает и возвращает np.ndarray.
    """
    network.eval()
    with torch.no_grad():
        prediction = network(x)
    return prediction


def compute_gaussian(
    tile_size: Union[Tuple[int, ...], List[int]],
    sigma_scale: float = 1. / 8,
    value_scaling_factor: float = 1.0,
    dtype=torch.float16,
    device='cpu'
) -> torch.Tensor:
    """
    Вычисляет N-мерную «гауссову» карту важности (importance map) размерности tile_size.

    Алгоритм:
    1. Создаёт массив tmp нулей формы tile_size и ставит «единицу» в центре.
    2. Применяет scipy.ndimage.gaussian_filter с сигмами = (tile_size[i] * sigma_scale).
    3. Нормализует так, чтобы максимум стал value_scaling_factor.
    4. Меняет все нулевые значения (если они остались) на минимальное ненулевое значение.

    :param tile_size: кортеж или список из целых — размерность создаваемой карты, напр. (Pz, Py, Px).
    :param sigma_scale: масштаб для расчёта сигм: sigma[i] = tile_size[i] * sigma_scale.
    :param value_scaling_factor: после применения фильтра минимум будет 0, максимум —
                                 значение, равное value_scaling_factor.
    :return: numpy.ndarray формы tile_size с отмасштабированной «гауссовой» картой.
    """

    # 1) Создаём массив нулей и ставим 1 в центре
    tmp = np.zeros(tile_size, dtype=np.float32)
    center_coords = tuple(i // 2 for i in tile_size)
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[center_coords] = 1.0

    gaussian_importance_map = gaussian_filter(tmp, sigma=sigmas, mode='constant', cval=0.0)

    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
    gaussian_importance_map /= (torch.max(gaussian_importance_map) / value_scaling_factor)
    gaussian_importance_map = gaussian_importance_map.to(device=device, dtype=dtype)

    mask = gaussian_importance_map == 0
    gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])
    return gaussian_importance_map

def internal_get_sliding_window_slicers(image_size: Tuple[int, ...], patch_size=[128, 96, 96], tile_step_size=0.5):
    slicers = []
    if len(patch_size) < len(image_size):
        assert len(patch_size) == len(
            image_size) - 1, 'if tile_size has less entries than image_size, ' \
                             'len(tile_size) ' \
                             'must be one shorter than len(image_size) ' \
                             '(only dimension ' \
                             'discrepancy of 1 allowed).'
        steps = compute_steps_for_sliding_window(image_size[1:], patch_size,
                                                 tile_step_size)

        for d in range(image_size[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicers.append(
                        tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                 zip((sx, sy), patch_size)]]))
    else:
        steps = compute_steps_for_sliding_window(image_size, patch_size,
                                                 tile_step_size)

        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicers.append(
                        tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                              zip((sx, sy, sz), patch_size)]]))
    return slicers


def compute_steps_for_sliding_window(image_size: Tuple[int, ...], tile_size: Tuple[int, ...], tile_step_size: float) -> \
        List[List[int]]:
    assert [i >= j for i, j in zip(image_size, tile_size)], "image size must be as large or larger than patch_size"
    assert 0 < tile_step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = [i * tile_step_size for i in tile_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)]

    steps = []
    for dim in range(len(tile_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps