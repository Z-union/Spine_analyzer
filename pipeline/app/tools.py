import numpy as np
import nibabel as nib
from dicom_io import SpinalScan
from skimage.transform import resize
from typing import Union, List, Tuple



def resize_fn(image: np.ndarray, shape: Tuple[int, ...], order: int, mode: str) -> np.ndarray:
    """
    Изменяет размер изображения с помощью skimage.transform.resize.

    :param image: Входное изображение (np.ndarray)
    :param shape: Новая форма
    :param order: Порядок интерполяции
    :param mode: Режим обработки краёв
    :return: Изображение нового размера
    """
    return resize(image, shape, order=order, mode=mode, cval=0, clip=True, anti_aliasing=False)


def resample(image: np.ndarray, label: np.ndarray, spacing: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ресемплирует изображение и (опционально) метку под заданное spacing.

    :param image: Входное изображение
    :param label: Метка (или None)
    :param spacing: Желаемое пространственное разрешение
    :return: Кортеж (ресемплированное изображение, метка)
    """
    shape = calculate_new_shape(spacing, image.shape)
    if check_anisotrophy(spacing):
        image = resample_anisotrophic_image(image, shape)
        if label is not None:
            label = resample_anisotrophic_label(label, shape)
    else:
        image = resample_regular_image(image, shape)
        if label is not None:
            label = resample_regular_label(label, shape)
    image = image.astype(np.float32)
    if label is not None:
        label = label.astype(np.uint8)
    return image, label


def calculate_new_shape(spacing: List[float], shape: Tuple[int, ...]) -> List[int]:
    """
    Вычисляет новую форму изображения для заданного spacing.

    :param spacing: Желаемое разрешение
    :param shape: Исходная форма
    :return: Новая форма (list[int])
    """
    spacing_ratio = np.array(spacing) / np.array([1.0, 1.0, 1.0])
    new_shape = (spacing_ratio * np.array(shape)).astype(int).tolist()
    return new_shape


def check_anisotrophy(spacing: List[float]) -> bool:
    """
    Проверяет, является ли spacing анизотропным (отношение max/min >= 3).

    :param spacing: Вектор spacing
    :return: True, если анизотропия выражена
    """
    def check(spacing):
        return np.max(spacing) / np.min(spacing) >= 3
    return check(spacing) or check([1.0, 1.0, 1.0])


def resample_anisotrophic_image(image: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Ресемплирует анизотропное изображение (сначала по плоскости, потом по глубине).

    :param image: Входное изображение (3D или 4D)
    :param shape: Новая форма
    :return: Ресемплированное изображение
    """
    if image.ndim == 4:  # (C, D, H, W)
        resized_channels = []
        for image_c in image:
            resized = [resize_fn(i, shape[1:], 3, "edge") for i in image_c]
            resized = np.stack(resized, axis=0)
            resized = resize_fn(resized, shape, 0, "constant")
            resized_channels.append(resized)
        resized = np.stack(resized_channels, axis=0)
    elif image.ndim == 3:  # (D, H, W)
        resized = [resize_fn(i, shape[1:], 3, "edge") for i in image]
        resized = np.stack(resized, axis=0)
        resized = resize_fn(resized, shape, 0, "constant")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return resized


def resample_regular_image(image: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Ресемплирует изотропное изображение (по всем осям одновременно).

    :param image: Входное изображение (3D или 4D)
    :param shape: Новая форма
    :return: Ресемплированное изображение
    """
    resized_channels = []
    for image_c in image:
        resized_channels.append(resize_fn(image_c, shape, 3, "edge"))
    resized = np.stack(resized_channels, axis=0)
    return resized


def resample_anisotrophic_label(label: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Ресемплирует анизотропную карту меток (сначала по плоскости, потом по глубине, с one-hot логикой).

    :param label: Входная карта меток
    :param shape: Новая форма
    :return: Ресемплированная карта меток
    """
    depth = label.shape
    reshaped = np.zeros(shape, dtype=np.uint8)
    shape_2d = shape
    reshaped_2d = np.zeros((depth, *shape_2d), dtype=np.uint8)
    n_class = np.max(label)
    for class_ in range(1, n_class + 1):
        for depth_ in range(depth):
            mask = label[0, depth_] == class_
            resized_2d = resize_fn(mask.astype(float), shape_2d, 1, "edge")
            reshaped_2d[depth_][resized_2d >= 0.5] = class_
    for class_ in range(1, n_class + 1):
        mask = reshaped_2d == class_
        resized = resize_fn(mask.astype(float), shape, 0, "constant")
        reshaped[resized >= 0.5] = class_
    reshaped = np.expand_dims(reshaped, 0)
    return reshaped


def resample_regular_label(label: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Ресемплирует изотропную карту меток (по всем осям одновременно, с one-hot логикой).

    :param label: Входная карта меток
    :param shape: Новая форма
    :return: Ресемплированная карта меток
    """
    reshaped = np.zeros(shape, dtype=np.uint8)
    n_class = np.max(label)
    for class_ in range(1, n_class + 1):
        mask = label[0] == class_
        resized = resize_fn(mask.astype(float), shape, 1, "edge")
        reshaped[resized >= 0.5] = class_
    reshaped = np.expand_dims(reshaped, 0)
    return reshaped


def reorient_spinal_scan_to_canonical(med_datas: list) -> list:
    """
    Переориентирует список SpinalScan к канонической (LPI) системе координат.

    :param med_datas: Список объектов SpinalScan
    :return: Список переориентированных SpinalScan
    """
    return [_reorient_spinal_scan_to_lpi(med_data) for med_data in med_datas]


def _reorient_spinal_scan_to_lpi(scan: SpinalScan) -> SpinalScan:
    """
    Переориентирует SpinalScan в систему координат LPI с помощью nibabel.

    :param scan: Объект SpinalScan
    :return: Переориентированный SpinalScan
    """
    affine = scan.get_affine()
    volume = scan.volume.astype(np.float32)
    # Определяем текущую и целевую ориентации
    orig_ornt = nib.orientations.io_orientation(affine)
    target_ornt = nib.orientations.axcodes2ornt(('R', 'A', 'S'))
    # Вычисляем трансформацию и переориентируем volume
    transform = nib.orientations.ornt_transform(orig_ornt, target_ornt)
    volume_lpi = nib.orientations.apply_orientation(volume, transform)
    # Обновляем аффинную матрицу после переориентации
    affine_lpi = affine @ nib.orientations.inv_ornt_aff(transform, volume.shape)
    # Вычисляем spacing из аффинной матрицы
    voxel_sizes = np.sqrt((affine_lpi[:3, :3] ** 2).sum(axis=0))  # (X, Y, Z)
    new_pixel_spacing = voxel_sizes
    # Обновляем IOP и IPP
    new_iop = _lpi_iop()
    new_ipps = _infer_ipps_from_affine(affine_lpi, volume_lpi.shape)
    return SpinalScan(
        volume=volume_lpi,
        pixel_spacing=new_pixel_spacing,
        iop=new_iop,
        ipps=new_ipps
    )


def _lpi_iop() -> np.ndarray:
    """
    Возвращает канонический IOP для LPI-ориентации.
    Строки вдоль Y (P), столбцы вдоль X (L).
    :return: Массив IOP (6,)
    """
    row_cosines = [0, -1, 0]  # Y axis → Posterior
    col_cosines = [-1, 0, 0]  # X axis → Left
    return np.array(row_cosines + col_cosines, dtype=np.float64)


def _infer_ipps_from_affine(affine: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Восстанавливает ImagePositionPatient (IPPs) для каждого среза из аффинной матрицы.
    Предполагается аксиальное сканирование (меняется Z).

    :param affine: Аффинная матрица (4x4)
    :param shape: Форма объёма (X, Y, Z)
    :return: Массив IPP (Z, 3)
    """
    ipps = []
    for k in range(shape[2]):  # (X, Y, Z)
        corner_voxel = np.array([0, 0, k, 1])
        ipp = affine @ corner_voxel
        ipps.append(ipp[:3])
    return np.array(ipps)


def pad_nd_image(
    image: np.ndarray,
    new_shape: Tuple[int, ...] = None,
    mode: str = "constant",
    kwargs: dict = None,
    return_slicer: bool = False,
    shape_must_be_divisible_by: Union[int, Tuple[int, ...], List[int]] = None,
    step: str = "step_1"
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[slice, ...]]]:
    """
    Паддинг произвольного N-мерного массива (np.ndarray), центрирование содержимого.
    Если new_shape[i] < image.shape[i], по этой оси паддинг не делается (новый размер = старый).
    При непарном паддинге "выше" (high) добавляется на 1 элемент больше, чем "ниже" (low).

    :param image: np.ndarray, произвольной размерности.
    :param new_shape: желаемая форма выходного массива.
        - Если len(new_shape) < len(image.shape), то эти размеры считаются
          для последних осей, старшие оси оставляются без изменений.
        - Если new_shape[i] < image.shape[i], паддинг по i-й оси не выполняется (оставляем старую длину).
        Если new_shape=None, приводим new_shape = image.shape (только если shape_must_be_divisible_by задан).
    :param mode: режим заполнения для np.pad (например, "constant", "reflect", "edge" и т.д.).
    :param kwargs: дополнительные аргументы, которые будут переданы в np.pad (например, для "constant" можно задать constant_values).
    :param return_slicer: если True, вместе с падженым массивом возвращается tuple slicer —
        кортеж объектов slice, которыми можно «обрезать» результат обратно до исходного image:
            padded, slicer = pad_nd_image(img, new_shape=..., return_slicer=True)
            restored = padded[slicer]  # restored.shape == img.shape
    :param shape_must_be_divisible_by: целое или список/кортеж из int.
        Требует, чтобы итоговая форма (после выравнивания по new_shape) дополнительно
        была кратна этим числам по каждой оси. Если int, то
        приводится в список той же длины, что и число осей.
        При нехватке длины дополняется слева единицами:
            image.shape=(5,100,200), shape_must_be_divisible_by=(16,16) → [1,16,16].
    :return: если return_slicer=False, возвращается только падденый np.ndarray.
             Если return_slicer=True, возвращается (padded_array, slicer),
             где slicer — tuple(slice(start_i, end_i), ...),
             восстанавливающий исходный диапазон.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(f"pad_nd_image: ожидается np.ndarray, получил {type(image)}")

    if kwargs is None:
        kwargs = {}

    old_shape = np.array(image.shape)

    # --- 1) Подготовка shape_must_be_divisible_by ---
    if shape_must_be_divisible_by is not None:
        if isinstance(shape_must_be_divisible_by, int):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(image.shape)
        else:
            shape_must_be_divisible_by = list(shape_must_be_divisible_by)
            # Если меньше осей, дополняем слева единицами
            if len(shape_must_be_divisible_by) < len(image.shape):
                shape_must_be_divisible_by = [1] * (len(image.shape) - len(shape_must_be_divisible_by)) + shape_must_be_divisible_by
            # Если больше — просто возьмём первые len(image.shape) элементов
            if len(shape_must_be_divisible_by) > len(image.shape):
                shape_must_be_divisible_by = shape_must_be_divisible_by[-len(image.shape):]

    # --- 2) Если new_shape не задан, но задана кратность,
    #        берём за основу старый размер ---
    if new_shape is None:
        assert shape_must_be_divisible_by is not None, "new_shape и shape_must_be_divisible_by не оба не могут быть None"
        new_shape = tuple(image.shape)

    # --- 3) Если new_shape короче по рангам, «приклеиваем» старые старшие оси ---
    if len(new_shape) < len(image.shape):
        # Первые (len(image.shape)-len(new_shape)) осей оставляем из старой формы
        prefix = list(image.shape[: len(image.shape) - len(new_shape) ])
        new_shape = tuple(prefix + list(new_shape))

    # --- 4) Гарантируем, что new_shape[i] >= old_shape[i] по каждой оси ---
    # (иначе паддинг по этой оси не нужен; никогда не обрезаем)
    new_shape = np.array([ max(new_shape[i], old_shape[i]) for i in range(len(old_shape)) ])

    # --- 5) Доведение до кратности shape_must_be_divisible_by ---
    if shape_must_be_divisible_by is not None:
        # shape_must_be_divisible_by уже приведён к списку той же длины, что и число осей
        shape_must = np.array(shape_must_be_divisible_by)

        # Шаг 1: для тех осей, где new_shape кратен shape_must, отснимем один «шаг»
        for i in range(len(new_shape)):
            if new_shape[i] % shape_must[i] == 0:
                new_shape[i] -= shape_must[i]

        # Шаг 2: добиваем до следующего кратного (или оставляем тем же, если уже кратно после вычитания)
        new_shape = np.array([
            new_shape[i] + shape_must[i] - (new_shape[i] % shape_must[i])
            for i in range(len(new_shape))
        ])

    # --- 6) Вычисляем паддинг ---
    difference = new_shape - old_shape                   # сколько всего добавить по каждой оси
    pad_below  = difference // 2                          # «низу» (перед началом оси)
    pad_above  = difference // 2 + (difference % 2)       # «сверху» (после конца оси)
    pad_list   = [ [int(pad_below[i]), int(pad_above[i])] for i in range(len(old_shape)) ]

    # --- 7) Если паддинг ненулевой, применяем np.pad ---
    if any(pad_below[i] != 0 or pad_above[i] != 0 for i in range(len(old_shape))):
        # np.pad ожидает список вида [(low_dim0, high_dim0), (low_dim1, high_dim1), ...]
        res = np.pad(image, pad_list, mode=mode, **kwargs)
    else:
        # Если паддинг везде 0, просто возвращаем исходный
        res = image

    # --- 8) Если не нужно возвращать slicer, только результат ---
    if not return_slicer:
        return res

    # --- 9) Иначе создаём slicer, чтобы «обрезать» назад до old_shape ---
    #    - Каждая пара pad_list[i] = [low, high_original],
    #      а в res.shape[i] длина = old_shape[i] + pad_below[i] + pad_above[i].
    #    - Чтобы получить диапазон, где лежит оригинал, нам нужно:
    #         start = pad_below[i]
    #         end   = (res.shape[i] - pad_above[i]) == (pad_below[i] + old_shape[i])
    #
    pad_arr = np.array(pad_list, dtype=int)
    # pad_arr[:, 0] = pad_below, pad_arr[:, 1] = pad_above (в оригинальном виде)
    # Хочется заменить pad_arr[:,1] на «end index» = res.shape[i] - pad_above[i]
    ends = np.array(res.shape) - pad_arr[:, 1]
    starts = pad_arr[:, 0]
    # if step == 'step_1':
    #     slicer = (slice(0, 1, None),) + tuple(slice(int(starts[i]), int(ends[i])) for i in range(len(starts)))
    # else:
    slicer = tuple(slice(int(starts[i]), int(ends[i])) for i in range(len(starts)))

    return res, slicer

def compute_sliding_steps(image_size: Tuple[int, ...], patch_size: Tuple[int, ...], step_fraction: float = 0.5) -> List[
    List[int]]:
    steps = []
    for img_dim, patch_dim in zip(image_size, patch_size):
        max_step = img_dim - patch_dim
        if max_step <= 0:
            steps.append([0])
            continue
        num_steps = int(np.ceil(max_step / (patch_dim * step_fraction))) + 1
        actual_step = max_step / max(num_steps - 1, 1)
        steps.append([int(round(i * actual_step)) for i in range(num_steps)])
    return steps


def get_sliding_window_slicers(
        image_size: Tuple[int, ...],
        patch_size: Tuple[int, ...] = (128, 96, 96),
        step_fraction: float = 0.5
) -> List[Tuple[slice, ...]]:
    slicers = []
    steps = compute_sliding_steps(image_size, patch_size, step_fraction)
    if len(patch_size) < len(image_size):
        for d in range(image_size[0]):
            for sx in steps[0]:
                for sy in steps[1]:
                    slicers.append(
                        tuple([slice(None), d, slice(sx, sx + patch_size[0]), slice(sy, sy + patch_size[1])]))
    else:
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicers.append(tuple([slice(None), slice(sx, sx + patch_size[0]),
                                          slice(sy, sy + patch_size[1]), slice(sz, sz + patch_size[2])]))
    return slicers



