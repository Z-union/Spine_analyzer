from typing import Tuple, Union, List, Dict, Optional
import scipy.ndimage as ndi
from scipy.ndimage import binary_fill_holes
from scipy.ndimage.interpolation import map_coordinates
import numpy as np
import torch
import nibabel as nib
import SimpleITK as sitk
from skimage.transform import resize
import pandas as pd

ANISO_THRESHOLD = 3

def largest_component(
        seg: 'nib.nifti1.Nifti1Image',
        binarize: bool = False,
        dilate: int = 0,
    ) -> 'nib.nifti1.Nifti1Image':
    """
    Оставляет только крупнейшую компоненту для каждого лейбла в сегментации.

    :param seg: nib.nifti1.Nifti1Image — изображение сегментации.
    :param binarize: bool — если True, бинаризует сегментацию перед поиском компоненты.
    :param dilate: int — количество вокселей для дилатации перед поиском компоненты.
    :return: nib.nifti1.Nifti1Image — изображение с крупнейшей компонентой.
    """
    seg_data = np.asanyarray(seg.dataobj).round().astype(np.uint8)

    if binarize:
        seg_data_src = seg_data.copy()
        seg_data = (seg_data != 0).astype(np.uint8)

    binary_dilation_structure = ndi.iterate_structure(ndi.generate_binary_structure(3, 1), dilate)
    output_seg_data = np.zeros_like(seg_data)

    for l in [_ for _ in np.unique(seg_data) if _ != 0]:
        mask = seg_data == l
        if dilate > 0:
            # Дилатация
            mask_labeled, num_labels = ndi.label(ndi.binary_dilation(mask, binary_dilation_structure), np.ones((3, 3, 3)))
            # Отмена дилатации
            mask_labeled *= mask
        else:
            mask_labeled, num_labels = ndi.label(mask, np.ones((3, 3, 3)))
        # Поиск самой крупной компоненты
        label_sizes = np.bincount(mask_labeled.ravel())[1:]  # Пропускаем 0
        largest_label = label_sizes.argmax() + 1  # +1, т.к. bincount с 0
        output_seg_data[mask_labeled == largest_label] = l

    if binarize:
        output_seg_data = output_seg_data * seg_data_src

    output_seg = nib.Nifti1Image(output_seg_data, seg.affine, seg.header)
    return output_seg

def bounding_box_to_slice(bounding_box: List[List[int]]) -> Tuple[slice, ...]:
    """
    Преобразует bounding box в кортеж срезов для numpy-индексации.
    Все bounding box в acvl_utils и nnU-Netv2 — полуоткрытые интервалы [start, end)!
    :param bounding_box: Список пар [start, end] по каждому измерению.
    :return: Кортеж срезов для numpy.
    """
    return tuple([slice(*i) for i in bounding_box])

class ImageReader:
    """
    Вспомогательный класс для чтения медицинских изображений с помощью nib и извлечения метаданных.
    """
    @staticmethod
    def read_nifti(nib_image: 'nib.nifti1.Nifti1Image') -> Tuple[np.ndarray, Dict]:
        """
        Читает объект nib Nifti1Image и возвращает данные изображения и свойства.

        :param nib_image: nib.nifti1.Nifti1Image — NIfTI-объект изображения
        :return: Кортеж (данные, свойства), где данные — numpy-массив (C, Z, Y, X), свойства — словарь
        """
        np_image = nib_image.get_fdata().astype(np.float32)
        data = np_image.transpose(2, 1, 0) if np_image.ndim == 3 else np_image.transpose(0, 3, 2, 1)
        spacing = nib_image.header.get_zooms()
        affine = nib_image.affine
        origin = affine[:3, 3]
        direction = affine[:3, :3].flatten()

        origin = tuple((-1 * origin[i] if i in (0, 1) else origin[i]) for i in range(len(origin)))
        direction = tuple((-1 * direction[i] if i in (0, 4) else direction[i]) for i in range(len(direction)))

        spacing = tuple((float(spacing[i]) for i in range(len(spacing))))[:-1]

        properties = {
            'spacing': spacing,
            'sitk_stuff': {
                'spacing': spacing,
                'origin': origin,
                'direction': direction
            }
        }
        return data, properties

class Normalizer:
    """
    Вспомогательный класс для нормализации данных изображений.
    """
    @staticmethod
    def normalize(image: np.ndarray) -> np.ndarray:
        """
        Нормализует изображение до нулевого среднего и единичного стандартного отклонения.

        :param image: Входной массив изображения
        :return: Нормализованный массив изображения
        """
        image = image.astype(np.float32, copy=False)
        for idx in range(image.shape[0]):
            mean = image[idx].mean()
            std = image[idx].std()
            image[idx] -= mean
            image[idx] /= (max(std, 1e-8))
        return image

class MaskUtils:
    """
    Вспомогательный класс для операций с масками: создание маски ненулевых значений и bounding box.
    """
    @staticmethod
    def create_nonzero_mask(data: np.ndarray) -> np.ndarray:
        """
        Создаёт бинарную маску, где данные ненулевые по всем каналам.

        :param data: Входной массив (C, X, Y, Z) или (C, X, Y)
        :return: Бинарная маска ненулевых областей
        """
        assert data.ndim in (3, 4), "data должен иметь форму (C, X, Y, Z) или (C, X, Y)"
        nonzero_mask = data[0] != 0
        for c in range(1, data.shape[0]):
            nonzero_mask |= data[c] != 0
        return binary_fill_holes(nonzero_mask)

    @staticmethod
    def get_bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
        """
        Вычисляет bounding box бинарной маски.

        :param mask: Бинарная маска
        :return: Bounding box в формате [[z_start, z_end], [x_start, x_end], [y_start, y_end]]
        """
        Z, X, Y = mask.shape
        minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
        zidx = list(range(Z))
        for z in zidx:
            if np.any(mask[z]):
                minzidx = z
                break
        for z in zidx[::-1]:
            if np.any(mask[z]):
                maxzidx = z + 1
                break
        xidx = list(range(X))
        for x in xidx:
            if np.any(mask[:, x]):
                minxidx = x
                break
        for x in xidx[::-1]:
            if np.any(mask[:, x]):
                maxxidx = x + 1
                break
        yidx = list(range(Y))
        for y in yidx:
            if np.any(mask[:, :, y]):
                minyidx = y
                break
        for y in yidx[::-1]:
            if np.any(mask[:, :, y]):
                maxyidx = y + 1
                break
        return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

class Resampler:
    """
    Класс для ресемплирования медицинских изображений и сегментаций.
    """
    @staticmethod
    def compute_new_shape(
        old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
        old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
        new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]
    ) -> np.ndarray:
        """
        Вычисляет новую форму изображения при изменении пространственного разрешения.

        :param old_shape: Исходная форма (размеры) изображения
        :param old_spacing: Исходное пространственное разрешение
        :param new_spacing: Новое пространственное разрешение
        :return: Новая форма изображения
        """
        return np.round(np.array(old_shape) * np.array(old_spacing) / np.array(new_spacing)).astype(int)

    @staticmethod
    def get_do_separate_z(
        spacing: Union[Tuple[float, ...], List[float], np.ndarray],
        anisotropy_threshold=ANISO_THRESHOLD
    ) -> bool:
        """
        Определяет, требуется ли отдельное ресемплирование по оси Z из-за анизотропии.

        :param spacing: Вектор разрешения
        :param anisotropy_threshold: Порог анизотропии
        :return: True, если требуется отдельное ресемплирование по Z
        """
        return (np.max(spacing) / np.min(spacing)) > anisotropy_threshold

    @staticmethod
    def get_lowres_axis(new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
        """
        Находит ось с наименьшим разрешением (наибольшим spacing).

        :param new_spacing: Вектор разрешения
        :return: Индексы осей с низким разрешением
        """
        axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]
        return axis

    @staticmethod
    def resample_data_or_seg(
        data: np.ndarray,
        new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
        is_seg: bool = False,
        axis: Union[None, int] = None,
        order: int = 3,
        do_separate_z: bool = False,
        order_z: int = 0,
        dtype_out=None
    ) -> np.ndarray:
        """
        Ресемплирует изображение или сегментацию до новой формы.

        :param data: Входные данные (C, X, Y, Z)
        :param new_shape: Целевая форма (X, Y, Z)
        :param is_seg: True — если данные являются сегментацией
        :param axis: Ось для отдельного ресемплирования по Z
        :param order: Порядок интерполяции для плоскости
        :param do_separate_z: Ресемплировать ли отдельно по Z
        :param order_z: Порядок интерполяции по оси Z
        :param dtype_out: Тип выходных данных
        :return: Ресемплированные данные
        """
        assert data.ndim == 4, "data must be (c, x, y, z)"
        assert len(new_shape) == data.ndim - 1
        if is_seg:
            resize_fn = DefaultPreprocessor.resize_segmentation
            kwargs = {}
        else:
            resize_fn = resize
            kwargs = {'mode': 'edge', 'anti_aliasing': False}
        shape = np.array(data[0].shape)
        new_shape = np.array(new_shape)
        if dtype_out is None:
            dtype_out = data.dtype
        reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
        if np.any(shape != new_shape):
            data = data.astype(float, copy=False)
            if do_separate_z:
                assert len(axis) == 1, "only one anisotropic axis supported"
                axis = axis[0]
                if axis == 0:
                    new_shape_2d = new_shape[1:]
                elif axis == 1:
                    new_shape_2d = new_shape[[0, 2]]
                else:
                    new_shape_2d = new_shape[:-1]
                for c in range(data.shape[0]):
                    reshaped_here = np.zeros((data.shape[1], *new_shape_2d))
                    for slice_id in range(shape[axis]):
                        if axis == 0:
                            reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs)
                        elif axis == 1:
                            reshaped_here[slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs)
                        else:
                            reshaped_here[slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs)
                    if shape[axis] != new_shape[axis]:
                        rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                        orig_rows, orig_cols, orig_dim = reshaped_here.shape
                        row_scale = float(orig_rows) / rows
                        col_scale = float(orig_cols) / cols
                        dim_scale = float(orig_dim) / dim
                        map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                        map_rows = row_scale * (map_rows + 0.5) - 0.5
                        map_cols = col_scale * (map_cols + 0.5) - 0.5
                        map_dims = dim_scale * (map_dims + 0.5) - 0.5
                        coord_map = np.array([map_rows, map_cols, map_dims])
                        if not is_seg or order_z == 0:
                            reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode='nearest')[None]
                        else:
                            unique_labels = np.sort(pd.unique(reshaped_here.ravel()))
                            for cl in unique_labels:
                                reshaped_final[c][np.round(
                                    map_coordinates((reshaped_here == cl).astype(float), coord_map, order=order_z,
                                                    mode='nearest')) > 0.5] = cl
                    else:
                        reshaped_final[c] = reshaped_here
            else:
                for c in range(data.shape[0]):
                    reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
            return reshaped_final
        else:
            return data


def _resample_data_or_seg_to_shape(
    data: Union[torch.Tensor, np.ndarray],
    new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
    is_seg: bool = False,
    order: int = 3,
    order_z: int = 0,
    force_separate_z: Union[bool, None] = False,
    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD
) -> np.ndarray:
    """
    Ресемплирует изображение или сегментацию до новой формы и разрешения.

    :param data: Входные данные (np.ndarray или torch.Tensor)
    :param new_shape: Целевая форма
    :param current_spacing: Текущее разрешение
    :param new_spacing: Целевое разрешение
    :param is_seg: True — если данные являются сегментацией
    :param order: Порядок интерполяции для плоскости
    :param order_z: Порядок интерполяции по оси Z
    :param force_separate_z: Принудительно разделять ресемплирование по Z
    :param separate_z_anisotropy_threshold: Порог анизотропии
    :return: Ресемплированные данные
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if force_separate_z is not None:
        do_separate_z = force_separate_z
        if force_separate_z:
            axis = Resampler.get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        if Resampler.get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = Resampler.get_lowres_axis(current_spacing)
        elif Resampler.get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = Resampler.get_lowres_axis(new_spacing)
        else:
            do_separate_z = False
            axis = None
    if axis is not None:
        if len(axis) == 3:
            do_separate_z = False
        elif len(axis) == 2:
            do_separate_z = False
    if data is not None:
        assert data.ndim == 4, "data must be c x y z"
    data_reshaped = Resampler.resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped


class DefaultPreprocessor:
    """
    Основной класс для нормализации и ресемплирования медицинских изображений.
    Обеспечивает чтение, нормализацию и ресемплирование изображений и сегментаций.
    """
    def __init__(self, verbose: bool = True):
        """
        Инициализация препроцессора.

        :param verbose: True — если нужен подробный вывод
        """
        self.verbose = verbose
        self.properties = None
        self.transpose_forward = None

    def run_case_npy(
        self,
        data: np.ndarray,
        seg: Optional[np.ndarray],
        properties: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обрабатывает один случай (изображение и, опционально, сегментацию) в формате numpy.
        Применяет нормализацию и ресемплирование.

        :param data: Входные данные изображения
        :param seg: Сегментация или None
        :param properties: Свойства изображения (spacing и др.)
        :return: Кортеж (обработанное изображение, обработанная сегментация)
        """
        data = data.astype(np.float32)
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation."
            seg = np.copy(seg)
        data = data if data.ndim == 4 else data[np.newaxis, ...]
        data = data.transpose([0, *[i + 1 for i in self.transpose_forward]])
        if seg is not None:
            seg = seg if seg.ndim == 4 else seg[np.newaxis, ...]
            seg = seg.transpose([0, *[i + 1 for i in self.transpose_forward]])
        original_spacing = [self.properties['spacing'][i] for i in self.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        self.properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = self._crop_to_nonzero(data, seg)
        self.properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        self.properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        target_spacing = properties['spacing']
        if len(target_spacing) < len(data.shape[1:]):
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = Resampler.compute_new_shape(data.shape[1:], original_spacing, target_spacing)
        data = Normalizer.normalize(data)
        data = _resample_data_or_seg_to_shape(data, new_shape, original_spacing, target_spacing)
        seg = _resample_data_or_seg_to_shape(seg, new_shape, original_spacing, target_spacing) if seg is not None else None
        if seg is not None:
            if np.max(seg) > 127:
                seg = seg.astype(np.int16)
            else:
                seg = seg.astype(np.int8)
        return data, seg

    def run_case(self, image, transpose_forward, seg=None):
        """
        Обрабатывает один случай из nib-изображения (и, опционально, сегментации): нормализация и ресемплирование.

        :param image: Входное изображение (nib.nifti1.Nifti1Image)
        :param seg: Опциональная сегментация (nib.nifti1.Nifti1Image)
        :return: Кортеж (обработанное изображение, сегментация, свойства)
        """
        data, data_properties = ImageReader.read_nifti(image)
        self.properties = data_properties
        self.transpose_forward = transpose_forward
        if seg is not None:
            seg, _ = ImageReader.read_nifti(seg)
        else:
            seg = None
        data, seg = self.run_case_npy(data, seg, data_properties)
        return data, seg, data_properties

    @staticmethod
    def resize_segmentation(segmentation: np.ndarray, new_shape: tuple, order: int = 3) -> np.ndarray:
        """
        Изменяет размер карты сегментации с помощью one-hot кодирования для предотвращения артефактов интерполяции.

        :param segmentation: Входная карта сегментации
        :param new_shape: Целевая форма
        :param order: Порядок интерполяции
        :return: Сегментация нового размера
        """
        tpe = segmentation.dtype
        assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
        if order == 0:
            return resize(segmentation.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False).astype(tpe)
        else:
            reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
            unique_labels = np.sort(pd.unique(segmentation.ravel()))
            for c in unique_labels:
                mask = segmentation == c
                reshaped_multihot = resize(mask.astype(float), new_shape, order, mode="edge", clip=True, anti_aliasing=False)
                reshaped[reshaped_multihot >= 0.5] = c
            return reshaped

    def _crop_to_nonzero(self, data: np.ndarray, seg: Optional[np.ndarray] = None, nonzero_label: int = -1):
        """
        Обрезает изображение и сегментацию по маске ненулевых значений.

        :param data: Входные данные
        :param seg: Сегментация или None
        :param nonzero_label: Значение для областей вне маски
        :return: Кортеж (обрезанные данные, сегментация, bounding box)
        """
        nonzero_mask = self._create_nonzero_mask(data)
        bbox = self._get_bbox_from_mask(nonzero_mask)
        slicer = self._bounding_box_to_slice(bbox)
        nonzero_mask = nonzero_mask[slicer][None]

        slicer = (slice(None),) + slicer
        data = data[slicer]
        if seg is not None:
            seg = seg[slicer]
            seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
        else:
            seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
        return data, seg, bbox

    @staticmethod
    def _create_nonzero_mask(data: np.ndarray) -> np.ndarray:
        """
        Создаёт бинарную маску, где данные ненулевые по всем каналам.

        :param data: Входной массив (C, X, Y, Z) или (C, X, Y)
        :return: Бинарная маска ненулевых областей
        """
        assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
        nonzero_mask = data[0] != 0
        for c in range(1, data.shape[0]):
            nonzero_mask |= data[c] != 0
        return binary_fill_holes(nonzero_mask)

    @staticmethod
    def _get_bbox_from_mask(mask: np.ndarray) -> List[List[int]]:
        """
        Вычисляет bounding box бинарной маски (полуоткрытый интервал [start, end)).

        :param mask: Бинарная маска
        :return: Bounding box в формате [[z_start, z_end], [x_start, x_end], [y_start, y_end]]
        """
        Z, X, Y = mask.shape
        minzidx, maxzidx, minxidx, maxxidx, minyidx, maxyidx = 0, Z, 0, X, 0, Y
        zidx = list(range(Z))
        for z in zidx:
            if np.any(mask[z]):
                minzidx = z
                break
        for z in zidx[::-1]:
            if np.any(mask[z]):
                maxzidx = z + 1
                break
        xidx = list(range(X))
        for x in xidx:
            if np.any(mask[:, x]):
                minxidx = x
                break
        for x in xidx[::-1]:
            if np.any(mask[:, x]):
                maxxidx = x + 1
                break
        yidx = list(range(Y))
        for y in yidx:
            if np.any(mask[:, :, y]):
                minyidx = y
                break
        for y in yidx[::-1]:
            if np.any(mask[:, :, y]):
                maxyidx = y + 1
                break
        return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

    @staticmethod
    def _bounding_box_to_slice(bounding_box: List[List[int]]) -> Tuple[slice, ...]:
        """
        Преобразует bounding box в кортеж срезов для numpy-индексации.
        :param bounding_box: Список пар [start, end] по каждому измерению.
        :return: Кортеж срезов для numpy.
        """
        return tuple([slice(*i) for i in bounding_box])

    def convert_predicted_logits_to_segmentation_with_correct_shape(
        self,
        predicted_logits: Union[torch.Tensor, np.ndarray],
        num_threads_torch: int = 2
    ):
        """
        Преобразует предсказанные логиты в сегментацию с восстановлением исходной формы изображения.

        :param predicted_logits: Логиты (torch.Tensor или np.ndarray)
        :param num_threads_torch: Количество потоков для torch
        :return: Сегментация исходной формы
        """
        old_threads = torch.get_num_threads()
        torch.set_num_threads(num_threads_torch)
        # ресемплируем к исходной форме
        current_spacing = self.properties['spacing'] if \
            len(self.properties['spacing']) == \
            len(self.properties['shape_after_cropping_and_before_resampling']) else \
            [self.properties['spacing'][0], *self.properties['spacing']]
        predicted_logits = _resample_data_or_seg_to_shape(
            predicted_logits,
            self.properties['shape_after_cropping_and_before_resampling'],
            current_spacing,
            self.properties['spacing']
        )
        # возвращаем вероятности
        predicted_probabilities = self._apply_inference_nonlin(predicted_logits)
        del predicted_logits
        segmentation = self.convert_probabilities_to_segmentation(predicted_probabilities)
        # переводим в numpy
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().numpy()
        # возвращаем сегментацию в bounding box (отмена кропа)
        segmentation_reverted_cropping = np.zeros(self.properties['shape_before_cropping'], dtype=np.uint8)
        slicer = self._bounding_box_to_slice(self.properties['bbox_used_for_cropping'])
        segmentation_reverted_cropping[slicer] = segmentation
        del segmentation
        predicted_probabilities = self._revert_cropping_on_probabilities(predicted_probabilities,
                                                                         self.properties[
                                                                             'bbox_used_for_cropping'],
                                                                         self.properties[
                                                                             'shape_before_cropping'])
        predicted_probabilities = predicted_probabilities.cpu().numpy()
        # revert transpose
        predicted_probabilities = predicted_probabilities.transpose([0] + [i + 1 for i in
                                                                           [0, 1, 2]])
        torch.set_num_threads(old_threads)

        segmentation_reverted_cropping = self._format_seg(segmentation_reverted_cropping)

        return segmentation_reverted_cropping, predicted_probabilities

    def _format_seg(self, seg: np.ndarray):
        spacing = tuple(float(x) for x in self.properties['sitk_stuff']['spacing'])
        origin = tuple(float(x) for x in self.properties['sitk_stuff']['origin'])
        direction = tuple(float(x) for x in self.properties['sitk_stuff']['direction'])

        # Формируем direction-матрицу (обычно 3x3)
        direction_matrix = np.array(direction).reshape(3, 3)

        # Формируем affine: direction * diag(spacing), затем добавляем origin
        affine = np.eye(4)
        affine[:3, :3] = direction_matrix * spacing  # direction по столбцам, умножить на spacing
        affine[:3, 3] = origin
        # seg = seg.transpose(1, 0, 2)

        # Создаём Nifti1Image
        seg_img = nib.Nifti1Image(seg.astype(np.uint8), affine)

        return seg_img


    def _apply_inference_nonlin(self, logits: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Применяет нелинейность (например, сигмоиду) к логитам для получения вероятностей.
        Ожидается форма (c, x, y, (z)), где c — количество классов/регионов.

        :param logits: Логиты (np.ndarray или torch.Tensor)
        :return: Вероятности (torch.Tensor)
        """
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        with torch.no_grad():
            logits = logits.float()
            probabilities = torch.sigmoid(logits)
        return probabilities

    @staticmethod
    def convert_probabilities_to_segmentation(predicted_probabilities: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Преобразует вероятности (после нелинейности) в карту сегментации.
        Ожидается форма (c, x, y, (z)), где c — количество классов/регионов.

        :param predicted_probabilities: Вероятности (np.ndarray или torch.Tensor)
        :return: Карта сегментации (np.ndarray или torch.Tensor)
        """
        regions_class_order = [i for i in range(1, predicted_probabilities.shape[0] + 1)]
        if not isinstance(predicted_probabilities, (np.ndarray, torch.Tensor)):
            raise RuntimeError(f"Unexpected input type. Expected np.ndarray or torch.Tensor, got {type(predicted_probabilities)}")
        if isinstance(predicted_probabilities, np.ndarray):
            segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.uint16)
        else:
            segmentation = torch.zeros(predicted_probabilities.shape[1:], dtype=torch.int16, device=predicted_probabilities.device)
        for i, c in enumerate(regions_class_order):
            segmentation[predicted_probabilities[i] > 0.5] = c
        return segmentation

    def _revert_cropping_on_probabilities(
        self,
        predicted_probabilities: Union[torch.Tensor, np.ndarray],
        bbox: List[List[int]],
        original_shape: Union[List[int], Tuple[int, ...]]
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Восстанавливает вероятности после кропа до исходной формы (только для вероятностей, не для логитов и не для карт сегментации!).

        :param predicted_probabilities: Вероятности (c, x, y, (z))
        :param bbox: Bounding box, использованный для кропа
        :param original_shape: Исходная форма
        :return: Вероятности, дополненные до исходной формы
        """
        if isinstance(predicted_probabilities, np.ndarray):
            probs_reverted_cropping = np.zeros((predicted_probabilities.shape[0], *original_shape), dtype=predicted_probabilities.dtype)
        else:
            probs_reverted_cropping = torch.zeros((predicted_probabilities.shape[0], *original_shape), dtype=predicted_probabilities.dtype)
        slicer = bounding_box_to_slice(bbox)
        probs_reverted_cropping[tuple([slice(None)] + list(slicer))] = predicted_probabilities
        return probs_reverted_cropping
