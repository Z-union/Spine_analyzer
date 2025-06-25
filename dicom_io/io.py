import os
from typing import List, Union, Optional, Dict, Tuple
import glob
from dataclasses import dataclass, field

import argparse
import numpy as np
from pydicom import dcmread, FileDataset, DataElement
from pydicom.tag import Tag
import nibabel as nib


@dataclass
class SpinalScan:
    volume: np.ndarray
    pixel_spacing: Union[np.ndarray, list]
    iop: Union[np.ndarray, list]
    ipps: Union[np.ndarray, list] = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.pixel_spacing = np.asarray(self.pixel_spacing, dtype=np.float64)
        self.iop = np.asarray(self.iop, dtype=np.float64)
        self.ipps = np.asarray(self.ipps, dtype=np.float64)

    def get_affine(self) -> np.ndarray:
        """
        Construct 4x4 affine matrix from DICOM orientation and spacing metadata.

        :return: Affine matrix (4x4) for NIfTI
        :rtype: np.ndarray
        """
        row_cosines = self.iop[:3]  # direction along rows (X)
        col_cosines = self.iop[3:]  # direction along columns (Y)
        normal = np.cross(row_cosines, col_cosines)  # direction along Z (slices)

        dx, dy = self.pixel_spacing[:2]
        dz = self.pixel_spacing[2]

        affine = np.eye(4)
        affine[:3, 0] = row_cosines * dx
        affine[:3, 1] = col_cosines * dy
        affine[:3, 2] = normal * dz
        affine[:3, 3] = self.ipps[0]

        return affine

    def to_nifti(self) -> nib.Nifti1Image:
        """
        Convert to nibabel Nifti1Image.

        :return: NIfTI image object
        :rtype: nib.Nifti1Image
        """
        # Ensure volume is in (X, Y, Z) order
        vol = self.volume
        if vol.ndim == 3 and not (vol.shape[0] <= vol.shape[1] and vol.shape[1] <= vol.shape[2]):
            # Heuristic: if not (X <= Y <= Z), try to transpose
            vol = np.ascontiguousarray(vol)
            vol = np.rot90(vol, 1, axes=(0, 1))
        return nib.Nifti1Image(vol.astype(np.float32), affine=np.array(self.get_affine(), dtype=np.float64))


def load_dicoms(
    paths_sag: List[Union[os.PathLike, str, bytes]],
    paths_ax: List[Union[os.PathLike, str, bytes]],
    require_extensions: bool = True,
    metadata_overwrites: Optional[dict] = None,
) -> Tuple[SpinalScan, SpinalScan]:
    '''
    Создает два объекта SpinalScan — из сагиттальных и аксиальных DICOM-файлов.

    Параметры
    ----------
    paths_sag : List[Union[os.PathLike, str, bytes]]
        Список путей к сагиттальным DICOM-файлам.
    paths_ax : List[Union[os.PathLike, str, bytes]]
        Список путей к аксиальным DICOM-файлам.
    require_extensions : bool
        Если True, проверяется, что у всех путей расширение ".dcm".
    metadata_overwrites : dict
        Словарь для перезаписи метаданных в скане.

    Возвращает
    -------
    Tuple[SpinalScan, SpinalScan]
        Кортеж: (SpinalScan сагиттальный, SpinalScan аксиальный)
    '''
    if metadata_overwrites is None:
        metadata_overwrites = {}

    def process_series(paths: List[Union[os.PathLike, str, bytes]]) -> SpinalScan:
        if require_extensions:
            for path in paths:
                if not str(path).endswith(".dcm"):
                    raise ValueError(f"Файл {path} не имеет расширения .dcm")

        dicom_files = [dcmread(path) for path in paths]

        for idx, dicom_file in enumerate(dicom_files):
            dicom_files[idx] = overwrite_tags(dicom_file, metadata_overwrites)

        for dicom_idx, dicom_file in enumerate(dicom_files):
            missing_tags = check_missing_tags(dicom_file)
            if missing_tags:
                raise ValueError(f"В файле {paths[dicom_idx]} отсутствуют теги: {missing_tags}")

        # Проверка проекции
        is_sagittal = all(is_sagittal_dicom_slice(dcm) for dcm in dicom_files)
        is_axial = all(is_axial_dicom_slice(dcm) for dcm in dicom_files)

        if not (is_sagittal or is_axial):
            raise ValueError("Срезы не являются ни сагиттальными, ни аксиальными")

        dicom_files.sort(key=lambda d: d.InstanceNumber)

        pixel_spacing = [
            *np.mean([np.array(d.PixelSpacing) for d in dicom_files], axis=0),
            np.mean([float(d.SliceThickness) for d in dicom_files])
        ]

        volume = np.stack([d.pixel_array for d in dicom_files], axis=-1)

        return SpinalScan(volume=np.mean(volume, axis=2) if volume.ndim == 4 else volume, pixel_spacing=pixel_spacing,
                          iop=np.array(dicom_files[0].ImageOrientationPatient, dtype=np.float64),
                          ipps=np.array([dicom_file.ImagePositionPatient for dicom_file in dicom_files],
                                                dtype=np.float64))

    # Обработка обеих проекций
    sag_scan = process_series(paths_sag)
    ax_scan = process_series(paths_ax)

    return sag_scan, ax_scan


def is_sagittal_dicom_slice(dicom_file: FileDataset) -> bool:
    '''
    Проверяет, является ли DICOM-срез сагиттальным.

    Параметры
    ----------
    dicom_file : FileDataset
        DICOM-файл для проверки.

    Возвращает
    -------
    bool
        True, если срез сагиттальный; иначе False.
    '''
    if Tag("ImageOrientationPatient") in dicom_file:
        iop = np.array(dicom_file.ImageOrientationPatient).astype(float)
        row_cosine = iop[:3]
        col_cosine = iop[3:]
        normal = np.cross(row_cosine, col_cosine).round()
        return np.allclose(np.abs(normal), [1, 0, 0])
    else:
        raise ValueError("Метаданные ImageOrientationPatient отсутствуют в DICOM-файле")


def is_axial_dicom_slice(dicom_file: FileDataset) -> bool:
    '''
    Проверяет, является ли DICOM-срез аксиальным.

    Параметры
    ----------
    dicom_file : FileDataset
        DICOM-файл для проверки.

    Возвращает
    -------
    bool
        True, если срез аксиальный; иначе False.
    '''
    if Tag("ImageOrientationPatient") in dicom_file:
        iop = np.array(dicom_file.ImageOrientationPatient).astype(float)
        row_cosine = iop[:3]
        col_cosine = iop[3:]
        normal = np.cross(row_cosine, col_cosine).round()
        return np.allclose(np.abs(normal), [0, 0, 1])
    else:
        raise ValueError("Метаданные ImageOrientationPatient отсутствуют в DICOM-файле")


def overwrite_tags(dicom_file: FileDataset, metadata_overwrites: dict) -> FileDataset:
    '''
    Перезаписывает теги в DICOM-файле. В данный момент поддерживается перезапись:
    PixelSpacing, SliceThickness и ImageOrientationPatient.

    Параметры
    ----------
    dicom_file : FileDataset
        DICOM-файл, в котором нужно изменить значения.
    metadata_overwrites : dict
        Словарь метаданных для перезаписи (например, PixelSpacing, SliceThickness, ImageOrientationPatient).

    Возвращает
    -------
    FileDataset
        DICOM-файл с изменёнными значениями.
    '''
    possible_overwrites = {
        "PixelSpacing": "DS",
        "SliceThickness": "DS",
        "ImageOrientationPatient": "DS",
    }

    for tag, value in metadata_overwrites.items():
        if tag not in possible_overwrites:
            raise NotImplementedError(f"Перезапись тега {tag} не поддерживается")
        else:
            if Tag(tag) in dicom_file:
                dicom_file[Tag(tag)] = DataElement(
                    Tag(tag), possible_overwrites[tag], value
                )
            else:
                dicom_file.add_new(Tag(tag), possible_overwrites[tag], value)
    return dicom_file


def check_missing_tags(dicom_file: FileDataset) -> List[str]:
    '''
    Определяет, какие теги отсутствуют в DICOM-файле.
    Требуются теги: PixelData, PixelSpacing, SliceThickness, InstanceNumber.

    Параметры
    ----------
    dicom_file : FileDataset
        DICOM-файл для проверки.

    Возвращает
    -------
    List[str]
        Список отсутствующих тегов.
    '''
    required_tags = ["PixelData", "PixelSpacing", "SliceThickness", "InstanceNumber"]
    missing_tags = [
        tag_name for tag_name in required_tags if Tag(tag_name) not in dicom_file
    ]
    return missing_tags


def is_dicom_file(path: Union[os.PathLike, str, bytes]) -> bool:
    '''
    Проверяет, является ли файл DICOM-файлом.

    Параметры
    ----------
    path : Union[os.PathLike, str, bytes]
        Путь к файлу.

    Возвращает
    -------
    bool
        True, если файл DICOM, иначе False.
    '''
    try:
        dcmread(path)
        return True
    except:
        return False


def load_dicoms_from_folder(
    args: argparse.Namespace,
    require_extensions: bool = True,
    metadata_overwrites: Optional[Dict] = None,
) -> tuple[SpinalScan, SpinalScan]:
    '''
    Загружает DICOM-скан из папки, содержащей срезы.

    Параметры
    ----------
    args : argparse.Namespace
        Объект с путями к папкам: args.input_sag и args.input_ax.
    require_extensions : bool
        Если True, проверяет, что все DICOM-файлы имеют расширение .dcm.
    metadata_overwrites : dict
        Словарь метаданных для перезаписи.

    Возвращает
    -------
    SpinalScan
        Объект, представляющий скан позвоночника, сформированный из DICOM-файлов.
    '''
    if metadata_overwrites is None:
        metadata_overwrites = {}

    slices_sag = [f for f in glob.glob(os.path.join(args.input_sag, "*")) if is_dicom_file(f)]
    slices_ax = [f for f in glob.glob(os.path.join(args.input_ax, "*")) if is_dicom_file(f)]

    return load_dicoms(slices_sag, slices_ax, require_extensions, metadata_overwrites)


def load_study_dicoms(
    study_folder: str,
    require_extensions: bool = True,
    metadata_overwrites: Optional[Dict] = None,
) -> Tuple[Tuple[SpinalScan, SpinalScan], list]:
    '''
    Загружает DICOM-сканы на уровне исследования (study):
    - выбирает аксиальную и сагиттальную серии
    - строит таблицу соответствия срезов по координатам

    Параметры
    ----------
    study_folder : str
        Путь к папке с DICOM-файлами исследования.
    require_extensions : bool
        Проверять ли расширение .dcm.
    metadata_overwrites : dict
        Словарь для перезаписи метаданных.

    Возвращает
    -------
    Tuple[SpinalScan, SpinalScan, list]
        (SpinalScan сагиттальный, SpinalScan аксиальный, таблица соответствия)
    '''
    if metadata_overwrites is None:
        metadata_overwrites = {}

    # Рекурсивно собрать все DICOM-файлы
    dicom_paths = [f for f in glob.glob(os.path.join(study_folder, '**', '*'), recursive=True) if os.path.isfile(f) and is_dicom_file(f)]
    if not dicom_paths:
        raise ValueError(f"В папке {study_folder} не найдено DICOM-файлов")

    # Группировка по SeriesInstanceUID
    series_dict = {}
    for path in dicom_paths:
        try:
            dcm = dcmread(path, stop_before_pixels=True)
            series_uid = getattr(dcm, 'SeriesInstanceUID', None)
            if series_uid is not None:
                series_dict.setdefault(series_uid, []).append(path)
        except Exception as e:
            continue

    # Определить типы серий (только чистые серии одной проекции)
    sag_paths, ax_paths = None, None
    for uid, paths in series_dict.items():
        try:
            sagittal_count = 0
            axial_count = 0
            for p in paths:
                try:
                    dcm = dcmread(p, stop_before_pixels=True)
                    if is_sagittal_dicom_slice(dcm):
                        sagittal_count += 1
                    elif is_axial_dicom_slice(dcm):
                        axial_count += 1
                except Exception:
                    continue
            if sagittal_count == len(paths) and sag_paths is None:
                sag_paths = paths
                print(f"Выбрана сагиттальная серия: {uid} ({len(paths)} файлов)")
            elif axial_count == len(paths) and ax_paths is None:
                ax_paths = paths
                print(f"Выбрана аксиальная серия: {uid} ({len(paths)} файлов)")
            # Если есть смешение — пропускаем
        except Exception as e:
            print(f"Ошибка при анализе серии {uid}: {e}")
            continue
        if sag_paths and ax_paths:
            break

    if sag_paths is None or ax_paths is None:
        raise ValueError("Не удалось найти обе серии: аксиальную и сагиттальную")

    # Загрузить SpinalScan через существующий пайплайн
    sag_scan, ax_scan = load_dicoms(sag_paths, ax_paths, require_extensions, metadata_overwrites)

    # Сопоставление срезов по координатам (используем Z-координату)
    sag_ipps = np.array([dcmread(p, stop_before_pixels=True).ImagePositionPatient for p in sag_paths])
    ax_ipps = np.array([dcmread(p, stop_before_pixels=True).ImagePositionPatient for p in ax_paths])
    sag_z = sag_ipps[:, 2]
    ax_z = ax_ipps[:, 2]
    correspondence = []
    for i, sz in enumerate(sag_z):
        # Найти ближайший аксиальный срез по Z
        j = np.argmin(np.abs(ax_z - sz))
        distance = np.abs(ax_z[j] - sz)
        correspondence.append((i, j, sz, ax_z[j], distance))

    return (sag_scan, ax_scan), correspondence
