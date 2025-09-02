import os
from typing import List, Union, Optional, Dict, Tuple, Any
import glob
from dataclasses import dataclass, field

import argparse
import numpy as np
from pydicom import dcmread, FileDataset, DataElement
from pydicom.tag import Tag
from nibabel.nifti1 import Nifti1Image
from dicomweb_client.api import DICOMwebClient
import io
import requests


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

    def to_nifti(self) -> Nifti1Image:
        """
        Convert to nibabel Nifti1Image.

        :return: NIfTI image object
        :rtype: Nifti1Image
        """
        # Ensure volume is in (X, Y, Z) order
        vol = self.volume
        if vol.ndim == 3 and not (vol.shape[0] <= vol.shape[1] and vol.shape[1] <= vol.shape[2]):
            # Heuristic: if not (X <= Y <= Z), try to transpose
            vol = np.ascontiguousarray(vol)
            vol = np.rot90(vol, 1, axes=(0, 1))
        return Nifti1Image(vol.astype(np.float32), affine=np.array(self.get_affine(), dtype=np.float64))


def get_sequence_type(dicom_file: FileDataset) -> Optional[str]:
    """
    Определяет тип последовательности (T1, T2, STIR) из DICOM метаданных.
    
    :param dicom_file: DICOM файл для анализа
    :return: Тип последовательности ('T1', 'T2', 'STIR') или None если не удалось определить
    """
    try:
        # Проверяем различные теги для определения типа последовательности
        sequence_name = getattr(dicom_file, 'SequenceName', '').upper()
        protocol_name = getattr(dicom_file, 'ProtocolName', '').upper()
        series_description = getattr(dicom_file, 'SeriesDescription', '').upper()
        
        # Проверяем на T1
        if any(keyword in sequence_name or keyword in protocol_name or keyword in series_description 
               for keyword in ['T1', 'T1W', 'T1-WEIGHTED', 'T1_WEIGHTED', 'T1W_', 'T1_']):
            return 'T1'
        
        # Проверяем на T2 (включая с подавлением жира)
        if any(keyword in sequence_name or keyword in protocol_name or keyword in series_description 
               for keyword in ['T2', 'T2W', 'T2-WEIGHTED', 'T2_WEIGHTED', 'T2W_', 'T2_', 'ST2W']):
            return 'T2'
        
        # Проверяем на STIR
        if any(keyword in sequence_name or keyword in protocol_name or keyword in series_description 
               for keyword in ['STIR', 'SHORT TAU INVERSION RECOVERY', 'INVERSION RECOVERY']):
            return 'STIR'
        
        # Дополнительные проверки по TE/TR
        if hasattr(dicom_file, 'EchoTime') and hasattr(dicom_file, 'RepetitionTime'):
            te = float(dicom_file.EchoTime)
            tr = float(dicom_file.RepetitionTime)
            
            # Эвристики для определения типа по TE/TR
            if tr < 1000:  # Короткий TR
                if te < 30:
                    return 'T1'
                else:
                    return 'T2'
            else:  # Длинный TR
                if te > 80:
                    return 'T2'
                else:
                    return 'T1'
        
        return None
        
    except Exception:
        return None


def process_single_series(paths: List[Any], 
                         require_extensions: bool = True,
                         metadata_overwrites: Optional[dict] = None,
                         dicomweb: bool = False,
                         dicomweb_client: Optional[DICOMwebClient] = None) -> Optional[SpinalScan]:
    """
    Обрабатывает одну серию DICOM-файлов и возвращает SpinalScan.
    
    :param paths: Список путей к DICOM-файлам или кортежей (study_uid, series_uid, sop_uid) для dicomweb
    :param require_extensions: Проверять расширение .dcm
    :param metadata_overwrites: Словарь для перезаписи метаданных
    :param dicomweb: Если True, использовать DICOMweb
    :param dicomweb_client: Экземпляр DICOMwebClient
    :return: SpinalScan или None если обработка не удалась
    """
    if metadata_overwrites is None:
        metadata_overwrites = {}
    
    if not paths:
        return None
        
    try:
        if dicomweb:
            # paths — список кортежей (study_uid, series_uid, sop_uid)
            dicom_files = [download_instance(dicomweb_client, *p) for p in paths]
        else:
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

        dicom_files.sort(key=lambda d: d.InstanceNumber)

        pixel_spacing = [
            *np.mean([np.array(d.PixelSpacing) for d in dicom_files], axis=0),
            np.mean([float(d.SliceThickness) for d in dicom_files])
        ]

        volume = np.stack([d.pixel_array for d in dicom_files], axis=-1)

        return SpinalScan(volume=np.mean(volume, axis=2) if volume.ndim == 4 else volume, 
                         pixel_spacing=pixel_spacing,
                         iop=np.array(dicom_files[0].ImageOrientationPatient, dtype=np.float64),
                         ipps=np.array([dicom_file.ImagePositionPatient for dicom_file in dicom_files],
                                      dtype=np.float64))
    except Exception as e:
        print(f"Ошибка при обработке серии: {e}")
        return None


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

    slices_sag: List[Union[os.PathLike, str, bytes]] = [f for f in glob.glob(os.path.join(args.input_sag, "*")) if is_dicom_file(f)]
    slices_ax: List[Union[os.PathLike, str, bytes]] = [f for f in glob.glob(os.path.join(args.input_ax, "*")) if is_dicom_file(f)]

    return load_dicoms(slices_sag, slices_ax, require_extensions, metadata_overwrites)


def get_dicomweb_client(url, username=None, password=None):
    if username is not None and password is not None:
        session = requests.Session()
        session.auth = (username, password)
        return DICOMwebClient(url, session=session)
    else:
        return DICOMwebClient(url)

def get_study_instances(client, study_uid):
    # Получить все series
    series = client.search_for_series(search_filters={"StudyInstanceUID": study_uid})
    all_instances = []
    for s in series:
        series_uid = s['0020000E']['Value'][0]
        # Получить все instances в серии
        instances = client.search_for_instances(search_filters={
            "StudyInstanceUID": study_uid,
            "SeriesInstanceUID": series_uid
        })
        for inst in instances:
            sop_uid = inst['00080018']['Value'][0]
            all_instances.append((study_uid, series_uid, sop_uid))
    return all_instances

def download_instance(client, study_uid, series_uid, sop_uid):
    dcm_bytes = client.retrieve_instance(study_uid, series_uid, sop_uid)
    return dcmread(io.BytesIO(dcm_bytes))

ORTHANC_DICOMWEB_URL = os.environ.get('ORTHANC_DICOMWEB_URL', 'http://localhost:8042/dicom-web')
ORTHANC_USERNAME = os.environ.get('ORTHANC_USERNAME', 'orthanc')
ORTHANC_PASSWORD = os.environ.get('ORTHANC_PASSWORD', 'orthanc')

def get_orthanc_client():
    """
    Возвращает DICOMwebClient для Orthanc, используя адрес и логин из переменных.
    """
    from dicomweb_client.api import DICOMwebClient
    import requests
    session = requests.Session()
    session.auth = (ORTHANC_USERNAME, ORTHANC_PASSWORD)
    return DICOMwebClient(ORTHANC_DICOMWEB_URL, session=session)


def find_series(series_dict, is_sagittal=None, is_axial=None, is_coronal=None, seq_type=None):
    for uid, paths in series_dict.items():
        try:
            # Проверяем первый срез серии на локализатор
            first_dcm = dcmread(paths[0], stop_before_pixels=True)
            if is_localizer_series(first_dcm):
                continue
            # Проверяем все срезы в серии, а не только первый
            sagittal_slices = []
            axial_slices = []
            coronal_slices = []

            for path in paths:
                dcm = dcmread(path, stop_before_pixels=True)

                # Проверяем тип последовательности
                if seq_type and get_sequence_type(dcm) != seq_type:
                    continue

                # Группируем срезы по проекции
                if is_sagittal_dicom_slice(dcm):
                    sagittal_slices.append(path)
                elif is_axial_dicom_slice(dcm):
                    axial_slices.append(path)
                elif is_coronal:
                    # Если не сагиттальный и не аксиальный, считаем корональным
                    coronal_slices.append(path)

            # Возвращаем срезы нужной проекции
            if is_sagittal and sagittal_slices:
                return sagittal_slices
            elif is_axial and axial_slices:
                return axial_slices
            elif is_coronal and coronal_slices:
                return coronal_slices

        except Exception:
            continue
    return None

def load_study_dicoms(
    study_folder: str,
    require_extensions: bool = False,
    metadata_overwrites: Optional[Dict] = None,
) -> Tuple[Tuple[Tuple[Optional[SpinalScan], Optional[SpinalScan], Optional[SpinalScan]],
                Tuple[Optional[SpinalScan], Optional[SpinalScan], Optional[SpinalScan]],
                Tuple[Optional[SpinalScan], Optional[SpinalScan], Optional[SpinalScan]]], list]:
    '''
    Загружает DICOM-сканы на уровне исследования (study):
    - если dicomweb=True, использует DICOMwebClient и study_uid
    - иначе работает с локальными файлами
    '''
    if metadata_overwrites is None:
        metadata_overwrites = {}

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

    # Для каждой проекции и режима ищем подходящую серию


    # Для каждой проекции и режима
    seqs = ['T1', 'T2', 'STIR']
    sag_scans = []
    ax_scans = []
    cor_scans = []
    for seq in seqs:
        sag_paths = find_series(series_dict, is_sagittal=True, seq_type=seq)
        ax_paths = find_series(series_dict, is_axial=True, seq_type=seq)
        cor_paths = find_series(series_dict, is_coronal=True, seq_type=seq)

        # Используем новую функцию process_single_series вместо load_dicoms
        sag_scan = process_single_series(sag_paths, require_extensions, metadata_overwrites) if sag_paths else None
        ax_scan = process_single_series(ax_paths, require_extensions, metadata_overwrites) if ax_paths else None
        cor_scan = process_single_series(cor_paths, require_extensions, metadata_overwrites) if cor_paths else None

        sag_scans.append(sag_scan)
        ax_scans.append(ax_scan)
        cor_scans.append(cor_scan)

    # correspondence только для основной пары (например, T2)
    # Найти первую непустую пару для correspondence
    correspondence = []
    for i, (sag, ax) in enumerate(zip(sag_scans, ax_scans)):
        if sag is not None and ax is not None:
            # Сопоставление срезов по координатам (используем Z-координату)
            sag_ipps = np.array([dcmread(p, stop_before_pixels=True).ImagePositionPatient for p in series_dict[list(series_dict.keys())[i]]])
            ax_ipps = np.array([dcmread(p, stop_before_pixels=True).ImagePositionPatient for p in series_dict[list(series_dict.keys())[i]]])
            sag_z = sag_ipps[:, 2]
            ax_z = ax_ipps[:, 2]
            for j, sz in enumerate(sag_z):
                k = np.argmin(np.abs(ax_z - sz))
                distance = np.abs(ax_z[k] - sz)
                correspondence.append((j, k, sz, ax_z[k], distance))
            break

    return ((tuple(sag_scans), tuple(ax_scans), tuple(cor_scans)), correspondence)


def is_localizer_series(dicom_file: FileDataset) -> bool:
    desc_fields = [
        getattr(dicom_file, 'SeriesDescription', '').upper(),
        getattr(dicom_file, 'ProtocolName', '').upper(),
        getattr(dicom_file, 'SequenceName', '').upper()
    ]
    keywords = ['LOCALIZER', 'LOC', 'SCOUT', 'SURVEY', 'PLANNING', 'REFERENCE', 'TOPO', 'POSITION', 'LOCALISATOR']
    return any(any(kw in field for kw in keywords) for field in desc_fields)
