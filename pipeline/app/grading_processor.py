"""
Исправленный интегрированный процессор для grading анализа позвоночника.
Объединяет логику из spine_grading_pipeline.py и grading_dual_channel.py
для использования в основном пайплайне main_refactored.py.
"""

import numpy as np
import torch
import cv2
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import requests
from scipy.ndimage import affine_transform, label, center_of_mass
import tritonclient.grpc as grpcclient

import os
import matplotlib.pyplot as plt

from config import settings
from predictor import triton_inference

# Импорт модели
try:
    from grading_dual_channel import DualChannelGradingModel, create_dual_channel_model
    DUAL_CHANNEL_AVAILABLE = True
except ImportError:
    DUAL_CHANNEL_AVAILABLE = False
    DualChannelGradingModel = None
    create_dual_channel_model = None

logger = logging.getLogger(__name__)

class SpineGradingPipeline:
    def __init__(self, patch_size=(64, 128, 128), batch_size=4,
                 triton_url="http://localhost:8000/v2/models/grading/infer"):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.triton_url = triton_url

    @staticmethod
    def _align_disc(mri, seg, disc_mask):
        """
        mri: np.array (C, D, H, W)  -- индексный порядок (z,y,x) внутри D,H,W
        seg: np.array (D, H, W)
        disc_mask: булев массив (D, H, W)
        """
        coords = np.argwhere(disc_mask > 0)  # coords in (z, y, x)
        if coords.size == 0:
            return mri, seg

        center = coords.mean(axis=0)  # (z, y, x)

        # --- PCA: нормаль к плоскости диска (наименьшая дисперсия)
        coords_centered = coords - center
        _, _, vh = np.linalg.svd(coords_centered, full_matrices=False)
        normal = vh[-1]  # vector in (z, y, x) coordinates

        # Целевая нормаль — ось Z тома в coordinate order (z,y,x) -> [1,0,0]
        target_normal = np.array([1.0, 0.0, 0.0])
        if np.dot(normal, target_normal) < 0:
            normal = -normal

        # Ротация, переводящая normal -> target_normal (Rodrigues)
        v = np.cross(normal, target_normal)
        s = np.linalg.norm(v)
        if s != 0:
            c = np.dot(normal, target_normal)
            vx = np.array([[0.0, -v[2], v[1]],
                           [v[2], 0.0, -v[0]],
                           [-v[1], v[0], 0.0]])
            R_align = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s ** 2))
        else:
            R_align = np.eye(3)

        # Применим предварительную ротацию к координатам (не к объему) чтобы найти in-plane ориентацию
        coords_rot1 = (R_align @ coords_centered.T).T

        # Ещё одна PCA уже в выровненной системе: главный вектор задаёт направление "длины" диска в плоскости
        _, _, vh2 = np.linalg.svd(coords_rot1, full_matrices=False)
        in_plane = vh2[0]  # основной вектор (в (z,y,x)), должен лежать почти в плоскости (z≈0)

        # Берём компоненты в плоскости Y-X: vy = in_plane[1], vx = in_plane[2]
        vx = in_plane[2]
        vy = in_plane[1]
        # угол в плоскости XY (стандарт: atan2(y, x) — угол от Ox к вектору)
        ang = np.arctan2(vy, vx)
        # хотим, чтобы этот in_plane вектор оказался направлен вдоль +Y, т.е. угол -> pi/2
        delta = (np.pi / 2.0) - ang

        # Ротация вокруг "оси Z тома" (в ordering (z,y,x) это ось index 0)
        cosd = np.cos(delta)
        sind = np.sin(delta)
        R_z = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cosd, -sind],
            [0.0, sind, cosd]
        ])

        # Полная матрица вращения
        R_total = R_z @ R_align

        # Гарантируем "вверх" (если после поворота up смотрит вниз, отражаем по Y)
        up = np.array([0.0, 1.0, 0.0])  # в (z,y,x)
        up_rot = R_total @ up
        if up_rot[1] < 0:
            reflect = np.diag([1.0, -1.0, 1.0])
            R_total = reflect @ R_total

        # Для affine_transform: matrix должен быть такой, чтобы input_coords = matrix @ output_coords + offset
        # Для вращения R_total (которая действует на векторы координат), matrix = R_total.T
        mat = R_total.T
        offset = center - mat @ center

        # Применяем к каналам MRI и к сегментации
        mri_rot = np.zeros_like(mri)
        for ch in range(mri.shape[0]):
            mri_rot[ch] = affine_transform(mri[ch], mat, offset=offset, order=1, mode='nearest')

        seg_rot = affine_transform(seg, mat, offset=offset, order=0, mode='nearest')

        return mri_rot, seg_rot

    def slice_patches_with_alignment(self, mri_2ch, seg_dict, is_align=True):
        patches_mri = []
        patches_seg = []
        flags = []
        codes = []

        mask_disc = seg_dict["disc"]
        mask_vertebra = seg_dict["vertebra"]
        mask_canal = seg_dict["canal"]

        # Определяем метку канала (если есть несколько — берём первый ненулевой)
        canal_values = np.unique(mask_canal)
        canal_values = canal_values[canal_values != 0]
        canal_label = int(canal_values[0]) if len(canal_values) > 0 else None

        # Для каждого диска формируем патч
        for disc_id in np.unique(mask_disc):
            if disc_id == 0:
                continue

            try:
                disc_mask = (mask_disc == disc_id)
                if is_align:
                    mri_aligned, seg_aligned = self._align_disc(mri_2ch, seg_dict["full"], disc_mask)
                else:
                    mri_aligned, seg_aligned = mri_2ch, seg_dict["full"]

                disc_mask_rot = (seg_aligned == disc_id)
                coords = np.argwhere(disc_mask_rot)

                if coords.size == 0:
                    print(f"Диск {disc_id} пуст после поворота")
                    continue

                # Центр масс диска
                center = coords.mean(axis=0).astype(int)  # (z, y, x)

                # Фиксированный размер патча
                pd, ph, pw = self.patch_size  # D, H, W

                # Границы патча
                zc, yc, xc = center
                zmin = zc - pd // 2
                zmax = zmin + pd
                ymin = yc - ph // 2
                ymax = ymin + ph
                xmin = xc - pw // 2
                xmax = xmin + pw

                # Пустые массивы
                patch_mri = np.zeros((mri_aligned.shape[0], pd, ph, pw), dtype=mri_aligned.dtype)
                patch_seg = np.zeros((pd, ph, pw), dtype=seg_aligned.dtype)

                # Копируем с учётом границ
                zmin_src = max(zmin, 0)
                zmax_src = min(zmax, mri_aligned.shape[1])
                ymin_src = max(ymin, 0)
                ymax_src = min(ymax, mri_aligned.shape[2])
                xmin_src = max(xmin, 0)
                xmax_src = min(xmax, mri_aligned.shape[3])

                zmin_dst = zmin_src - zmin
                zmax_dst = zmin_dst + (zmax_src - zmin_src)
                ymin_dst = ymin_src - ymin
                ymax_dst = ymin_dst + (ymax_src - ymin_src)
                xmin_dst = xmin_src - xmin
                xmax_dst = xmin_dst + (xmax_src - xmin_src)

                patch_mri[:, zmin_dst:zmax_dst, ymin_dst:ymax_dst, xmin_dst:xmax_dst] = \
                    mri_aligned[:, zmin_src:zmax_src, ymin_src:ymax_src, xmin_src:xmax_src]

                patch_seg[zmin_dst:zmax_dst, ymin_dst:ymax_dst, xmin_dst:xmax_dst] = \
                    seg_aligned[zmin_src:zmax_src, ymin_src:ymax_src, xmin_src:xmax_src]

                # Диагностика по patch_seg
                labels_ids = np.unique(patch_seg)
                vertebra_ids = labels_ids[np.isin(labels_ids, np.unique(mask_vertebra))]
                vertebra_ids = vertebra_ids[vertebra_ids != 0]
                disk_ids = labels_ids[np.isin(labels_ids, np.unique(mask_disc))]
                disk_ids = disk_ids[disk_ids != 0]

                # Условия по позвонкам/диску
                has_two_or_less_verts = len(vertebra_ids) <= 3  # вы сами ставите <=3
                has_one_disc = len(disk_ids) == 1

                # Проверка наличия канала в патче (по метке canal_label)
                canal_present = False
                canal_area_total = 0
                coronal_area = 0
                sagittal_area = 0
                axial_area = 0

                if canal_label is not None:
                    canal_present = np.any(patch_seg == canal_label)
                    canal_area_total = int(np.sum(patch_seg == canal_label))

                    # Центральные срезы (по соглашению: coronal mid x, sagittal mid y, axial mid z)
                    mid_x = pw // 2
                    mid_y = ph // 2
                    mid_z = pd // 2

                    # Коронарный срез: (D, H) на mid_x -> patch_seg[:, :, mid_x]
                    sagittal_area = patch_seg[:, :, mid_x]
                    sagittal_area = int(np.sum(sagittal_area == canal_label))

                    # Сагиттальный срез: (D, W) на mid_y -> patch_seg[:, mid_y, :]
                    coronal_area = patch_seg[:, mid_y, :]
                    coronal_area = int(np.sum(coronal_area == canal_label))

                    # Аксиальный срез: (H, W) на mid_z -> patch_seg[mid_z, :, :]
                    axial_slice = patch_seg[mid_z, :, :]
                    axial_area = int(np.sum(axial_slice == canal_label))

                # Новое анатомическое условие:
                # площадь канала в сагиттале должна быть больше, чем в коронарном и аксиальном
                canal_area_condition = False
                if canal_label is not None and canal_present:
                    canal_area_condition = (sagittal_area > coronal_area) and (sagittal_area > axial_area)
                else:
                    canal_area_condition = False

                flag_valid = bool(has_two_or_less_verts and has_one_disc and canal_present and canal_area_condition)

                patches_mri.append(patch_mri)
                patches_seg.append(patch_seg)
                flags.append(flag_valid)
                codes.append({
                    "disc_id": int(disc_id),
                    "labels_ids": labels_ids.tolist(),
                    "vertebra_ids": vertebra_ids.tolist(),
                    "disk_ids": disk_ids.tolist(),
                    "has_two_or_less_verts": bool(has_two_or_less_verts),
                    "has_one_disc": bool(has_one_disc),
                    "canal_label": int(canal_label) if canal_label is not None else None,
                    "canal_present": bool(canal_present),
                    "canal_area_total": canal_area_total,
                    "coronal_area": coronal_area,
                    "sagittal_area": sagittal_area,
                    "axial_area": axial_area,
                    "canal_area_condition": bool(canal_area_condition),
                    "flag_valid": bool(flag_valid)
                })
            except Exception as e:
                print(f"Пропускаем диск {disc_id}, ошибка: {e}")
                continue

        return patches_mri, patches_seg, flags, codes

    def send_to_triton(self, patches_mri):
        results = []
        for i in range(0, len(patches_mri), self.batch_size):
            batch = patches_mri[i:i+self.batch_size]
            batch_np = np.stack(batch, axis=0).astype(np.float32)  # (B, 2, D, H, W)

            # Формируем JSON для Triton
            inputs = [{
                "name": "input__0",
                "shape": list(batch_np.shape),
                "datatype": "FP32",
                "data": batch_np.tolist()
            }]

            response = requests.post(self.triton_url, json={"inputs": inputs})
            response.raise_for_status()
            outputs = response.json()["outputs"][0]["data"]

            for out in outputs:
                results.append({
                    'ModicChanges': out[0],
                    'UpperEndplateDefect': out[1],
                    'LowerEndplateDefect': out[2],
                    'Spondylolisthesis': out[3],
                    'Herniation': out[4],
                    'Narrowing': out[5],
                    'Bulging': out[6],
                    'Pfirrmann': out[7]
                })

        return results

    @staticmethod
    def visualize_patch_with_results(patch_mri, patch_seg, flag_valid, result=None, save_path=None):
        if save_path is None:
            os.makedirs("test", exist_ok=True)
            save_path = "test/patch_vis.png"

        c, D, H, W = patch_mri.shape
        assert c >= 2

        zc, yc, xc = D // 2, H // 2, W // 2

        dif_t1t2 = np.abs(patch_mri[0] - patch_mri[1])

        fig, axes = plt.subplots(3, 4, figsize=(16, 12))

        # Проекции по строкам
        projections = [
            ('Axial', zc, lambda arr: arr[zc, :, :]),
            ('Coronal', yc, lambda arr: arr[:, yc, :]),
            ('Sagittal', xc, lambda arr: arr[:, :, xc]),
        ]

        patch_seg[patch_seg != 0] -= 190

        # Каналы по столбцам
        channels = [
            ('T1', patch_mri[0]),
            ('T2', patch_mri[1]),
            ('Diff (|T1-T2|)', dif_t1t2),
            ('Seg', patch_seg)
        ]

        for row_idx, (proj_name, coord, slicer) in enumerate(projections):
            for col_idx, (ch_name, data) in enumerate(channels):
                ax = axes[row_idx, col_idx]

                # Для маски (Seg) применяем cmap='tab20', иначе 'gray'
                cmap = 'tab20' if ch_name == 'Seg' else 'gray'

                img = slicer(data)

                # Отображаем
                ax.imshow(img, cmap=cmap)
                ax.set_title(f'{proj_name} {ch_name} (slice={coord})')
                ax.axis('off')

        valid_str = "Valid anatomy" if flag_valid else "Invalid anatomy"
        info_lines = [valid_str]
        if result:
            for k, v in result.items():
                info_lines.append(f"{k}: {v}")

        info_text = "\n".join(info_lines)
        plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=10)

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(save_path)
        plt.close(fig)

    def visualize_patch_set(self, patches_mri, patches_seg, flags, results=None, save_dir="test_vis"):
        """
        Визуализировать и сохранить все патчи из списков с их флагами и опциональными результатами.

        patches_mri: list of np.array (C, D, H, W)
        patches_seg: list of np.array (D, H, W)
        flags: list of bool
        results: list of dict или None
        save_dir: папка для сохранения
        """
        os.makedirs(save_dir, exist_ok=True)

        for i, (patch_mri, patch_seg, flag) in enumerate(zip(patches_mri, patches_seg, flags)):
            result = results[i] if results is not None and i < len(results) else None
            save_path = os.path.join(save_dir, f"patch_{i}.png")
            self.visualize_patch_with_results(patch_mri, patch_seg, flag, result, save_path)
            print(f"Сохранена визуализация патча {i} -> {save_path}")


class SpinePostProcessor:
    def __init__(self, voxel_spacing=(1.0, 1.0, 1.0), class_map=None):
        """
        voxel_spacing: (z, y, x) мм
        class_map: словарь индексов классов сегментации
        """
        self.voxel_spacing = voxel_spacing
        self.class_map = class_map or {
            "disc": 1,
            "vertebra": 2,
            "canal": 3,
            "dural_sac": 4,
            "hernia": 99
        }

    def process(self, seg_full, patches_seg, grading_results):
        seg_full = seg_full.copy()
        new_patches_seg = []
        updated_results = []

        voxel_vol_mm3 = np.prod(self.voxel_spacing)
        voxel_area_mm2 = self.voxel_spacing[1] * self.voxel_spacing[2]

        for i, (patch_seg, result) in enumerate(zip(patches_seg, grading_results)):
            patch_seg_mod = patch_seg.copy()

            # ----- Выделение грыжи -----
            hernia_size_mm2 = 0.0
            hernia_vol_mm3 = 0.0
            hernia_length_mm = 0.0  # добавлено

            if result["Herniation"] == 1:
                disc_mask = (patch_seg == self.class_map["disc"])
                canal_mask = (patch_seg == self.class_map["canal"])
                dural_mask = (patch_seg == self.class_map["dural_sac"])

                hernia_mask = disc_mask & (canal_mask | dural_mask)

                if np.any(hernia_mask):
                    # Площадь — берём макс. по срезу Z
                    for z in range(hernia_mask.shape[0]):
                        slice_area = np.sum(hernia_mask[z]) * voxel_area_mm2
                        hernia_size_mm2 = max(hernia_size_mm2, slice_area)

                    # Объём
                    hernia_vol_mm3 = np.sum(hernia_mask) * voxel_vol_mm3

                    # Длина (по оси Y - индекс 1)
                    hernia_coords = np.argwhere(hernia_mask)
                    length_voxels = hernia_coords[:, 1].max() - hernia_coords[:, 1].min() + 1
                    hernia_length_mm = length_voxels * self.voxel_spacing[1]

                    # Записываем в сегментацию
                    patch_seg_mod[hernia_mask] = self.class_map["hernia"]

            # ----- Измерение листеза -----
            listhesis_mm = 0.0
            if result["Spondylolisthesis"] == 1:
                vert_mask = (patch_seg == self.class_map["vertebra"])
                disc_mask = (patch_seg == self.class_map["disc"])

                labeled, num = label(vert_mask)
                if num >= 2:
                    # Берём 2 крупнейших
                    sizes = [np.sum(labeled == lbl) for lbl in range(1, num + 1)]
                    idxs = np.argsort(sizes)[-2:]
                    centers = [center_of_mass(labeled == (lbl + 1)) for lbl in idxs]
                    # Смещение по оси X или Y
                    listhesis_mm = abs(centers[0][2] - centers[1][2]) * self.voxel_spacing[2]

            # Обновляем результаты
            result_upd = result.copy()
            result_upd.update({
                "HerniaLength_mm": hernia_length_mm,
                "HerniaSize_mm2": hernia_size_mm2,
                "HerniaVolume_mm3": hernia_vol_mm3,
                "Listhesis_mm": listhesis_mm
            })

            updated_results.append(result_upd)
            new_patches_seg.append(patch_seg_mod)

            # ----- Вставка в полную сегментацию -----
            seg_full[(patch_seg_mod == self.class_map["hernia"])] = self.class_map["hernia"]

        return seg_full, new_patches_seg, updated_results

class SpineGradingProcessor:
    """
    Процессор для grading анализа позвоночника.
    Интегрируется в основной пайплайн main_refactored.py.
    """
    
    def __init__(self, client: grpcclient.InferenceServerClient):
        self.client = client
        # Маппинг меток дисков из основного проекта (исправленный)
        self.disc_labels_map = settings.VERTEBRA_DESCRIPTIONS
        
        # Категории патологий
        self.categories = [
            'Modic', 'UP endplate', 'LOW endplate',
            'Spondylolisthesis', 'Disc herniation', 'Disc narrowing', 'Disc bulging', 'Pfirrmann grade'
        ]
        
        # Описания категорий
        self.category_descriptions = {
            'ModicChanges': 'Изменения костного мозга (0-3)',
            'UpperEndplateDefect': 'Дефект верхней замыкательной пластины (0-1)',
            'LowerEndplateDefect': 'Дефект нижней замыкательной пластины (0-1)',
            'Spondylolisthesis': 'Спондилолистез (0-1)',
            'Herniation': 'Грыжа диска (0-1)',
            'Narrowing': 'Сужение диска (0-1)',
            'Bulging': 'Протрузия диска (0-1)',
            'Pfirrmann': 'Дегенерация по Pfirrmann (0-4)'
        }

        try:
            self.slice_pipeline = SpineGradingPipeline(patch_size=(32, 112, 32), batch_size=4)

        except Exception as e:
            self.slice_pipeline = None
            logger.warning(f"не удалось инициализировать grading_pipeline для нарезки: {e}")

    
    def process_disks(self, 
                     mri_data: List[np.ndarray],
                     mask_data: np.ndarray,
                     present_disks: List[int]) -> Dict[int, Dict]:
        """
        Обрабатывает все найденные диски для grading анализа.
        
        Args:
            mri_data: Данные МРТ изображения (может быть многоканальным)
            mask_data: Данные сегментации
            present_disks: Список меток найденных дисков
            
        Returns:
            Словарь с результатами grading для каждого диска
        """
        results = {}


        # Многоканальные данные - используем первые два канала как T1 и T2
        t1_data = mri_data[0]
        t2_data = mri_data[1]

        

        # Единая нарезка патчей через grading_pipeline (is_align=False) и инференс по патчам
        try:
            if self.slice_pipeline is None:
                raise RuntimeError("slice_pipeline не инициализирован")

            # Готовим многоканальные данные (C, D, H, W)
            mri_2ch = np.stack([t1_data, t2_data], axis=0)

            mri_2ch = np.rot90(np.transpose(mri_2ch, (0, 3, 2, 1)), k=2, axes=(1, 2))
            mask_data = np.rot90(np.transpose(mask_data, (2, 1, 0)), k=2, axes=(0, 1))
            disc_mask = np.where(np.isin(mask_data, present_disks), mask_data, 0).astype(np.uint16)
            vertebra_mask = np.where(((mask_data >= 11) & (mask_data <= 50)), mask_data, 0).astype(np.uint16)
            canal_mask = np.where(mask_data == settings.CANAL_LABEL, settings.CANAL_LABEL, 0).astype(np.uint8)

            seg_dict = {
                "disc": disc_mask,
                "vertebra": vertebra_mask,
                "canal": canal_mask,
                "full": mask_data.astype(mask_data.dtype)
            }

            patches_mri, patches_seg, flags, codes = self.slice_pipeline.slice_patches_with_alignment(
                mri_2ch, seg_dict, is_align=False
            )

            for i, patch in enumerate(patches_mri):
                try:
                    disc_id = int(codes[i].get("disc_id", 0)) if i < len(codes) else 0
                    if i == 0 or disc_id not in present_disks:
                        continue

                    t1_processed = self._process_single_channel(patch[0])
                    t2_processed = self._process_single_channel(patch[1])
                    if t1_processed is None or t2_processed is None:
                        results[disc_id] = {"error": "Failed to process patch"}
                        continue

                    prepared_volume = np.stack([t1_processed, t2_processed], axis=0)
                    predictions = self._predict_dual_channel(prepared_volume)

                    # # Визуализация патча и результатов в results/patch_vis
                    # try:
                    #     save_dir = Path('results') / 'patch_vis'
                    #     save_dir.mkdir(parents=True, exist_ok=True)
                    #     save_path = save_dir / f"disc_{disc_id}_patch_{i}.png"
                    #     self.slice_pipeline.visualize_patch_with_results(
                    #         patch_mri=patch,
                    #         patch_seg=patches_seg[i],
                    #         flag_valid=flags[i],
                    #         result=predictions,
                    #         save_path=str(save_path)
                    #     )
                    #     logger.info(f"Сохранена визуализация патча: {save_path}")
                    # except Exception as vis_e:
                    #     logger.error(f"Ошибка визуализации патча disc_id={disc_id}, i={i}: {vis_e}")

                    # Глобальные bbox/центр из исходной маски
                    coords = np.argwhere(mask_data == disc_id)
                    if coords.size > 0:
                        min_coords = coords.min(axis=0)
                        max_coords = coords.max(axis=0)
                        center = coords.mean(axis=0).astype(int)
                        bounds = (min_coords.tolist(), max_coords.tolist())
                        center_list = center.tolist()
                    else:
                        bounds = (None, None)
                        center_list = None

                    level_name = self.disc_labels_map.get(disc_id, f"DISC_{disc_id}")
                    result = {
                        'disc_label': int(disc_id),
                        'level_name': level_name,
                        'center': center_list,
                        'bounds': bounds,
                        'predictions': predictions,
                        'confidence_scores': self._calculate_confidence(predictions),
                        'crop_shape': tuple(patch.shape[1:]),  # (D, H, W)
                        'disk_voxels': int(np.sum(mask_data == disc_id)),
                        'model_type': 'dual_channel'
                    }
                    results[disc_id] = result
                except Exception as e:
                    logger.error(f"Ошибка при обработке патча #{i} (disc_id={codes[i].get('disc_id', 'N/A') if i < len(codes) else 'N/A'}): {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Не перезаписываем существующий результат, если уже есть
                    key = int(codes[i].get('disc_id', 0)) if i < len(codes) and 'disc_id' in codes[i] else None
                    if key is not None:
                        results[key] = {"error": str(e)}

        except Exception as e:
            logger.error(f"Ошибка при нарезке/инференсе через grading_pipeline: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            for disk_label in present_disks:
                try:
                    disk_result = self._process_single_disk_dual_channel(t1_data, t2_data, mask_data, disk_label)
                    results[disk_label] = disk_result
                except Exception as ee:
                    logger.error(f"Ошибка при fallback обработке диска {disk_label}: {ee}")
                    results[disk_label] = {"error": str(ee)}

        return results
    
    def _process_single_disk_dual_channel(self, 
                                        t1_data: np.ndarray,
                                        t2_data: np.ndarray,
                                        mask_data: np.ndarray, 
                                        disk_label: int) -> Dict:
        """
        Обрабатывает отдельный диск в двухканальном режиме.
        Использует логику подготовки кропов из spine_grading_pipeline.py.
        """
        try:
            logger.info(f"Двухканальная обработка диска {disk_label}")
            logger.info(f"T1 data shape: {t1_data.shape}, T2 data shape: {t2_data.shape}")
            
            # Находим диск в маске
            disk_mask = (mask_data == disk_label)
            if not np.any(disk_mask):
                return {"error": "Disk not found in mask"}
            
            # Находим bounding box диска
            coords = np.argwhere(disk_mask)
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            center = coords.mean(axis=0).astype(int)
            
            logger.info(f"Диск {disk_label}: центр={center}, границы=({min_coords}, {max_coords})")
            
            # Создаем crop вокруг диска с увеличенными границами
            padding = [15, 25, 10]  # depth, height, width
            crop_slices = []
            for i in range(3):
                start = max(0, min_coords[i] - padding[i])
                end = min(mask_data.shape[i], max_coords[i] + padding[i])
                crop_slices.append(slice(start, end))
            
            # Кропаем данные
            t1_cropped = t1_data[tuple(crop_slices)]
            t2_cropped = t2_data[tuple(crop_slices)]
            mask_cropped = mask_data[tuple(crop_slices)]
            disk_mask_cropped = disk_mask[tuple(crop_slices)]
            
            logger.info(f"Кропы: T1={t1_cropped.shape}, T2={t2_cropped.shape}, mask={mask_cropped.shape}")
            
            # Подготавливаем двухканальный объем для модели
            prepared_volume = self._prepare_dual_channel_volume(
                t1_cropped, t2_cropped, disk_mask_cropped, mask_cropped, disk_label
            )
            
            if prepared_volume is None:
                return {"error": "Failed to prepare volume"}
            
            logger.info(f"Подготовленный объем: {prepared_volume.shape}")
            
            # Предсказание модели
            predictions = self._predict_dual_channel(prepared_volume)
            
            # Создаем результат
            level_name = self.disc_labels_map.get(disk_label, f"DISC_{disk_label}")
            
            result = {
                'disc_label': int(disk_label),
                'level_name': level_name,
                'center': center.tolist(),
                'bounds': (min_coords.tolist(), max_coords.tolist()),
                'predictions': predictions,
                'confidence_scores': self._calculate_confidence(predictions),
                'crop_shape': t1_cropped.shape,
                'disk_voxels': int(np.sum(disk_mask)),
                'model_type': 'dual_channel'
            }
            
            logger.info(f"Результат для диска {disk_label}: {predictions}")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при двухканальной обработке диска {disk_label}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"error": str(e)}
    
    def _process_single_disk_single_channel(self, 
                                          mri_data: np.ndarray,
                                          mask_data: np.ndarray, 
                                          disk_label: int) -> Dict:
        """
        Обрабатывает отдельный диск в одноканальном режиме.
        """
        try:
            logger.info(f"Одноканальная обработка диска {disk_label}")
            
            # Находим диск в маске
            disk_mask = (mask_data == disk_label)
            if not np.any(disk_mask):
                return {"error": "Disk not found in mask"}
            
            # Находим bounding box диска
            coords = np.argwhere(disk_mask)
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            center = coords.mean(axis=0).astype(int)
            
            # Создаем crop вокруг диска
            crop_margin = 10
            crop_slices = []
            for i in range(3):
                start = max(0, min_coords[i] - crop_margin)
                end = min(mri_data.shape[i], max_coords[i] + crop_margin)
                crop_slices.append(slice(start, end))
            
            # Кропаем данные
            cropped_mri = mri_data[tuple(crop_slices)]
            
            # Нормализация изображения
            mean = cropped_mri.mean()
            std = cropped_mri.std() if cropped_mri.std() > 0 else 1.0
            normalized_mri = (cropped_mri - mean) / std
            
            # Подготавливаем тензор для модели
            img_tensor = torch.tensor(normalized_mri).unsqueeze(0).unsqueeze(0).float().to(self.device)
            
            # Предсказание
            with torch.no_grad():
                grading_outputs = self.model(img_tensor)
            
            # Обрабатываем результаты
            predictions = {}
            for i, category in enumerate(self.categories):
                if i < len(grading_outputs):
                    predicted_class = torch.argmax(grading_outputs[i]).detach().cpu().numpy().item()
                    predictions[category] = predicted_class
                else:
                    predictions[category] = 0
            
            # Создаем результат
            level_name = self.disc_labels_map.get(disk_label, f"DISC_{disk_label}")
            
            result = {
                'disc_label': int(disk_label),
                'level_name': level_name,
                'center': center.tolist(),
                'bounds': (min_coords.tolist(), max_coords.tolist()),
                'predictions': predictions,
                'confidence_scores': self._calculate_confidence(predictions),
                'crop_shape': cropped_mri.shape,
                'disk_voxels': int(np.sum(disk_mask)),
                'model_type': 'single_channel'
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при одноканальной обработке диска {disk_label}: {e}")
            return {"error": str(e)}
    
    def _prepare_dual_channel_volume(self, 
                                   t1_volume: np.ndarray, 
                                   t2_volume: np.ndarray,
                                   disc_mask: np.ndarray, 
                                   mask_volume: np.ndarray,
                                   disc_label: int) -> Optional[np.ndarray]:
        """
        Подготовка двухканального объема для модели.
        Адаптировано из spine_grading_pipeline.py.
        """
        if t1_volume.size == 0 or t2_volume.size == 0:
            return None
        
        try:
            logger.info(f"Подготовка двухканального объема для диска {disc_label}")
            logger.info(f"Входные объемы: T1={t1_volume.shape}, T2={t2_volume.shape}")
            
            # Применить выравнивание
            t1_aligned, rotation_angle = self._rotate_and_align_volume(t1_volume, disc_mask, mask_volume)
            t2_aligned, _ = self._rotate_and_align_volume(t2_volume, disc_mask, mask_volume, rotation_angle)
            
            logger.info(f"После выравнивания: T1={t1_aligned.shape}, T2={t2_aligned.shape}, угол={rotation_angle}")
            
            # Обработать каждый канал
            t1_processed = self._process_single_channel(t1_aligned)
            t2_processed = self._process_single_channel(t2_aligned)
            
            if t1_processed is None or t2_processed is None:
                logger.error("Не удалось обработать каналы")
                return None
            
            logger.info(f"После обработки каналов: T1={t1_processed.shape}, T2={t2_processed.shape}")
            
            # Объединить каналы
            dual_channel_volume = np.stack([t1_processed, t2_processed], axis=0)
            
            logger.info(f"Финальный двухканальный объем: {dual_channel_volume.shape}")
            
            return dual_channel_volume
            
        except Exception as e:
            logger.error(f"Ошибка при подготовке двухканального объема: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _process_single_channel(self, volume: np.ndarray) -> Optional[np.ndarray]:
        """
        Обработка одного канала.
        Адаптировано из spine_grading_pipeline.py.
        """
        try:
            logger.info(f"Обработка канала: входной размер {volume.shape}")

            mean = volume.mean()
            std = volume.std()
            volume = (volume - mean) / std if std > 0 else (volume - mean)
            
            # # Нормализация
            # volume = volume.astype(np.float32)
            # vb_pair_median = np.median(volume[volume > 0]) if np.any(volume > 0) else 1.0
            # norm_med = 0.5
            #
            # volume = volume / vb_pair_median * norm_med
            # volume[volume < 0] = 0
            # volume[volume > 2.0] = 2.0
            # volume /= 2.0
            
            logger.info(f"После нормализации: {volume.shape}, min={volume.min()}, max={volume.max()}")
            
            # # Взять центральные срезы
            # depth = volume.shape[2]
            # if depth >= 15:
            #     center = depth // 2
            #     start = center - 7
            #     end = center + 8
            #     volume = volume[:, :, start:end]
            # else:
            #     pad_before = (15 - depth) // 2
            #     pad_after = 15 - depth - pad_before
            #     volume = np.pad(volume, ((0, 0), (0, 0), (pad_before, pad_after)),
            #                   mode='constant', constant_values=0)
            #
            # logger.info(f"После обработки срезов: {volume.shape}")
            
            # # Ресайз
            # target_size = (192, 320)
            # resized_slices = []
            #
            # for i in range(volume.shape[2]):
            #     slice_2d = volume[:, :, i]
            #     current_h, current_w = slice_2d.shape
            #
            #     # Обрезка/дополнение по высоте
            #     if current_h > target_size[0]:
            #         start_h = (current_h - target_size[0]) // 2
            #         slice_2d = slice_2d[start_h:start_h + target_size[0], :]
            #     elif current_h < target_size[0]:
            #         pad_h = (target_size[0] - current_h) // 2
            #         slice_2d = np.pad(slice_2d, ((pad_h, target_size[0] - current_h - pad_h), (0, 0)),
            #                         mode='constant', constant_values=0)
            #
            #     # Обрезка/дополнение по ширине
            #     if current_w > target_size[1]:
            #         start_w = (current_w - target_size[1]) // 2
            #         slice_2d = slice_2d[:, start_w:start_w + target_size[1]]
            #     elif current_w < target_size[1]:
            #         pad_w = (target_size[1] - current_w) // 2
            #         slice_2d = np.pad(slice_2d, ((0, 0), (pad_w, target_size[1] - current_w - pad_w)),
            #                         mode='constant', constant_values=0)
            #
            #     if slice_2d.shape != target_size:
            #         slice_2d = cv2.resize(slice_2d, (target_size[1], target_size[0]),
            #                             interpolation=cv2.INTER_CUBIC)
            #
            #     resized_slices.append(slice_2d)
            #
            # volume = np.stack(resized_slices, axis=2)
            # logger.info(f"После ресайза: {volume.shape}")
            #
            # # Взять центральные 9 срезов
            # center_idx = volume.shape[2] // 2
            # start_idx = center_idx - 4
            # end_idx = center_idx + 5
            #
            # final_volume = volume[:, :, start_idx:end_idx]
            # final_volume = final_volume[20:172, 24:296, :]  # (152, 272, 9)
            # final_volume = np.transpose(final_volume, (2, 0, 1))  # (9, 152, 272)
            #
            # logger.info(f"Финальный размер канала: {final_volume.shape}")
            
            return volume
            
        except Exception as e:
            logger.error(f"Ошибка при обработке кан��ла: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _rotate_and_align_volume(self, volume: np.ndarray, disc_mask: np.ndarray, 
                               mask_volume: np.ndarray = None, fixed_angle: float = None) -> Tuple[np.ndarray, float]:
        """
        Поворот и выравнивание объема.
        Адаптировано из spine_grading_pipeline.py.
        """
        try:
            if fixed_angle is not None:
                rotation_angle = fixed_angle
            else:
                center_slice_idx = volume.shape[2] // 2
                disc_slice = disc_mask[:, :, center_slice_idx].copy()
                
                contours, _ = cv2.findContours(disc_slice.astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    return volume, 0.0
                
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                (cx, cy), (width, height), angle = rect
                
                if width < height:
                    angle = angle - 90
                    
                if angle > 45:
                    angle = angle - 90
                elif angle < -45:
                    angle = angle + 90
                
                rotation_angle = -angle
            
            # Применить поворот
            h, w = volume.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((w/2, h/2), rotation_angle, 1.0)
            
            rotated_slices = []
            for i in range(volume.shape[2]):
                rotated_slice = cv2.warpAffine(volume[:, :, i], rotation_matrix, 
                                             (w, h), flags=cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_CONSTANT,
                                             borderValue=0)
                rotated_slices.append(rotated_slice)
            
            rotated_volume = np.stack(rotated_slices, axis=2)
            
            return rotated_volume, rotation_angle
            
        except Exception as e:
            logger.error(f"Ошибка при повороте объема: {e}")
            return volume, 0.0
    
    def _predict_dual_channel(self, volume: np.ndarray) -> Dict[str, int]:
        """
        Предсказание двухканальной модели для подготовленного объема.
        """
        try:
            output_names = [
                "Modic",
                "UP endplate",
                "LOW endplate",
                "Spondylolisthesis",
                "Disc herniation",
                "Disc narrowing",
                "Disc bulging",
                "Pfirrmann grade",
            ]
            outputs = triton_inference(
                    volume,
                    self.client,
                    'grading',
                    'input',
                    output_names
                )


            logger.info(f"Выходы модели: {len(outputs)} голов")

            # Преобразовать в предсказания
            predictions = {}
            for category in self.categories:
                logits = outputs[category]
                if logits.shape[1] > 1:
                    predicted_class = np.argmax(logits, axis=-1)
                else:
                    predicted_class = (logits > 0).astype(int)
                predictions[category] = predicted_class
                logger.info(f"{category}: {predicted_class}")

            
            return predictions
            
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {category: 0 for category in self.categories}
    
    def _calculate_confidence(self, predictions: Dict[str, int]) -> Dict[str, float]:
        """Вычисление уверенности предсказаний (заглушка)."""
        confidence_scores = {}
        for category, prediction in predictions.items():
            # Заглушка - высокая уверенность для ненулевых предсказаний
            if prediction > 0:
                confidence_scores[category] = np.random.uniform(0.8, 0.95)
            else:
                confidence_scores[category] = np.random.uniform(0.7, 0.85)
        
        return confidence_scores
    
    def create_summary(self, disk_results: Dict[int, Dict]) -> Dict:
        """Создание сводки по всем дискам."""
        if not disk_results:
            return {}
        
        # Фильтруем успешные результаты
        successful_results = [result for result in disk_results.values() if 'error' not in result]
        
        if not successful_results:
            return {'error': 'No successful disk processing'}
        
        summary = {
            'total_discs': len(disk_results),
            'successful_discs': len(successful_results),
            'failed_discs': len(disk_results) - len(successful_results),
            'pathology_counts': {},
            'severity_distribution': {}
        }
        
        # Подсчет патологий
        for category in self.categories:
            category_values = []
            for result in successful_results:
                if 'predictions' in result and category in result['predictions']:
                    category_values.append(result['predictions'][category])
            
            if category_values:
                unique_values, counts = np.unique(category_values, return_counts=True)
                summary['pathology_counts'][category] = {
                    'total': len(category_values),
                    'positive': sum(1 for v in category_values if v > 0),
                    'distribution': dict(zip(unique_values.astype(int), counts.astype(int)))
                }
        
        return summary