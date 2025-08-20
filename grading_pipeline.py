import numpy as np
import requests
from scipy.ndimage import affine_transform, label, center_of_mass

import os
import matplotlib.pyplot as plt

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
