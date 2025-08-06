"""
Обновленная двухканальная модель grading для анализа позвоночника.
Интегрирует компоненты из Deep_Spine_inf в проект Spine_analyzer.
"""

import os
import glob
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm
import logging

import sys
from garding import GradingModel, BasicBlock



# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Путь к подготовленному датасету (корректируйте при необходимости)
OUTPUT_DIR = r"F:\WorkSpace\Z-Union\chess\10159290\dataset_cuts"


class DualChannelGradingDataset(Dataset):
    """
    Датасет для двухканальной модели grading.
    Поддерживает загрузку изображения и сегментации как отдельных каналов.
    """
    
    def __init__(self, output_dir, augment=False, use_dual_channel=True):
        self.output_dir = output_dir
        self.augment = augment
        self.use_dual_channel = use_dual_channel
        self.samples = []
        
        # Ищем файлы с данными
        for img_path in glob.glob(os.path.join(output_dir, '*_img.nii.gz')):
            base = img_path.replace('_img.nii.gz', '')
            grading_path = base + '_grading.csv'
            mask_path = base + '_mask.nii.gz'
            
            if os.path.exists(grading_path):
                if self.use_dual_channel and os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path, grading_path))
                elif not self.use_dual_channel:
                    self.samples.append((img_path, None, grading_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.use_dual_channel:
            img_path, mask_path, grading_path = self.samples[idx]
        else:
            img_path, _, grading_path = self.samples[idx]
            mask_path = None
        
        # Загружаем изображение
        img = nib.load(img_path).get_fdata().astype(np.float32)
        
        # Если изображение многоканальное, берем первый канал
        if img.ndim == 4:
            img = img[0]  # Берем первый канал
        
        # Обработка изображения
        img[img == -1000] = 0  # Заменить -1000 на 0
        
        # Z-нормализация изображения
        mean = img.mean()
        std = img.std() if img.std() > 0 else 1.0
        img = (img - mean) / std
        
        if self.use_dual_channel and mask_path:
            # Загружаем маску
            mask = nib.load(mask_path).get_fdata().astype(np.float32)
            
            # Нормализация маски (приведение к 0-1)
            if mask.max() > 1:
                mask = mask / mask.max()
            
            # Создаем двухканальный тензор: [изображение, маска]
            dual_channel = np.stack([img, mask], axis=0)  # (2, D, H, W)
        else:
            # Одноканальный режим
            dual_channel = np.expand_dims(img, 0)  # (1, D, H, W)
        
        # Аугментация
        if self.augment:
            dual_channel = self._apply_augmentations(dual_channel)
        
        dual_channel = dual_channel.copy()  # fix negative strides
        
        # Загружаем grading
        grading = pd.read_csv(grading_path).iloc[0].values.astype(np.float32)
        
        # Pfirrman grade (последний столбец) привести к диапазону 0-4
        grading[-1] -= 1
        
        # Проверка диапазонов
        limits = [
            (0, 3),  # Modic
            (0, 1),  # UP endplate
            (0, 1),  # LOW endplate
            (0, 1),  # Spondylolisthesis
            (0, 1),  # Disc herniation
            (0, 1),  # Disc narrowing
            (0, 1),  # Disc bulging
            (0, 4),  # Pfirrman grade (0-4)
        ]
        
        for i, (val, (lo, hi)) in enumerate(zip(grading, limits)):
            if not (lo <= val <= hi):
                raise ValueError(f"Target value out of range in {grading_path}: index {i}, value {val}, expected [{lo}, {hi}]. Full row: {grading}")
        
        return torch.from_numpy(dual_channel), torch.from_numpy(grading)
    
    def _apply_augmentations(self, volume):
        """Применяет аугментации к объему."""
        # Случайный гауссов шум для изображения (только первый канал)
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.1, size=volume[0].shape)
            volume[0] = volume[0] + noise
        
        # Случайное изменение контраста для изображения (только первый канал)
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            volume[0] = volume[0] * alpha
        
        # Channel dropout для двухканального режима
        if volume.shape[0] == 2 and np.random.rand() > 0.7:  # 30% вероятность
            channel_to_drop = np.random.randint(0, 2)
            volume[channel_to_drop] = 0.0
        
        return volume


# --- FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, input, target):
        ce_loss = nn.functional.cross_entropy(input, target, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DualChannelGradingModel(nn.Module):
    """
    Двухканальная модель grading с улучшенной архитектурой.
    Основана на GradingModel из Deep_Spine_inf с поддержкой переменного количества каналов.
    """
    
    def __init__(self, block=BasicBlock, layers: List[int] = [3, 4, 6, 3], 
                 num_input_channels: int = 2, zero_init_residual: bool = False, 
                 groups: int = 1, width_per_group=64, norm_layer=None, dropout_rate=0.5):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        
        # Входной слой для переменного количества каналов
        self.conv1 = nn.Conv3d(num_input_channels, self.inplanes, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # ResNet слои
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(dropout_rate)
        
        # Классификационные головы
        feature_dim = 512 * block.expansion
        
        self.fc_modic = nn.Linear(feature_dim, 4)
        self.fc_up_endplate = nn.Linear(feature_dim, 2)
        self.fc_low_endplate = nn.Linear(feature_dim, 2)
        self.fc_spondy = nn.Linear(feature_dim, 2)
        self.fc_hern = nn.Linear(feature_dim, 2)
        self.fc_narrow = nn.Linear(feature_dim, 2)
        self.fc_bulge = nn.Linear(feature_dim, 2)
        self.fc_pfirrman = nn.Linear(feature_dim, 5)
        
        # Инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, (1, 1, 1), stride=(1, stride, stride), bias=False),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Входной блок
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet блоки
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global pooling и dropout
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        
        # Классификационные головы
        x_modic = self.fc_modic(x)
        x_up_endplate = self.fc_up_endplate(x)
        x_low_endplate = self.fc_low_endplate(x)
        x_spondy = self.fc_spondy(x)
        x_hern = self.fc_hern(x)
        x_narrow = self.fc_narrow(x)
        x_bulge = self.fc_bulge(x)
        x_pfirrman = self.fc_pfirrman(x)
        
        return (
            x_modic,
            x_up_endplate,
            x_low_endplate,
            x_spondy,
            x_hern,
            x_narrow,
            x_bulge,
            x_pfirrman,
        )


# --- Обучение ---
def train_val_split(dataset, val_ratio=0.2, seed=42):
    """Разделение датасета на обучающую и валидационную выборки."""
    idxs = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(idxs, test_size=val_ratio, random_state=seed, shuffle=True)
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)


def train_dual_channel_model(folds=5, seed=42, epochs=20, batch_size=8, 
                            train_head_only=False, head_idx=0, lr=1e-3,
                            model_save_path='dual_channel_grading.pth',
                            use_dual_channel=True):
    """
    Обучение двухканальной модели grading.
    
    Args:
        folds: Количество фолдов для кросс-валидации
        seed: Seed для воспроизводимости
        epochs: Количество эпох обучения
        batch_size: Размер бат��а
        train_head_only: Обучать только одну голову
        head_idx: Индекс головы для обучения (если train_head_only=True)
        lr: Learning rate
        model_save_path: Путь для сохранения модели
        use_dual_channel: Использовать двухканальный режим
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используется устройство: {device}")
    
    # Создаем датасет
    dataset = DualChannelGradingDataset(OUTPUT_DIR, augment=True, use_dual_channel=use_dual_channel)
    logger.info(f"Размер датасета: {len(dataset)}")
    
    if len(dataset) == 0:
        logger.error("Датасет пуст! Проверьте путь к данным.")
        return
    
    # Получаем метки Modic для стратификации
    targets = []
    for sample in tqdm(dataset.samples, desc="Загрузка меток"):
        if use_dual_channel:
            _, _, grading_path = sample
        else:
            _, _, grading_path = sample
        grading = pd.read_csv(grading_path).iloc[0].values.astype(np.float32)
        grading[-1] -= 1
        targets.append(int(grading[0]))  # Modic
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    all_metrics = []
    
    # Создаем модель
    num_channels = 2 if use_dual_channel else 1
    model = DualChannelGradingModel(num_input_channels=num_channels).to(device)
    
    # Загружаем предобученную модель, если существует
    if os.path.exists(model_save_path):
        logger.info(f'Загружаем предобученную модель: {model_save_path}')
        try:
            model.load_state_dict(torch.load(model_save_path, map_location=device))
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель: {e}")
    
    # Опциональное замораживание всех параметров кроме одной головы
    head_names = [
        'fc_modic', 'fc_up_endplate', 'fc_low_endplate', 'fc_spondy',
        'fc_hern', 'fc_narrow', 'fc_bulge', 'fc_pfirrman'
    ]
    
    if train_head_only:
        logger.info(f"[!] Обучение только головы: {head_names[head_idx]} (idx={head_idx})")
        for param in model.parameters():
            param.requires_grad = False
        getattr(model, head_names[head_idx]).weight.requires_grad = True
        getattr(model, head_names[head_idx]).bias.requires_grad = True
    
    # Кросс-валидация
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), targets), 1):
        logger.info(f"---- Fold {fold}/{folds} ----")
        
        # Oversampling по Modic
        y_train = np.array([targets[i] for i in train_idx])
        class_counts = np.bincount(y_train, minlength=4)  # 4 класса для Modic
        class_weights_os = {cls: 1.0/count if count > 0 else 0.0 for cls, count in enumerate(class_counts)}
        sample_weights = [class_weights_os[y_train[i]] for i in range(len(y_train))]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        
        # Создаем DataLoader'ы
        train_set = Subset(dataset, list(train_idx))
        val_set = Subset(dataset, list(val_idx))
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        # Подсчет весов для каждой головы
        logger.info("Вычисление весов классов...")
        all_targets = [[] for _ in range(8)]
        for sample in tqdm(dataset.samples):
            if use_dual_channel:
                _, _, grading_path = sample
            else:
                _, _, grading_path = sample
            grading = pd.read_csv(grading_path).iloc[0].values.astype(np.float32)
            grading[-1] -= 1
            for i in range(8):
                all_targets[i].append(int(grading[i]))
        
        # Количество классов для каждой головы
        n_classes = [4, 2, 2, 2, 2, 2, 2, 5]
        class_weights_heads = []
        
        for i, targets_head in enumerate(all_targets):
            vals, counts = np.unique(targets_head, return_counts=True)
            w = np.zeros(n_classes[i], np.float32)
            for v, c in zip(vals, counts):
                w[int(v)] = 1.0 / c if c > 0 else 0.0
            if w.sum() > 0:
                w = w / w.sum() * n_classes[i]
            class_weights_heads.append(torch.tensor(w, device=device))
        
        # Создаем функции потерь
        criterions = [(nn.CrossEntropyLoss(weight=class_weights_heads[i]),
                       FocalLoss(alpha=class_weights_heads[i])) for i in range(8)]
        
        # Оптимизатор
        if train_head_only:
            params = list(getattr(model, head_names[head_idx]).parameters())
        else:
            params = model.parameters()
        
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-7)
        
        best_val_f1 = 0
        f1_window: list[float] = []
        
        # Обучение
        for epoch in range(1, epochs+1):
            # TRAIN
            model.train()
            train_loss = 0
            
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
                x, y = x.float().to(device), y.float().to(device)
                optimizer.zero_grad()
                
                outs = model(x)
                
                if train_head_only:
                    # Только одна голова
                    out_i = outs[head_idx]
                    target_i = y[:, head_idx].long()
                    ce, focal = criterions[head_idx]
                    loss = ce(out_i, target_i) + focal(out_i, target_i)
                else:
                    loss = 0
                    for i, (ce, focal) in enumerate(criterions):
                        out_i = outs[i]
                        target_i = y[:, i].long()
                        loss += ce(out_i, target_i) + focal(out_i, target_i)
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # VALIDATION
            model.eval()
            val_loss = 0
            preds, trues = [[] for _ in range(8)], [[] for _ in range(8)]
            
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                    x, y = x.float().to(device), y.float().to(device)
                    outs = model(x)
                    
                    if train_head_only:
                        out_i = outs[head_idx]
                        target_i = y[:, head_idx].long()
                        ce, focal = criterions[head_idx]
                        preds[head_idx].extend(out_i.argmax(1).cpu().numpy())
                        trues[head_idx].extend(target_i.cpu().numpy())
                        val_loss += ce(out_i, target_i).item() + focal(out_i, target_i).item()
                    else:
                        for i, o in enumerate(outs):
                            preds[i].extend(o.argmax(1).cpu().numpy())
                            trues[i].extend(y[:, i].cpu().numpy())
                            val_loss += criterions[i][0](o, y[:, i].long()).item() + criterions[i][1](o, y[:, i].long()).item()
            
            val_loss /= len(val_loader)
            
            # Метрики
            if train_head_only:
                acc = accuracy_score(trues[head_idx], preds[head_idx])
                f1 = f1_score(trues[head_idx], preds[head_idx], average='macro')
                kappa = cohen_kappa_score(trues[head_idx], preds[head_idx])
                mean_f1 = float(f1)
                logger.info(f"Fold {fold} Epoch {epoch:02d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {acc:.4f} | Val f1: {f1:.4f}")
                logger.info(f"  Kappa: {kappa:.4f}")
            else:
                accs = [accuracy_score(trues[i], preds[i]) for i in range(8)]
                f1s = [f1_score(trues[i], preds[i], average='macro') for i in range(8)]
                kappas = [cohen_kappa_score(trues[i], preds[i]) for i in [0, 7]]
                
                # Приводим все значения f1s к float
                f1s = [float(f) if not isinstance(f, float) else f for f in f1s]
                mean_f1 = float(np.mean(f1s))
                
                logger.info(f"Fold {fold} Epoch {epoch:02d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {np.mean(accs):.4f} | Val f1: {mean_f1:.4f}")
                logger.info("  Accuracies:", [f"{acc:.4f}" for acc in accs])
                logger.info("  F1s:", [f"{f:.4f}" for f in f1s])
                logger.info("  Kappas (Modic,Pfirrman):", [f"{k:.4f}" for k in kappas])
            
            f1_window.append(mean_f1)
            if len(f1_window) > 3:
                f1_window.pop(0)
            smooth_f1 = float(np.mean(f1_window))
            logger.info(f"  Smooth F1: {smooth_f1:.4f}")
            
            scheduler.step(mean_f1)
            
            # Сохранение лучшей модели фолда
            if smooth_f1 > best_val_f1:
                best_val_f1 = smooth_f1
                torch.save(model.state_dict(), model_save_path)
                logger.info("  [*] Модель сохранена!")
        
        logger.info(f"== Fold {fold} Best F1: {best_val_f1:.4f}\n")
        all_metrics.append(best_val_f1)
    
    logger.info(f"Cross-val F1: mean={np.mean(all_metrics):.4f}, std={np.std(all_metrics):.4f}")


def create_dual_channel_model(model_path: Optional[str] = None, 
                             num_input_channels: int = 2) -> DualChannelGradingModel:
    """
    Создает и загружает двухканальную модель grading.
    
    Args:
        model_path: Путь к весам модели (опционально)
        num_input_channels: Количество входных каналов (1 или 2)
        
    Returns:
        Загруженная модель
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualChannelGradingModel(num_input_channels=num_input_channels).to(device)
    
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Модель загружена из: {model_path}")
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель: {e}")
            logger.info("Создана новая модель (веса не загружены)")
    else:
        logger.info("Создана новая модель (веса не загружены)")
    
    model.eval()
    return model


if __name__ == "__main__":
    # Обучение двухканальной модели
    logger.info("=== Обучение двухканальной модели grading ===")
    train_dual_channel_model(
        epochs=15, 
        lr=1e-3, 
        batch_size=6, 
        use_dual_channel=True,
        model_save_path='dual_channel_grading.pth'
    )
    
    # Дообучение отдельных голов
    logger.info("\n=== Дообучение отдельных голов ===")
    for idx in range(8):
        logger.info(f"\nОбучение головы {idx}")
        train_dual_channel_model(
            epochs=10, 
            train_head_only=True, 
            head_idx=idx, 
            lr=1e-4, 
            batch_size=8,
            use_dual_channel=True,
            model_save_path='dual_channel_grading.pth'
        )