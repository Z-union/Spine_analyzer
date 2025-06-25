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
from typing import Tuple, List
from tqdm import tqdm

# Путь к подготовленному датасету (корректируйте при необходимости)
OUTPUT_DIR = r"F:\WorkSpace\Z-Union\chess\10159290\dataset_cuts"

class GradingDataset(Dataset):
    def __init__(self, output_dir, augment=False):
        self.output_dir = output_dir
        self.augment = augment
        self.samples = []
        for img_path in glob.glob(os.path.join(output_dir, '*_img.nii.gz')):
            base = img_path.replace('_img.nii.gz', '')
            grading_path = base + '_grading.csv'
            if os.path.exists(grading_path):
                self.samples.append((img_path, grading_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, grading_path = self.samples[idx]
        img = nib.load(img_path).get_fdata().astype(np.float32)
        # Заменить -1000 на 0
        img[img == -1000] = 0
        # Z-нормализация по всему объёму
        mean = img.mean()
        std = img.std() if img.std() > 0 else 1.0
        img = (img - mean) / std
        # Аугментация: случайное зеркалирование по всем осям, поворот, шум, контраст
        if self.augment:
            # # Случайное зеркалирование
            # for axis in range(3):
            #     if np.random.rand() > 0.5:
            #         img = np.flip(img, axis=axis)
            # # Случайный поворот на 90 градусов по случайной оси
            # if np.random.rand() > 0.5:
            #     axes = [(0,1), (0,2), (1,2)]
            #     ax = axes[np.random.randint(0, 3)]
            #     k = np.random.randint(1, 4)
            #     img = np.rot90(img, k=k, axes=ax)
            # Случайный гауссов шум
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.1, size=img.shape)
                img = img + noise
            # Случайное изменение контраста
            if np.random.rand() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                img = img * alpha
        img = np.expand_dims(img, 0)  # (1, D, H, W)
        img = img.copy()  # fix negative strides after np.flip/rot90
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
        return torch.from_numpy(img), torch.from_numpy(grading)

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

# --- СЕТЬ И БЛОКИ ---
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        (3, 3, 3),
        stride=(1, stride, stride),
        padding=(1, dilation, dilation),
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        (1, 1, 1),
        stride=(1, stride, stride),
        padding=(0, 0, 0),
        bias=False,
    )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class GradingModel(nn.Module):
    def __init__(self, block=BasicBlock, layers: List[int] = [3, 4, 6, 3], num_classes: int = 2, zero_init_residual: bool = False, groups: int = 1, width_per_group=64, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(1, self.inplanes, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_modic = nn.Linear(512 * block.expansion, 4)
        self.fc_up_endplate = nn.Linear(512 * block.expansion, 2)
        self.fc_low_endplate = nn.Linear(512 * block.expansion, 2)
        self.fc_spondy = nn.Linear(512 * block.expansion, 2)
        self.fc_hern = nn.Linear(512 * block.expansion, 2)
        self.fc_narrow = nn.Linear(512 * block.expansion, 2)
        self.fc_bulge = nn.Linear(512 * block.expansion, 2)
        self.fc_pfirrman = nn.Linear(512 * block.expansion, 5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
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
                conv1x1(self.inplanes, planes * block.expansion, stride),
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
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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
    idxs = np.arange(len(dataset))
    train_idx, val_idx = train_test_split(idxs, test_size=val_ratio, random_state=seed, shuffle=True)
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, val_idx)

def train(folds=5, seed=42, epochs=20, batch_size=12, train_head_only=False, head_idx=0, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = GradingDataset(OUTPUT_DIR, augment=True)
    # метки Modic для стратификации
    targets = []
    for _, grading_path in tqdm(dataset.samples):
        grading = pd.read_csv(grading_path).iloc[0].values.astype(np.float32)
        grading[-1] -= 1
        targets.append(int(grading[0]))  # Modic
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    all_metrics = []

    model = GradingModel().to(device)
    if os.path.exists('best_grading_model_fold5.pth'):
        print('loading best model')
        model.load_state_dict(torch.load('best_grading_model_fold5.pth'))

    # --- Опциональное замораживание всех параметров кроме одной головы ---
    head_names = [
        'fc_modic', 'fc_up_endplate', 'fc_low_endplate', 'fc_spondy',
        'fc_hern', 'fc_narrow', 'fc_bulge', 'fc_pfirrman'
    ]
    if train_head_only:
        print(f"[!] Training only head: {head_names[head_idx]} (idx={head_idx})")
        for param in model.parameters():
            param.requires_grad = False
        getattr(model, head_names[head_idx]).weight.requires_grad = True
        getattr(model, head_names[head_idx]).bias.requires_grad = True

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(dataset)), targets), 1):
        print(f"---- Fold {fold}/{folds} ----")
        # Oversampling по Modic
        y_train = np.array([targets[i] for i in train_idx])
        class_counts = np.bincount(y_train, minlength=4)  # 4 класса для Modic
        class_weights_os = {cls: 1.0/count if count > 0 else 0.0 for cls, count in enumerate(class_counts)}
        sample_weights = [class_weights_os[y_train[i]] for i in range(len(y_train))]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_set = Subset(dataset, list(train_idx))
        val_set = Subset(dataset, list(val_idx))
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        # подсчет весов для каждой головы
        print("Вычисление весов классов...")
        all_targets = [[] for _ in range(8)]
        for _, grading_path in tqdm(dataset.samples):
            grading = pd.read_csv(grading_path).iloc[0].values.astype(np.float32)
            grading[-1] -= 1
            for i in range(8): all_targets[i].append(int(grading[i]))
        # Жёстко задаём количество классов для каждой головы
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
        criterions = [(nn.CrossEntropyLoss(weight=class_weights_heads[i]),
                       FocalLoss(alpha=class_weights_heads[i])) for i in range(8)]

        # --- Оптимизатор ---
        if train_head_only:
            params = list(getattr(model, head_names[head_idx]).parameters())
        else:
            params = model.parameters()
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-7)

        best_val_f1 = 0
        f1_window: list[float] = []
        for epoch in range(1, epochs+1):
            # TRAIN
            model.train()
            train_loss = 0
            for x, y in train_loader:
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
                for x, y in val_loader:
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
                print(f"Fold {fold} Epoch {epoch:02d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {acc:.4f} | Val f1: {f1:.4f}")
                print(f"  Kappa: {kappa:.4f}")
            else:
                accs = [accuracy_score(trues[i], preds[i]) for i in range(8)]
                f1s = [f1_score(trues[i], preds[i], average='macro') for i in range(8)]
                kappas = [cohen_kappa_score(trues[i], preds[i]) for i in [0,7]]
                # Приводим все значения f1s к float (если вдруг f1_score вернул массив)
                f1s = [float(f) if not isinstance(f, float) else f for f in f1s]
                mean_f1 = float(np.mean(f1s))
                print(f"Fold {fold} Epoch {epoch:02d} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {np.mean(accs):.4f} | Val f1: {mean_f1:.4f}")
                print("  Accuracies:", [f"{acc:.4f}" for acc in accs])
                print("  F1s:", [f"{f:.4f}" for f in f1s])
                print("  Kappas (Modic,Pfirrman):", [f"{k:.4f}" for k in kappas])
            f1_window.append(mean_f1)
            if len(f1_window) > 3: f1_window.pop(0)
            smooth_f1 = float(np.mean(f1_window))
            print(f"  Smooth F1: {smooth_f1:.4f}")

            scheduler.step(mean_f1)
            # сохранение лучшей модели фолда
            if smooth_f1 > best_val_f1:
                best_val_f1 = smooth_f1
                # torch.save(model.state_dict(), f'best_grading_model_fold{fold}.pth')
                torch.save(model, f'grading.pth')
                print("  [*] Model saved!")

        print(f"== Fold {fold} Best F1: {best_val_f1:.4f}\n")
        all_metrics.append(best_val_f1)

    print(f"Cross-val F1: mean={np.mean(all_metrics):.4f}, std={np.std(all_metrics):.4f}")


if __name__ == "__main__":
    train(epochs=15, lr=1e-3)
    for idx in range(8):
        train(epochs=10, train_head_only=True, head_idx=idx ,lr=1e-4)