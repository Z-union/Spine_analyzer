import os
import json
import random
import pandas as pd
from collections import Counter

import timm
import cv2
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms as T
from PIL import Image

import torch.cuda.amp as amp
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

import logging

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def is_one_hot(tensor: torch.Tensor) -> bool:
    result = bool((
        tensor.ndim == 2 and                     # [batch_size, num_classes]
        (tensor.sum(dim=1) == 1).all().item() and       # сумма по классам = 1
        ((tensor == 0) | (tensor == 1)).all().item()    # значения только 0 или 1
    ))
    return result

def label_smoothing(one_hot_labels, epsilon=0.1):
    num_classes = one_hot_labels.shape[-1]
    return (1 - epsilon) * one_hot_labels + (epsilon / num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def train_model_multilabel(model, dataloader_train, dataloader_val, criterion, optimizer, device,
                           num_classes, samples_per_class, train_dataset, num_epochs=2, only_val=False, forbidden_class=None,
                           label2idx=None):
    """
    Обучение модели для мультилейблинга, где метки представлены в виде multi-hot векторов.
    Предполагается, что последний класс (индекс num_classes) соответствует no object.
    Если label2idx передан, используется для построения confusion matrix.
    Помимо этого, в конце эпохи вычисляются и логируются метрики precision, recall, F1, ROC-AUC и average precision по каждому классу.
    """
    if forbidden_class is None:
        forbidden_class = num_classes

    model.to(device)
    best_val_metric = 0
    scaler = amp.GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs} - Начало обучения")
        if not only_val:
            model.train()
            running_loss = 0.0

            for images, targets in tqdm(dataloader_train, desc="Обучение"):
                images, targets = images.to(device), targets.to(device)
                mixed_targets = label_smoothing(targets)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, mixed_targets[:, :-1], samples_per_class[:-1])
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)

            epoch_loss = running_loss / len(dataloader_train.dataset)
            print(f"Train Loss: {epoch_loss:.4f}")

        model.eval()
        running_loss_val = 0.0
        all_probs, all_targets, all_targets_norm = [], [], []

        with torch.no_grad():
            for images, targets in tqdm(dataloader_val, desc="Валидация"):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets[:, :-1], samples_per_class[:-1])
                outputs_np = F.sigmoid(outputs).cpu().numpy()
                targets_np = targets[:, :-1].cpu().numpy()
                running_loss_val += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(targets[:, :-1].cpu().numpy())
                all_targets_norm.append(targets[:, :].cpu().numpy())

        val_loss = running_loss_val / len(dataloader_val.dataset)
        print(f"Val Loss: {val_loss:.4f}")

        all_probs_np = np.concatenate(all_probs, axis=0)
        all_targets_np = np.concatenate(all_targets, axis=0)
        all_targets_norm_np = np.concatenate(all_targets_norm, axis=0)

        optimal_thresholds = find_optimal_thresholds(all_targets_np, all_probs_np)
        preds = np.zeros_like(all_probs_np)
        for cls in range(all_probs_np.shape[1]):
            preds[:, cls] = (all_probs_np[:, cls] >= optimal_thresholds[cls][0]).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(all_targets_np.astype(int), preds.astype(int), average=None)
        roc_auc = roc_auc_score(all_targets_np, all_probs_np, average='macro')
        avg_precision = average_precision_score(all_targets_np, all_probs_np, average='macro')

        if label2idx is not None:
            idx2label = {v: k for k, v in label2idx.items()}
        else:
            idx2label = {i: str(i) for i in range(all_probs_np.shape[1])}

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.float_format', '{:.2f}'.format)

        metrics_df = pd.DataFrame({
            "Class": [idx2label[i] for i in range(all_probs_np.shape[1])],
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "ROC-AUC": roc_auc,
            "Average Precision": avg_precision,
            "Optimal Threshold": [optimal_thresholds[i][0] for i in range(all_probs_np.shape[1])],
            "Optimal F1": [optimal_thresholds[i][1] for i in range(all_probs_np.shape[1])]
        })

        print(metrics_df.to_string(index=False))

        macro_precision = np.nanmean(precision)
        macro_recall = np.nanmean(recall)
        macro_f1 = np.nanmean(f1)
        print(f"Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}")

        if macro_f1 > best_val_metric:
            best_val_metric = macro_f1
            torch.save(model, "best_model_v3.pth")
            print("Best model saved as best_model_v3.pth")
        torch.save(model, "last_model_v3.pth")
        print("Last model saved as last_model_v3.pth")

        scheduler.step()

    return model

def find_optimal_threshold_for_class(y_true_binary, y_prob):
    """
    Находит оптимальный порог для одного класса, максимизирующий F1-score.
    y_true_binary: массив (numpy array) бинарных истинных меток (0 или 1) для данного класса.
    y_prob: массив вероятностей для данного класса.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true_binary, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    if len(thresholds) == 0:
        return 0.5, f1_scores[optimal_idx]
    return thresholds[optimal_idx], f1_scores[optimal_idx]

def find_optimal_thresholds(all_targets_one_hot, all_probs):
    optimal_thresholds = {}
    all_targets_one_hot = np.array(all_targets_one_hot)
    all_probs = np.array(all_probs)
    for i in range(all_probs.shape[1]):
        y_true_i = all_targets_one_hot[:, i]
        y_prob_i = all_probs[:, i]
        thr, f1_val = find_optimal_threshold_for_class(y_true_i, y_prob_i)
        optimal_thresholds[i] = (thr, f1_val)
    return optimal_thresholds

def calculate_class_weights(df, label2idx, device):
    """
    Вычисляет веса классов и количество примеров каждого класса.

    Возвращает:
        - Тензор весов классов, упорядоченный по индексам классов из label2idx.
        - Тензор количества примеров каждого класса в том же порядке.
    """
    counts = Counter(df['label'])
    total = sum(counts.values())
    num_classes = len(label2idx)

    samples_per_class = [counts[label] for label in sorted(label2idx, key=lambda x: label2idx[x])]
    weights = [total / (count * num_classes) for count in samples_per_class]

    return (torch.log1p(torch.tensor(weights, dtype=torch.float, device=device)),
            torch.tensor(samples_per_class, dtype=torch.float, device=device))

def check_mapping(train_df, val_df):
    """
    Проверяет, что метки классов в train и val наборах совпадают.
    """
    train_labels = set(train_df['label'].unique())
    val_labels = set(val_df['label'].unique())

    if train_labels != val_labels:
        print(f"Предупреждение: Метки классов не совпадают между train и val наборами!")
        print(f"Метки в train наборе: {train_labels}")
        print(f"Метки в val наборе: {val_labels}")
    else:
        print("Mapping меток между train и val наборами совпадает.")

class EyesDataset(Dataset):
    def __init__(self, df, transform=None, label2idx=None):
        """
        df: DataFrame с колонками ['image', 'label']
        transform: torchvision.transforms для обработки изображения
        label2idx: словарь для преобразования метки в индекс. Если не задан, он будет сформирован автоматически.
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        if label2idx is None:
            all_labels = set()
            for lab in df['label']:
                if isinstance(lab, str):
                    labels = [l.strip() for l in lab.split('+')]
                elif isinstance(lab, list):
                    labels = lab
                else:
                    raise RuntimeError("Неподдерживаемый тип метки")
                all_labels.update(labels)
            labels = sorted(list(all_labels))
            self.label2idx = {label: idx for idx, label in enumerate(labels)}
        else:
            self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image']
        label = row['label']
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Ошибка чтения изображения {image_path}: {e}")

        if self.transform:
            image = self.transform(image)

        multihot = np.zeros(len(self.label2idx), dtype=np.float32)

        if isinstance(label, str):
            labels = [l.strip() for l in label.split('+')]
        elif isinstance(label, list):
            labels = label
        else:
            raise RuntimeError("Неподдерживаемый тип метки")

        for lab in labels:
            if lab not in self.label2idx:
                raise RuntimeError(f"Метка {lab} отсутствует в label2idx")
            idx_lab = self.label2idx[lab]
            multihot[idx_lab] = 1.0

        return image, multihot

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if targets.dim() == 1 or (targets.dim() == 2 and targets.size(1) == 1):
            num_classes = inputs.size(1)
            targets = torch.nn.functional.one_hot(targets.long().view(-1), num_classes=num_classes).float()
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

def partial_bce(inputs, targets, threshold=0.1):
    if targets.dim() == 1 or (targets.dim() == 2 and targets.size(1) == 1):
        num_classes = inputs.size(1)
        targets = torch.nn.functional.one_hot(targets.long().view(-1), num_classes=num_classes).float()

    neg_mask = (targets == 0) & (inputs > threshold)
    bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    bce[~neg_mask & (targets == 0)] = 0
    return bce.mean()

def cb_loss(inputs, targets, samples_per_class, beta=0.9999):
    if targets.dim() == 1 or (targets.dim() == 2 and targets.size(1) == 1):
        num_classes = inputs.size(1)
        targets = torch.nn.functional.one_hot(targets.long().view(-1), num_classes=num_classes).float()

    no_of_classes = inputs.shape[1]

    effective_num = 1.0 - torch.pow(beta, samples_per_class.float())
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / weights.sum() * no_of_classes

    weights = weights.to(inputs.device)
    weights = weights.unsqueeze(0).repeat(targets.shape[0], 1) * targets
    weights = weights.sum(1).unsqueeze(1).repeat(1, no_of_classes)

    loss = F.binary_cross_entropy_with_logits(inputs, targets, weights)
    return loss

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, logits, targets):
        if targets.dim() == 1 or (targets.dim() == 2 and targets.size(1) == 1):
            num_classes = logits.size(1)
            targets = torch.nn.functional.one_hot(targets.long().view(-1), num_classes=num_classes).float()
        targets = targets.float()

        pred_probs = torch.sigmoid(logits)
        if self.clip is not None and self.clip > 0:
            pred_probs = pred_probs.clamp(min=self.clip, max=1 - self.clip)

        pos_loss = targets * torch.log(pred_probs + self.eps)
        neg_loss = (1 - targets) * torch.log(1 - pred_probs + self.eps)

        if self.gamma_pos > 0 or self.gamma_neg > 0:
            with torch.set_grad_enabled(not self.disable_torch_grad_focal_loss):
                pt_pos = pred_probs * targets
                pt_neg = (1 - pred_probs) * (1 - targets)
                pos_loss *= (1 - pt_pos) ** self.gamma_pos
                neg_loss *= pt_neg ** self.gamma_neg

        loss = - (pos_loss + neg_loss)
        return loss.mean()

def process_dataset(images_folder, json_path, gray_folder, convert_to_gray=True):
    """
    Обрабатывает изображения и формирует DataFrame с путями и метками.
    """
    os.makedirs(gray_folder, exist_ok=True)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    annotations = data['annotations']
    data_list = []

    for image_name, info in tqdm(annotations.items(), desc=f"Обработка изображений из {json_path}"):
        pathology_labels = info.get('pathologies', [])
        image_path = os.path.join(images_folder, image_name)
        if convert_to_gray:
            gray_image_path = os.path.join(gray_folder, image_name)
            converted_path = image_path
        else:
            converted_path = image_path

        if converted_path and pathology_labels:
            data_list.append({'image': converted_path, 'label': pathology_labels[0]})
    if not data_list:
        print(f"Нет корректных изображений в {json_path}!")
    return pd.DataFrame(data_list)

def oversample_data(df: pd.DataFrame, only_downsampling: bool = True) -> pd.DataFrame:
    """
    Если only_downsampling=True, то выполняется только undersampling для класса Normal (если он является большинством)
    и возвращается датафрейм после уменьшения количества примеров для Normal.
    """
    y = df["label"].tolist()
    counts = Counter(y)

    if "Normal" in counts and counts["Normal"] == max(counts.values()):
        non_normal_counts = [count for cls, count in counts.items() if cls != "Normal"]
        if non_normal_counts:
            new_max = max(non_normal_counts)
        else:
            new_max = counts["Normal"]
        normal_indices = [i for i, label in enumerate(y) if label == "Normal"]
        random.seed(42)
        selected_normal = set(random.sample(normal_indices, new_max))
        selected_indices = [i for i, label in enumerate(y) if label != "Normal"] + list(selected_normal)
        df = df.iloc[selected_indices].reset_index(drop=True)

    return df

def create_model(num_classes):
    """
    Создает и инициализирует модель CAFormer.
    
    Args:
        num_classes: количество классов (без учета forbidden class)
    Returns:
        model: инициализированная модель
    """
    model = timm.create_model('caformer_b36.sail_in22k_ft_in1k', pretrained=True, num_classes=num_classes)
    
    # Добавляем dropout для регуляризации
    model.head.drop = nn.Dropout(p=0.2, inplace=False)
    model.head.fc.head_drop = nn.Dropout(p=0.2, inplace=False)
    
    return model

def train_fold(train_df, val_df, label2idx, mean, std, include_oversampling=False, convert_to_gray=False):
    num_classes = len(label2idx)
    model = create_model(num_classes-1)

    base_transforms = [
        GrayscaleToRGB(),
        T.Resize(384),
        T.CenterCrop(384),
        T.ToTensor(),
    ]

    transforms_train = T.Compose([
        *base_transforms,
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05)
        ),
        T.Normalize(mean=mean, std=std)
    ])
    transforms_val = T.Compose([
        *base_transforms,
        T.Normalize(mean=mean, std=std)
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_weight_tensor, samples_per_class = calculate_class_weights(train_df, label2idx, device)
    targets = train_df['label'].map(label2idx).tolist()
    class_weights = class_weight_tensor.cpu().numpy()
    sample_weights = [float(class_weights[t]) for t in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    criterion = lambda pred, target, samples_per_class: (
            0.5 * FocalLoss(gamma=4.0)(pred, target) +
            0.5 * cb_loss(pred, target, samples_per_class, beta=0.7) +
            0.3 * AsymmetricLoss()(pred, target) +
            0.3 * partial_bce(pred, target))
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)

    train_dataset = EyesDataset(train_df, transform=transforms_train, label2idx=label2idx)
    val_dataset = EyesDataset(val_df, transform=transforms_val, label2idx=label2idx)

    batch_size = 12
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4,
                                  persistent_workers=True, pin_memory=True, drop_last=True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                persistent_workers=True, pin_memory=True, drop_last=True)

    trained_model = train_model_multilabel(model, dataloader_train, dataloader_val, criterion, optimizer, device, 
                                         num_classes, samples_per_class, train_dataset, num_epochs=10, 
                                         label2idx=label2idx)

    torch.save(trained_model.state_dict(), f"final_model_fold.pth")
    print("Финальная модель сохранена как final_model_fold.pth")
    return trained_model

class GrayscaleToRGB:
    def __call__(self, img):
        # Convert to grayscale using PIL
        gray_img = img.convert('L')
        # Convert back to RGB by duplicating the channel
        return Image.merge('RGB', (gray_img, gray_img, gray_img))

def calculate_dataset_stats(dataloader):
    """Calculate mean and std of the dataset for z-normalization"""
    total_sum = 0
    total_sq_sum = 0
    total_samples = 0

    for images, _ in tqdm(dataloader, desc="Calculating dataset statistics"):
        # images are already grayscale duplicated to 3 channels, so we can just use one channel
        images_single = images[:, 0, :, :]  # Take first channel
        batch_samples = images.size(0) * images.size(2) * images.size(3)
        total_sum += images_single.sum().item()
        total_sq_sum += (images_single ** 2).sum().item()
        total_samples += batch_samples

    mean = total_sum / total_samples
    # Calculate std: sqrt(E[X^2] - (E[X])^2)
    std = (total_sq_sum / total_samples - mean ** 2) ** 0.5
    
    # Return the same stats for all channels since they're identical
    return [mean] * 3, [std] * 3

def prepare_full_df_and_labels(images_folder, json_path, gray_folder, convert_to_gray=True):
    # Чтение и обработка исходных данных
    full_df = process_dataset(images_folder, json_path, gray_folder, convert_to_gray)
    # Собираем все уникальные лейблы
    all_labels = set()
    for labs in full_df['label']:
        if isinstance(labs, str):
            labs = [l.strip() for l in labs.split('+')]
        all_labels.update(labs)
    labels = sorted(list(all_labels))
    label2idx = {label: idx for idx, label in enumerate(labels)}
    return full_df, label2idx

def get_multilabels(df, label2idx):
    multilabels = []
    for labs in df['label']:
        if isinstance(labs, str):
            labs = [l.strip() for l in labs.split('+')]
        arr = np.zeros(len(label2idx), dtype=int)
        for l in labs:
            arr[label2idx[l]] = 1
        multilabels.append(arr)
    return np.array(multilabels)

def run_crossval(full_df, label2idx, n_splits=5, mean=None, std=None, **train_kwargs):
    multilabels = get_multilabels(full_df, label2idx)
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    for fold, (train_idx, val_idx) in enumerate(mskf.split(full_df, multilabels)):
        print(f'--- Fold {fold+1}/{n_splits} ---')
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)
        result = train_fold(train_df, val_df, label2idx, mean, std, **train_kwargs)
        results.append(result)
    return results

if __name__ == "__main__":
    train_images_folder = r"C:\Buffer\eyes_dataset\train_2"
    train_json_path = r"C:\Buffer\eyes_dataset\train_2.json"
    train_gray_folder = r"C:\Buffer\eyes_dataset\train_rgb_2"

    tuning_images_folder = r"C:\Buffer\eyes_dataset\test_2"
    tuning_json_path = r"C:\Buffer\eyes_dataset\test_2.json"
    tuning_gray_folder = r"C:\Buffer\eyes_dataset\test_rgb_2"

    # 1. Объединяем train и test
    train_df, _ = prepare_full_df_and_labels(train_images_folder, train_json_path, train_gray_folder, convert_to_gray=False)
    test_df, _ = prepare_full_df_and_labels(tuning_images_folder, tuning_json_path, tuning_gray_folder, convert_to_gray=False)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    # 2. Строим label2idx по всему датасету
    all_labels = set()
    for labs in full_df['label']:
        if isinstance(labs, str):
            labs = [l.strip() for l in labs.split('+')]
        all_labels.update(labs)
    labels = sorted(list(all_labels))
    label2idx = {label: idx for idx, label in enumerate(labels)}

    # 3. Считаем статистики по всему датасету
    base_transforms = [
        GrayscaleToRGB(),
        T.Resize(384),
        T.CenterCrop(384),
        T.ToTensor(),
    ]
    temp_dataset = EyesDataset(full_df, transform=T.Compose(base_transforms), label2idx=label2idx)
    temp_loader = DataLoader(temp_dataset, batch_size=32, num_workers=4, shuffle=False)
    print("Calculating dataset statistics for z-normalization...")
    mean, std = calculate_dataset_stats(temp_loader)
    print(f"Dataset statistics - Mean: {mean}, Std: {std}")

    # 4. Кросс-валидация на всём датасете
    results = run_crossval(full_df, label2idx, n_splits=5, mean=mean, std=std, include_oversampling=True, convert_to_gray=False)

# ... rest of the code ... 