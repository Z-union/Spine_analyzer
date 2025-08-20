
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# Константы / ключи
# -----------------------
HEAD_KEYS = [
    "Modic",
    "UP endplate",
    "LOW endplate",
    "Spondylolisthesis",
    "Disc herniation",
    "Disc narrowing",
    "Disc bulging",
    "Pfirrmann grade",
]

MULTICLASS_HEADS = ["Modic", "Pfirrmann grade"]
BINARY_HEADS = [k for k in HEAD_KEYS if k not in MULTICLASS_HEADS]

# Добавляем UNKNOWN как дополнительный класс:
NUM_CLASSES = {
    "Modic": 5,            # 4 клинических + 1 UNKNOWN
    "Pfirrmann grade": 6   # 5 клинических + 1 UNKNOWN
}
UNKNOWN_LABEL = {
    k: NUM_CLASSES[k] - 1 for k in NUM_CLASSES
}

# Вероятность выключения ровно одного канала (перемешивается в датасете)
CHANNEL_DROP_PROB = 0.3

# -----------------------
# Архитектура и блоки
# -----------------------
class SEBlock3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)           # [B, C]
        y = self.fc(y).view(b, c, 1, 1, 1)        # [B, C, 1,1,1]
        return x * y


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        # kernel_size should be odd
        padding = kernel_size // 2
        # input channels = 2 (avg and max along channel)
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, D, H, W]
        # compute max and avg across channel
        max_pool, _ = torch.max(x, dim=1, keepdim=True)   # [B,1,D,H,W]
        avg_pool = torch.mean(x, dim=1, keepdim=True)     # [B,1,D,H,W]
        cat = torch.cat([max_pool, avg_pool], dim=1)      # [B,2,D,H,W]
        attn = self.conv(cat)                             # [B,1,D,H,W]
        attn = self.sigmoid(attn)
        return x * attn                                  # broadcast


class CBAM3D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = SEBlock3D(channels, reduction=reduction)
        self.spatial_attn = SpatialAttention3D(kernel_size=spatial_kernel)

    def forward(self, x):
        # channel attention then spatial attention (as in CBAM)
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


# -----------------------
# Residual block (unchanged)
# -----------------------
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False, use_se: bool = True):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.downsample = None
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )

        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock3D(out_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        if self.use_se:
            out = self.se(out)
        out += identity
        out = self.relu(out)
        return out


# -----------------------
# MultiHead model with coord + CBAM option
# -----------------------
class MultiHeadSpineModel(nn.Module):
    def __init__(self,
                 input_channels: int = 2,
                 dropout_p: float = 0.3,
                 depth: int = 4,
                 max_channels: int = 256,
                 use_coord: bool = True,
                 use_cbam: bool = True):
        """
        use_coord: добавлять 3 координатных канала (z,y,x) к входу
        use_cbam: применять CBAM3D перед глобальным пулом
        """
        super().__init__()
        self.use_coord = use_coord
        self.use_cbam = use_cbam

        self.base_in_channels = input_channels
        in_ch = input_channels + (3 if use_coord else 0)

        base_channels = 32
        channels_list = []
        for i in range(depth):
            ch = base_channels * (2 ** i)
            ch = min(ch, max_channels)
            channels_list.append(ch)

        layers = []
        for out_ch in channels_list:
            layers.append(ResidualBlock3D(in_ch, out_ch, downsample=True))
            in_ch = out_ch

        self.layers = nn.ModuleList(layers)

        final_channels = channels_list[-1]
        if use_cbam:
            self.cbam = CBAM3D(final_channels, reduction=16, spatial_kernel=7)
        else:
            self.cbam = None

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_p)

        def make_head(out_features: int):
            return nn.Sequential(
                nn.Linear(final_channels, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, out_features)
            )

        # out_dim_for — используй свою NUM_CLASSES/HEAD_KEYS как раньше
        def out_dim_for(key: str) -> int:
            if key in NUM_CLASSES:
                return NUM_CLASSES[key]
            return 1

        self.heads = nn.ModuleDict({key: make_head(out_dim_for(key)) for key in HEAD_KEYS})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, C=2, D, H, W]
        if self.use_coord:
            # build normalized coordinates in [-1,1]
            b, c, d, h, w = x.shape
            device = x.device
            z = torch.linspace(-1.0, 1.0, steps=d, device=device).view(1, 1, d, 1, 1).expand(b, 1, d, h, w)
            y = torch.linspace(-1.0, 1.0, steps=h, device=device).view(1, 1, 1, h, 1).expand(b, 1, d, h, w)
            xcoord = torch.linspace(-1.0, 1.0, steps=w, device=device).view(1, 1, 1, 1, w).expand(b, 1, d, h, w)
            coords = torch.cat([z, y, xcoord], dim=1)   # [B,3,D,H,W]
            x = torch.cat([x, coords], dim=1)           # [B, C+3, D,H,W]

        for layer in self.layers:
            x = layer(x)

        if self.cbam is not None:
            x = self.cbam(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)

        outputs: Dict[str, torch.Tensor] = {key: head(x) for key, head in self.heads.items()}
        return outputs

# -----------------------
# Loss: focal
# -----------------------
class FocalLossBinary(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: logits (B,), targets: float (0/1)
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        alpha = self.alpha if self.alpha is not None else 1.0
        focal_loss = alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class FocalLossMultiClass(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha  # tensor per class or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C), targets: (B,)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        if self.alpha is not None:
            # alpha per-target
            alpha = self.alpha[targets].to(inputs.device)
        else:
            alpha = 1.0
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss