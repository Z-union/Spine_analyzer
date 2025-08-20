CROP_SHAPE = (48, 68, 54)

# --- Константы ---
DEFAULT_CROP_MARGIN = 10
DEFAULT_CROP_SHAPE_PAD = (16, 16, 0)  # (D, H, W) добавка к crop_shape
LANDMARK_LABELS = [63, 64, 65, 66, 67, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 91, 92, 93, 94, 95, 96, 100]

# Пороговые значения и параметры анализа
HERN_THRESHOLD_STD = 1.5
BULGE_THRESHOLD_STD = 0.8
HERN_THRESHOLD_MAX = 0.9
BULGE_THRESHOLD_MAX = 0.7
MIN_HERNIA_SIZE = 10  # минимальный размер компонента для грыжи (вокселей)
MIN_BULGING_SIZE = 5  # минимальный размер компонента для выбухания (вокселей)
MAX_BULGING_SIZE = 100  # максимальный размер компонента для выбухания (вокселей)
MIN_BULGING_SHAPE_SIZE = 10  # минимальный размер компонента для выбухания по форме
MAX_BULGING_SHAPE_SIZE = 100  # максимальный размер компонента для выбухания по форме
BULGING_SHAPE_THRESHOLD_STD = 1.5  # порог для выбухания по форме
MIN_BULGING_COORDS = 5  # минимальное количество координат для выбухания
DILATE_SIZE = 5  # размер дилатации для largest_component
SPONDY_SIGNIFICANT_MM = 1.0  # минимальный листез, считающийся значимым
VERTEBRA_SEARCH_RANGE = 20  # диапазон поиска позвонков (1-20)
VERTEBRA_NEAR_DISK_DISTANCE = 50  # макс. расстояние до диска для поиска позвонков

# Ограничения расстояния патологий от границы диска
MAX_DISTANCE_FROM_DISK_BORDER = 15  # максимальное расстояние от границы диска в вокселях (увеличено)
MIN_DISTANCE_FROM_DISK_CENTER = 3  # минимальное расстояние от центра диска в вокселях
MIN_DISTANCE_FROM_DISK_BORDER = 0  # минимальное расстояние от границы (может начинаться прямо на границе)

# Метки для сегментации
CANAL_LABEL = 2
CORD_LABEL = 1
SACRUM_LABEL = 50
HERNIA_LABEL = 200
BULGING_LABEL = 201

# Размеры кропа и патчей
SAG_PATCH_SIZE = (128, 96, 96)
AX_PATCH_SIZE = (224, 224)

# Диапазоны для extract_alternate
EXTRACT_LABELS_RANGE = list(range(63, 101))

# Индексы классов в предсказаниях модели
IDX_MODIC = 0
IDX_UP_ENDPLATE = 1
IDX_LOW_ENDPLATE = 2
IDX_SPONDY = 3
IDX_HERN = 4
IDX_NARROW = 5
IDX_BULGE = 6
IDX_PFIRRMAN = 7

# Коэффициент вариации для симметрии выбухания
BULGING_SYMMETRY_CV = 0.3

# Цвета для разных типов структур (RGB)
COLORS = {
    'vertebra': (255, 255, 0),    # Желтый для позвонков
    'disk': (0, 255, 0),          # Зеленый для дисков
    'canal': (0, 0, 255),         # Синий для канала
    'cord': (255, 0, 0),          # Красный для спинного мозга
    'sacrum': (128, 0, 128),      # Фиолетовый для крестца
    'hernia': (255, 165, 0),      # Оранжевый для грыж
    'bulging': (255, 192, 203),   # Розовый для выбуханий
    'background': (0, 0, 0)       # Черный для фона
}

# Словарь для описания позвонков (универсальный)
VERTEBRA_DESCRIPTIONS = {
    63: 'C2-C3', 64: 'C3-C4', 65: 'C4-C5', 66: 'C5-C6', 67: 'C6-C7',
    71: 'C7-T1', 72: 'T1-T2', 73: 'T2-T3', 74: 'T3-T4', 75: 'T4-T5',
    76: 'T5-T6', 77: 'T6-T7', 78: 'T7-T8', 79: 'T8-T9', 80: 'T9-T10',
    81: 'T10-T11', 82: 'T11-T12', 91: 'T12-L1', 92: 'L1-L2', 93: 'L2-L3',
    94: 'L3-L4', 95: 'L4-L5', 96: 'L5-S1', 100: 'S1-S2'
}

