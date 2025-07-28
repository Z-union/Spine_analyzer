# Рефакторенный анализатор позвоночника

Этот проект представляет собой рефакторенную версию анализатора позвоночника с разделением ответственности и поддержкой различных источников данных и моделей.

## Основные принципы рефакторинга

### 1. Разделение ответственности

Код разделен на отдельные модули с четкими функциями:

- **Инициализация моделей** вынесена из пайплайна
- **Получение исследований** отделено от логики обработки
- **Обработка исследований** изолирована от источника данных

### 2. Гибкость источников данных

Поддерживаются различные источники DICOM данных:
- Локальные папки с файлами
- Orthanc PACS через DICOMweb API

### 3. Поддержка различных типов моделей

- Локальные PyTorch модели
- Удаленные модели через Triton Inference Server

## Структура проекта

```
spine_processor.py      # Основная логика обработки
orthanc_client.py      # Работа с Orthanc
triton_client.py       # Работа с Triton Inference Server
main_refactored.py     # Главный скрипт
example_usage.py       # Примеры использования
```

## Основные функции

### spine_processor.py

```python
def initialize_models(ax_model_path, sag_step_1_model_path, sag_step_2_model_path, grading_model_path):
    """Инициализирует все модели один раз"""
    
def process_study(studies, ax_model, sag_step_1_model, sag_step_2_model, grading_model):
    """Обрабатывает исследование с уже загруженными моделями"""
```

### orthanc_client.py

```python
def read_dicoms_in_folder(study_folder):
    """Читает DICOM файлы из папки"""
    
def download_study_from_orthanc(study_instance_uid, client):
    """Загружает исследование из Orthanc"""
```

## Примеры использования

### 1. Локальные модели + папка с файлами

```python
from spine_processor import initialize_models, process_study
from orthanc_client import read_dicoms_in_folder

# Инициализация моделей (один раз)
models = initialize_models()
ax_model, sag_step_1_model, sag_step_2_model, grading_model = models

# Получение исследования
studies = read_dicoms_in_folder("path/to/dicom/folder")

# Обработка
results = process_study(studies, ax_model, sag_step_1_model, sag_step_2_model, grading_model)
```

### 2. Локальные модели + Orthanc

```python
from spine_processor import initialize_models, process_study
from orthanc_client import create_orthanc_client, download_study_from_orthanc

# Инициализация моделей
models = initialize_models()
ax_model, sag_step_1_model, sag_step_2_model, grading_model = models

# Создание клиента Orthanc
client = create_orthanc_client("http://orthanc:8042", "username", "password")

# Получение исследования
studies = download_study_from_orthanc("study_instance_uid", client)

# Обработка
results = process_study(studies, ax_model, sag_step_1_model, sag_step_2_model, grading_model)
```

### 3. Triton Inference Server

```python
from triton_client import initialize_triton_models
from orthanc_client import read_dicoms_in_folder

# Инициализация Triton клиента
triton_client, model_names = initialize_triton_models("localhost:8000")

# Получение исследования
studies = read_dicoms_in_folder("path/to/dicom/folder")

# Обработка через Triton (требует адаптации process_study)
# results = process_study_triton(studies, triton_client, model_names)
```

## Запуск

### Командная строка

```bash
# Локальные модели + папка
python main_refactored.py --source folder --study_folder ST000000

# Локальные модели + Orthanc
python main_refactored.py --source orthanc --study_uid "1.2.3.4.5" \
    --orthanc_url "http://orthanc:8042" --orthanc_username admin --orthanc_password password

# Triton Inference Server
python main_refactored.py --use_triton --triton_url "localhost:8000" \
    --source folder --study_folder ST000000
```

### Программно

```python
# Смотрите example_usage.py для подробных примеров
python example_usage.py
```

## Преимущества новой архитектуры

### 1. Производительность
- Модели загружаются один раз, а не на каждый запрос
- Возможность кэширования клиентов (Orthanc, Triton)

### 2. Гибкость
- Легко заменить источник данных (папка ↔ Orthanc)
- Легко переключиться между локальными моделями и Triton
- Простое добавление новых источников данных

### 3. Масштабируемость
- Поддержка пакетной обработки
- Готовность к интеграции с FastAPI
- Возможность горизонтального масштабирования через Triton

### 4. Сопровождаемость
- Четкое разделение ответственности
- Модульная архитектура
- Легкое тестирование отдельных компонентов

## Интеграция с FastAPI

Новая архитектура идеально подходит для интеграции с FastAPI:

```python
from fastapi import FastAPI, Depends
from spine_processor import initialize_models, process_study
from orthanc_client import create_orthanc_client

app = FastAPI()

# Инициализация моделей при старте приложения
models = None
orthanc_client = None

@app.on_event("startup")
async def startup_event():
    global models, orthanc_client
    models = initialize_models()
    orthanc_client = create_orthanc_client("http://orthanc:8042", "admin", "password")

@app.post("/analyze_study")
async def analyze_study(study_uid: str):
    studies = download_study_from_orthanc(study_uid, orthanc_client)
    results = process_study(studies, *models)
    return results
```

## Развертывание с Triton

### 1. Подготовка моделей для Triton

```bash
# Конвертация PyTorch моделей в ONNX
python convert_models_to_onnx.py

# Создание конфигурации моделей для Triton
# model_repository/
#   spine_ax_segmentation/
#     config.pbtxt
#     1/model.onnx
#   spine_grading/
#     config.pbtxt  
#     1/model.onnx
```

### 2. Запуск Triton Server

```bash
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v /path/to/model_repository:/models \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models
```

### 3. Использование с Triton

```python
python main_refactored.py --use_triton --triton_url "localhost:8000"
```

## Миграц��я с старого кода

Для миграции с существующего кода:

1. Замените вызовы `main()` на:
   ```python
   models = initialize_models()
   studies = get_study_from_folder(folder_path)  # или get_study_from_orthanc()
   results = process_study(studies, *models)
   ```

2. Адаптируйте код для работы с новой структурой результатов

3. При необходимости добавьте поддержку Triton

## Требования

```
torch
nibabel
pydicom
dicomweb-client
tritonclient[http]  # для Triton
requests
numpy
scipy
matplotlib
pillow
pandas
```

## Тестирование

```bash
# Запуск примеров
python example_usage.py

# Тестирование отдельных компонентов
python -m pytest tests/
```