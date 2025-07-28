# Быстрый старт - Рефакторенный анализатор позвоночника

## Основные способы запуска

### 1. Локальные модели + папка с DICOM файлами

```bash
# Простейший запуск
python main_refactored.py --source folder --study_folder ST000000

# С указанием выходной папки
python main_refactored.py --source folder --study_folder ST000000 --output ./results_refactored
```

### 2. Локальные модели + Orthanc

```bash
python main_refactored.py \
    --source orthanc \
    --study_uid "1.2.840.113619.6.44.320724410489713965087995388011340888881" \
    --orthanc_url "http://158.160.109.200:8042" \
    --orthanc_username admin \
    --orthanc_password mypassword
```

### 3. Triton Inference Server + папка

```bash
python main_refactored.py \
    --use_triton \
    --triton_url "localhost:8000" \
    --source folder \
    --study_folder ST000000
```

### 4. Triton Inference Server + Orthanc

```bash
python main_refactored.py \
    --use_triton \
    --triton_url "localhost:8000" \
    --source orthanc \
    --study_uid "1.2.840.113619.6.44.320724410489713965087995388011340888881" \
    --orthanc_url "http://158.160.109.200:8042" \
    --orthanc_username admin \
    --orthanc_password mypassword
```

## Программное использование

### Локальные модели

```python
from spine_processor import initialize_models, process_study
from orthanc_client import read_dicoms_in_folder

# Инициализация моделей (один раз)
models = initialize_models()
ax_model, sag_step_1_model, sag_step_2_model, grading_model = models

# Получение данных
studies = read_dicoms_in_folder("ST000000")

# Обработка
results = process_study(studies, ax_model, sag_step_1_model, sag_step_2_model, grading_model)
```

### Triton Inference Server

```python
from triton_client import initialize_triton_models
from spine_processor_triton import process_study_triton
from orthanc_client import read_dicoms_in_folder

# Инициализация Triton (один раз)
triton_client, model_names = initialize_triton_models("localhost:8000")

# Получение данных
studies = read_dicoms_in_folder("ST000000")

# Обработка
results = process_study_triton(studies, triton_client, model_names)
```

### Orthanc

```python
from orthanc_client import create_orthanc_client, download_study_from_orthanc
from spine_processor import initialize_models, process_study

# Инициализация клиента Orthanc (один раз)
client = create_orthanc_client("http://orthanc:8042", "admin", "password")

# Инициализация моделей (один раз)
models = initialize_models()

# Получение данных
studies = download_study_from_orthanc("study_instance_uid", client)

# Обработка
results = process_study(studies, *models)
```

## Тестирование

```bash
# Тест всех компонентов
python test_refactored.py

# Примеры использования
python example_usage.py

# Демонстрация Triton vs локальные модели
python example_triton_usage.py
```

## Ключевые отличия от старого кода

| Старый подход | Новый подход |
|---------------|--------------|
| `python run.py input_folder output_folder` | `python main_refactored.py --source folder --study_folder input_folder --output output_folder` |
| Модели загружаются каждый раз | Модели загружаются один раз |
| Только локальные файлы | Поддержка папо�� и Orthanc |
| Только локальные модели | Поддержка локальных моделей и Triton |
| Монолитная архитектура | Модульная архитектура |

## Преимущества новой архитектуры

1. **Производительность**: Модели загружаются один раз
2. **Гибкость**: Легко менять источники данных и типы моделей
3. **Масштабируемость**: Готовность к FastAPI и Triton
4. **Сопровождаемость**: Четкое разделение ответственности

## Миграция с старого кода

Если у вас есть скрипт, который использует старый `run.py`:

```python
# Было:
from run import main
main()

# Стало:
from spine_processor import initialize_models, process_study
from orthanc_client import read_dicoms_in_folder

models = initialize_models()  # Один раз
studies = read_dicoms_in_folder("folder_path")
results = process_study(studies, *models)
```

## Требования

```bash
# Основные зависимости
pip install torch nibabel pydicom dicomweb-client numpy scipy matplotlib pillow pandas

# Для Triton (опционально)
pip install tritonclient[http]

# Или установить все сразу
pip install -r requirements_refactored.txt
```

## Устранение неполадок

### Ошибка "Can't get attribute 'GradingModel'"
Убедитесь, что классы моделей импортированы:
```python
from model.garding import GradingModel, BasicBlock, Bottleneck
```

### Ошибка "tritonclient not found"
Установите Triton клиент:
```bash
pip install tritonclient[http]
```

### Ошибка "DICOM files not found"
Проверьте путь к папке и наличие DICOM файлов:
```bash
ls -la ST000000/
```

### Ошибка подключения к Orthanc
Проверьте URL и учетные данные:
```bash
curl -u admin:password http://158.160.109.200:8042/system
```