# Spine Analyzer - Система AI-анализа МРТ позвоночника

## 📋 Обзор проекта

Spine Analyzer - это продвинутая система анализа медицинских изображений, которая автоматически обрабатывает МРТ-исследования позвоночника для обнаружения, классификации и измерения различных патологий. Система интегрируется с PACS-сервером Orthanc и использует модели глубокого обучения, развернутые на NVIDIA Triton Inference Server.

### Ключевые возможности

- **Автоматическая сегментация позвоночника**: Многоэтапная сегментация позвонков, дисков и спинномозгового канала
- **Обнаружение и классификация патологий**:
  - Классификация изменений Modic (тип 0-3)
  - Оценка дегенерации дисков по Pfirrmann (степень 1-5)
  - Обнаружение и измерение грыж дисков
  - Обнаружение и классификация спондилолистеза (по Meyerding)
  - Оценка стеноза позвоночного канала
  - Обнаружение дефектов замыкательных пластинок
- **Количественные измерения**:
  - Объем грыжи и расстояние выпячивания
  - Процент смещения при спондилолистезе
  - Высота и размеры дисков
- **Интеграция с DICOM**:
  - Автоматическая загрузка из Orthanc PACS
  - Генерация структурированных отчетов DICOM (SR)
  - Создание изображений Secondary Capture (SC) с наложениями
  - Загрузка результатов обратно в Orthanc
- **Несколько режимов обработки**:
  - REST API для синхронной обработки
  - Kafka consumer для асинхронной пакетной обработки

## 🏗️ Архитектура системы

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Orthanc   │────▶│   Pipeline   │────▶│  Triton Server  │
│    PACS     │◀────│   Service    │◀────│   (ML Models)   │
└─────────────┘     └──────────────┘     └─────────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
              ┌─────▼───┐   ┌────▼────┐
              │  Kafka  │   │  REST   │
              │Consumer │   │   API   │
              └─────────┘   └─────────┘
```

## 🚀 Быстрый старт

### Требования

- Docker и Docker Compose
- NVIDIA GPU с поддержкой CUDA (для Triton Server)
- Сервер Orthanc PACS
- (Опционально) Кластер Kafka для асинхронной обработки

### 1. Клонирование репозитория

```bash
git clone https://github.com/your-org/spine-analyzer.git
cd spine-analyzer
```

### 2. Настройка окружения

Создайте файл `.env` в корне проекта:

```env
# Конфигурация Orthanc
ORTHANC_URL=http://localhost:8042
ORTHANC_USER=orthanc
ORTHANC_PASSWORD=orthanc

# Triton Inference Server
TRITON_URL=localhost:8001

# Конфигурация Kafka (опционально - для асинхронной обработки)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=spine-analysis-requests
KAFKA_GROUP_ID=spine-analyzer-group
KAFKA_SECURITY_PROTOCOL=PLAINTEXT
# Для SSL/SASL аутентификации (при необходимости):
# KAFKA_SSL_CAFILE=/path/to/ca.pem
# KAFKA_SSL_CERTFILE=/path/to/cert.pem
# KAFKA_SSL_KEYFILE=/path/to/key.pem

# Конфигурация обработки
WORKERS=2
SAG_PATCH_SIZE=[128,96,96]
AX_PATCH_SIZE=[224,224]
DILATE_SIZE=5
CANAL_LABEL=2
```

### 3. Запуск сервисов через Docker Compose

```bash
# Запуск всех сервисов
docker-compose up -d

# Или запуск отдельных сервисов
docker-compose up -d orthanc
docker-compose up -d triton
docker-compose up -d pipeline
```

### 4. Обработка исследования

#### Через REST API:

```bash
# Обработка исследования по его Orthanc ID
curl -X POST http://localhost:8000/process-study/ \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "study_id=YOUR_STUDY_ID"
```

#### Через Kafka:

Отправьте сообщение в настроенный Kafka топик:

```json
{
  "study_id": "YOUR_STUDY_ID"
}
```

### 5. Просмотр результатов

Результаты автоматически загружаются в Orthanc:
- Перейдите на `http://localhost:8042` (веб-интерфейс Orthanc)
- Найдите ваше исследование
- Просмотрите сгенерированные SR (структурированный отчет) и SC (изображения с наложениями) серии

## 📁 Структура проекта

```
spine-analyzer/
├── pipeline/
│   ├── app/
│   │   ├── main.py                 # Точка входа FastAPI приложения
│   │   ├── pipeline.py             # Основной пайплайн обработки
│   │   ├── config.py               # Управление конфигурацией
│   │   ├── kafka_worker.py         # Kafka consumer
│   │   ├── orthanc_client.py       # Клиент Orthanc API
│   │   ├── dicom_io.py             # Обработка DICOM файлов
│   │   ├── dicom_reports.py        # Генерация SR/SC
│   │   ├── grading_processor.py    # Логика оценки патологий
│   │   ├── pathology_measurements.py # Количественные измерения
│   │   ├── predictor.py            # Клиент Triton inference
│   │   ├── preprocessor.py         # Предобработка изображений
│   │   └── ...                     # Другие модули обработки
│   ├── requirements.txt            # Python зависимости
│   └── Dockerfile                  # Контейнер сервиса pipeline
├── triton/
│   ├── models/                     # Репозиторий ML моделей
│   │   ├── seg_sag_stage_1/       # Модель сегментации этап 1
│   │   ├── seg_sag_stage_2/       # Модель сегментации этап 2
│   │   └── grading/                # Модель оценки патологий
│   └── Dockerfile                  # Контейнер Triton server
├── docker-compose.yml              # Оркестрация сервисов
├── .env.example                    # Шаблон переменных окружения
└── README.md                       # Этот файл
```

## 🔧 Конфигурация

### Переменные окружения

| Переменная | Описание | По умолчанию | Обязательно |
|------------|----------|--------------|-------------|
| `ORTHANC_URL` | URL сервера Orthanc | `http://localhost:8042` | Да |
| `ORTHANC_USER` | Имя пользователя Orthanc | `orthanc` | Да |
| `ORTHANC_PASSWORD` | Пароль Orthanc | `orthanc` | Да |
| `TRITON_URL` | URL сервера Triton | `localhost:8001` | Да |
| `KAFKA_BOOTSTRAP_SERVERS` | Брокеры Kafka | - | Нет |
| `KAFKA_TOPIC` | Топик Kafka для запросов | - | Нет |
| `KAFKA_GROUP_ID` | Группа Kafka consumer | - | Нет |
| `WORKERS` | Количество рабочих потоков | `2` | Нет |
| `SAG_PATCH_SIZE` | Размеры сагиттального патча | `[128,96,96]` | Нет |
| `DILATE_SIZE` | Размер ядра дилатации | `5` | Нет |

### Маппинг меток дисков

Система использует стандартизированные метки дисков:

```python
{
    63: 'C2-C3', 64: 'C3-C4', 65: 'C4-C5', 66: 'C5-C6', 67: 'C6-C7',
    71: 'C7-T1', 72: 'T1-T2', 73: 'T2-T3', 74: 'T3-T4', 75: 'T4-T5',
    76: 'T5-T6', 77: 'T6-T7', 78: 'T7-T8', 79: 'T8-T9', 80: 'T9-T10',
    81: 'T10-T11', 82: 'T11-T12', 91: 'T12-L1', 92: 'L1-L2', 93: 'L2-L3',
    94: 'L3-L4', 95: 'L4-L5', 96: 'L5-S1', 100: 'S1-S2'
}
```

## 📊 Формат вывода

### DICOM структурированный отчет (SR)

Система генерирует подробный SR, содержащий:
- Информацию о пациенте и исследовании
- Результаты анализа для каждого диска:
  - Оценки патологий (Modic, Pfirrmann и др.)
  - Измерения патологий (объемы, расстояния)
  - Классификации тяжести

### Изображения Secondary Capture (SC)

Наложения сегментации, показывающие:
- Границы позвонков
- Области дисков
- Обнаруженные патологии (грыжи и др.)
- Цветовые индикаторы тяжести

### Формат JSON ответа

```json
{
  "status": "ok",
  "study_id": "abc123",
  "pipeline_result": {
    "series_detected": {
      "sagittal": 3,
      "axial": 1,
      "coronal": 0
    },
    "disk_results": {
      "95": {
        "disc_label": 95,
        "level_name": "L4-L5",
        "predictions": {
          "Modic": 0,
          "Pfirrmann": 3,
          "Herniation": 1,
          "Spondylolisthesis": 0,
          "Stenosis": 0
        },
        "confidence_scores": {...}
      }
    },
    "pathology_measurements": {
      "95": {
        "herniation": {
          "detected": true,
          "volume_mm3": 245.5,
          "max_protrusion_mm": 4.2,
          "severity": "moderate"
        }
      }
    }
  },
  "report_upload": {
    "sr_uploaded": true,
    "sc_uploaded": true,
    "sc_count": 15
  }
}
```

## 🐳 Docker развертывание

### Сборка образов

```bash
# Сборка всех образов
docker-compose build

# Или сборка по отдельности
docker build -t spine-analyzer-pipeline ./pipeline
docker build -t spine-analyzer-triton ./triton
```

### Развертывание в продакшене

Для продакшена используйте предоставленный `docker-compose.yml`:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Масштабирование

```bash
# Масштабирование pipeline вор��еров
docker-compose up -d --scale pipeline=3
```

## 🧪 Тестирование

### Запуск юнит-тестов

```bash
cd pipeline
python -m pytest tests/
```

### Тест с примерными данными

```bash
# Обработка тестового исследования
python -m pipeline.app.pipeline test_data/sample_study.zip
```

## 📈 Мониторинг

### Логи

```bash
# Просмотр логов pipeline
docker-compose logs -f pipeline

# Просмотр логов Triton
docker-compose logs -f triton
```

### Проверка здоровья

```bash
# Проверка состояния сервиса
curl http://localhost:8000/health
```

## 🛠️ Использование без Docker

### Установка зависимостей

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows

# Установка зависимостей
cd pipeline
pip install -r requirements.txt
```

### Запуск сервиса

```bash
# Запуск FastAPI приложения
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Обработка локального файла

```bash
# Обработка ZIP архива с DICOM файлами
python -m app.pipeline /path/to/study.zip
```

## 🔍 Детали обработки

### Этапы пайплайна

1. **Загрузка данных**: Получение исследования из Orthanc
2. **Предобработка**: 
   - Конвертация DICOM в NIfTI
   - Ресемплинг и нормализация
   - Выравнивание ориентации
3. **Сегментация** (2 этапа):
   - Этап 1: Груб��я сегментация основных структур
   - Этап 2: Точная сегментация с разметкой дисков
4. **Анализ патологий**:
   - Извлечение патчей для каждого диска
   - Классификация патологий через ML модели
5. **Измерения**:
   - Вычисление объемов и расстояний
   - Определение степени тяжести
6. **Генерация отчетов**:
   - Создание DICOM SR с результатами
   - Создание SC изображений с визуализацией
7. **Загрузка результатов**: Отправка обратно в Orthanc

### Поддерживаемые патологии

| Патология | Описание | Градация                              |
|-----------|----------|---------------------------------------|
| Modic изменения | Изменения костного мозга | 0-3 (нет, тип I, II, III)             |
| Дегенерация по Pfirrmann | Дегенерация межпозвонкового диска | 1-5 (от нормы до тяжелой)             |
| Грыжа диска | Выпячивание диска | 0-1 (нет/есть) + измерения            |
| Спондилолистез | Смещение позвонков | 0-1 (нет/есть) + степень по Meyerding |
| Стеноз | Сужение канала | 0-1 (нет/есть)                        |
| Дефекты замыкательных пластинок | Повреждения пластинок | 0-1 (нет/есть)                        |


## 📝 Лицензия

Этот проект лицензирован под лицензией MIT - см. файл [LICENSE](LICENSE) для деталей.

## 📚 Ссылки

- [Классификация Pfirrmann](https://doi.org/10.1097/00007632-200109010-00011)
- [Изменения Modic](https://doi.org/10.1148/radiology.166.1.3336678)
- [Классификация Meyerding](https://doi.org/10.1001/archsurg.1932.01160130121008)
- [Стандарт DICOM](https://www.dicomstandard.org/)

---

**Важно**: Эта система предназначена только для исследовательских целей и не должна использоваться для клинической диагностики без надлежащей валидации и регуляторного одобрения.