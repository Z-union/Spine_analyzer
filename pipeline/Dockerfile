FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Установим зависимости для OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip config set global.trusted-host "pypi.org pypi.python.org files.pythonhosted.org" && \
    pip install --no-cache-dir -r /app/requirements.txt

# код приложения
COPY . /app/pipeline

EXPOSE 8000

CMD ["uvicorn", "pipeline.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
