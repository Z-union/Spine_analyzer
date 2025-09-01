# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies from requirements.txt with layer caching
COPY pipeline/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code so that module path is /app/pipeline/...
COPY pipeline /app/pipeline

EXPOSE 8000

CMD ["uvicorn", "pipeline.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
