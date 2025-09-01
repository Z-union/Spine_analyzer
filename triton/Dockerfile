# Базовый образ Triton (GPU + Python)
FROM nvcr.io/nvidia/tritonserver:24.05-py3

# Копируем весь репозиторий моделей внутрь контейнера
COPY model_repo /models

# Открываем стандартные порты Triton
EXPOSE 8000 8001 8002

# Запускаем Triton с указанием директории моделей
ENTRYPOINT ["/opt/tritonserver/bin/tritonserver", "--model-repository=/models"]