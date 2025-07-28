"""
Модуль для работы с Triton Inference Server.
Заменяет локальные модели на удаленные вызовы.
"""

import numpy as np
from typing import List, Tuple, Optional
import logging

try:
    import tritonclient.http as httpclient
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    httpclient = None


class TritonModelClient:
    """
    Клиент для работы с моделями на Triton Inference Server.
    """
    
    def __init__(self, triton_url: str = "localhost:8000", verbose: bool = False):
        """
        Инициализирует клиент Triton.
        
        Args:
            triton_url: URL Triton сервера
            verbose: Включить подробное логирование
        """
        self.triton_url = triton_url
        self.verbose = verbose
        self.client = httpclient.InferenceServerClient(url=triton_url, verbose=verbose)
        
    def is_server_ready(self) -> bool:
        """Проверяет готовность сервера."""
        try:
            return self.client.is_server_ready()
        except Exception as e:
            print(f"Ошибка подключения к Triton: {e}")
            return False
    
    def is_model_ready(self, model_name: str) -> bool:
        """Проверяет готовность модели."""
        try:
            return self.client.is_model_ready(model_name)
        except Exception as e:
            print(f"Модель {model_name} не готова: {e}")
            return False
    
    def get_model_metadata(self, model_name: str) -> dict:
        """Получает метаданные модели."""
        try:
            return self.client.get_model_metadata(model_name)
        except Exception as e:
            print(f"Ошибка получения метаданных модели {model_name}: {e}")
            return {}
    
    def infer_segmentation(
        self, 
        model_name: str, 
        input_data: np.ndarray,
        input_name: str = "INPUT",
        output_name: str = "OUTPUT"
    ) -> Optional[np.ndarray]:
        """
        Выполняет инференс сегментации.
        
        Args:
            model_name: Имя модели на Triton
            input_data: Входные данные
            input_name: Имя входного тензора
            output_name: Имя выходного т��нзора
            
        Returns:
            Результат сегментации или None при ошибке
        """
        try:
            # Подготовка входных данных
            inputs = []
            inputs.append(httpclient.InferInput(input_name, input_data.shape, "FP32"))
            inputs[0].set_data_from_numpy(input_data.astype(np.float32))
            
            # Подготовка выходных данных
            outputs = []
            outputs.append(httpclient.InferRequestedOutput(output_name))
            
            # Выполнение инференса
            response = self.client.infer(model_name, inputs, outputs=outputs)
            
            # Получение результата
            result = response.as_numpy(output_name)
            return result
            
        except Exception as e:
            print(f"Ошибка инференса модели {model_name}: {e}")
            return None
    
    def infer_grading(
        self, 
        model_name: str, 
        input_data: np.ndarray,
        input_name: str = "INPUT"
    ) -> Optional[List[np.ndarray]]:
        """
        Выполняет инференс grading (множественные выходы).
        
        Args:
            model_name: Имя модели на Triton
            input_data: Входные данные
            input_name: Имя входного тензора
            
        Returns:
            Список результатов grading или None при ошибке
        """
        try:
            # Подготовка входных данных
            inputs = []
            inputs.append(httpclient.InferInput(input_name, input_data.shape, "FP32"))
            inputs[0].set_data_from_numpy(input_data.astype(np.float32))
            
            # Получаем метаданные модели для определения выходов
            metadata = self.get_model_metadata(model_name)
            output_names = []
            if 'outputs' in metadata:
                for output in metadata['outputs']:
                    output_names.append(output['name'])
            else:
                # Fallback: стандартные имена выходов для grading
                output_names = [
                    "OUTPUT_MODIC", "OUTPUT_UP_ENDPLATE", "OUTPUT_LOW_ENDPLATE",
                    "OUTPUT_SPONDY", "OUTPUT_HERN", "OUTPUT_NARROW", 
                    "OUTPUT_BULGE", "OUTPUT_PFIRRMAN"
                ]
            
            # Подготовка выходных данных
            outputs = []
            for output_name in output_names:
                outputs.append(httpclient.InferRequestedOutput(output_name))
            
            # Выполнение инференса
            response = self.client.infer(model_name, inputs, outputs=outputs)
            
            # Получение результатов
            results = []
            for output_name in output_names:
                try:
                    result = response.as_numpy(output_name)
                    results.append(result)
                except:
                    # Если выход не найден, добавляем None
                    results.append(None)
            
            return results
            
        except Exception as e:
            print(f"Ошибка инференса grading модели {model_name}: {e}")
            return None


def initialize_triton_models(
    triton_url: str = "localhost:8000",
    ax_model_name: str = "spine_ax_segmentation",
    sag_step_1_model_name: str = "spine_sag_step1_segmentation", 
    sag_step_2_model_name: str = "spine_sag_step2_segmentation",
    grading_model_name: str = "spine_grading"
) -> Tuple[TritonModelClient, dict]:
    """
    Инициализирует подключение к Triton и проверяет до��тупность моделей.
    
    Args:
        triton_url: URL Triton сервера
        ax_model_name: Имя модели аксиальной сегментации
        sag_step_1_model_name: Имя модели первого шага сагиттальной сегментации
        sag_step_2_model_name: Имя модели второго шага сагиттальной сегментации
        grading_model_name: Имя модели grading
        
    Returns:
        Кортеж (клиент, словарь с именами моделей)
    """
    if not TRITON_AVAILABLE:
        raise ImportError("tritonclient не установлен. Установите: pip install tritonclient[http]")
    
    client = TritonModelClient(triton_url)
    
    # Проверяем готовность сервера
    if not client.is_server_ready():
        raise ConnectionError(f"Triton сервер {triton_url} недоступен")
    
    model_names = {
        "ax_model": ax_model_name,
        "sag_step_1_model": sag_step_1_model_name,
        "sag_step_2_model": sag_step_2_model_name,
        "grading_model": grading_model_name
    }
    
    # Проверяем готовность всех моделей
    for model_key, model_name in model_names.items():
        if not client.is_model_ready(model_name):
            print(f"Предупреждение: модель {model_name} ({model_key}) не готова")
        else:
            print(f"Модель {model_name} ({model_key}) готова")
    
    return client, model_names


# Адаптированные функции для работы с Triton
def triton_segmentation_inference(
    client: TritonModelClient,
    model_name: str,
    input_data: np.ndarray
) -> Optional[np.ndarray]:
    """
    Выполняет сегментацию через Triton.
    
    Args:
        client: Клиент Triton
        model_name: Имя модели
        input_data: Входные данные
        
    Returns:
        Результат сегментации
    """
    return client.infer_segmentation(model_name, input_data)


def triton_grading_inference(
    client: TritonModelClient,
    model_name: str,
    input_data: np.ndarray
) -> Optional[List[int]]:
    """
    Выполняет grading через Triton.
    
    Args:
        client: Клиент Triton
        model_name: Имя модели
        input_data: Входные данные
        
    Returns:
        Список результатов grading
    """
    results = client.infer_grading(model_name, input_data)
    if results is None:
        return None
    
    # Преобразуем результаты в список целых чисел (argmax)
    predictions = []
    for result in results:
        if result is not None:
            if result.ndim > 1:
                # Если результат многомерный, берем argmax
                prediction = np.argmax(result)
            else:
                # Если одномерный, берем первый элемент
                prediction = int(result[0])
            predictions.append(prediction)
        else:
            predictions.append(0)  # Значение по умолчанию
    
    return predictions


# Пример использования
if __name__ == "__main__":
    try:
        # Инициализация клиента Triton
        client, model_names = initialize_triton_models("localhost:8000")
        print("Triton клиент инициализирован успешно")
        
        # Пример данных для тестирования
        test_input = np.random.rand(1, 1, 64, 64, 64).astype(np.float32)
        
        # Тест сегментации
        seg_result = triton_segmentation_inference(
            client, 
            model_names["ax_model"], 
            test_input
        )
        if seg_result is not None:
            print(f"Сегментация выполнена, форма результата: {seg_result.shape}")
        
        # Тест grading
        grading_result = triton_grading_inference(
            client,
            model_names["grading_model"],
            test_input
        )
        if grading_result is not None:
            print(f"Grading выполнен, результаты: {grading_result}")
            
    except Exception as e:
        print(f"Ошибка при работе с Triton: {e}")