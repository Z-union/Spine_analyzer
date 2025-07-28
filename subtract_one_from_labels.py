import os
import nibabel as nib
import numpy as np

def process_nii_files(input_dir, output_dir=None):
    """
    Обрабатывает все nii файлы в папке, вычитая 1 из всех значений больше 0.
    
    :param input_dir: Папка с исходными nii файлами
    :param output_dir: Папка для сохранения результатов (если None, перезаписывает исходные)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Получаем все nii файлы в папке
    nii_files = [f for f in os.listdir(input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    for filename in nii_files:
        input_path = os.path.join(input_dir, filename)
        
        try:
            # Загружаем файл
            img = nib.load(input_path)
            data = np.asanyarray(img.dataobj)
            
            # Вычитаем 1 из всех значений больше 0
            data_processed = data.copy()
            mask = data_processed > 0
            data_processed[mask] = data_processed[mask] - 1
            
            # Создаем новый Nifti1Image
            processed_img = nib.Nifti1Image(data_processed, img.affine, img.header)
            
            # Определяем путь для сохранения
            if output_dir:
                output_path = os.path.join(output_dir, filename)
            else:
                output_path = input_path
            
            # Сохраняем результат
            nib.save(processed_img, output_path)
            print(f"Обработан: {filename}")
            
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}")

def process_nii_files_recursive(input_dir, output_dir=None):
    """
    Рекурсивно обрабатывает все nii файлы в папке и подпапках.
    
    :param input_dir: Папка с исходными nii файлами
    :param output_dir: Папка для сохранения результатов (если None, перезаписывает исходные)
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Рекурсивно находим все nii файлы
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.nii') or filename.endswith('.nii.gz'):
                input_path = os.path.join(root, filename)
                
                try:
                    # Загружаем файл
                    img = nib.load(input_path)
                    data = np.asanyarray(img.dataobj)
                    
                    # Вычитаем 1 из всех значений больше 0
                    data_processed = data.copy()
                    mask = data_processed > 0
                    data_processed[mask] = data_processed[mask] - 1
                    
                    # Создаем новый Nifti1Image
                    processed_img = nib.Nifti1Image(data_processed, img.affine, img.header)
                    
                    # Определяем путь для сохранения
                    if output_dir:
                        # Сохраняем структуру папок
                        rel_path = os.path.relpath(input_path, input_dir)
                        output_path = os.path.join(output_dir, rel_path)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    else:
                        output_path = input_path
                    
                    # Сохраняем результат
                    nib.save(processed_img, output_path)
                    print(f"Обработан: {rel_path if output_dir else input_path}")
                    
                except Exception as e:
                    print(f"Ошибка при обработке {input_path}: {e}")

if __name__ == "__main__":
    # Пример использования
    input_directory = r"C:\Users\Gleb\PycharmProjects\totalspineseg\nnUNet_raw\Dataset003_SpineAX\labelsTr"  # Папка с исходными файлами
    output_directory = r"C:\Users\Gleb\PycharmProjects\totalspineseg\nnUNet_raw\Dataset003_SpineAX\labelsTr2"  # Папка для результатов (можно None для перезаписи)
    
    # Для обработки только в одной папке (без подпапок)
    process_nii_files(input_directory, output_directory)
    
    # Для рекурсивной обработки (включая подпапки)
    # process_nii_files_recursive(input_directory, output_directory) 