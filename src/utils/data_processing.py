import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple
import rasterio
from rasterio.windows import Window

def filter_images_by_cloud_coverage(image_paths: List[str], max_cloud_coverage: float = 0.3) -> List[str]:
    """
    Фильтрует изображения по проценту облачности
    
    Args:
        image_paths: Список путей к изображениям
        max_cloud_coverage: Максимально допустимый процент облачности
        
    Returns:
        Список путей к изображениям с допустимым процентом облачности
    """
    filtered_paths = []
    
    for path in image_paths:
        try:
            with rasterio.open(path) as src:
                # Получаем метаданные облачности из файла
                cloud_coverage = src.meta.get('cloud_coverage', 1.0)
                
                if cloud_coverage <= max_cloud_coverage:
                    filtered_paths.append(path)
        except Exception as e:
            print(f"Ошибка при обработке {path}: {str(e)}")
            continue
    
    return filtered_paths

def interpolate_missing_data(data: pd.DataFrame, 
                           time_column: str,
                           value_column: str,
                           max_gap_days: int = 3) -> pd.DataFrame:
    """
    Интерполирует пропущенные данные во временном ряду
    
    Args:
        data: DataFrame с данными
        time_column: Название столбца с временем
        value_column: Название столбца со значениями
        max_gap_days: Максимальный размер пропуска в днях для интерполяции
        
    Returns:
        DataFrame с интерполированными данными
    """
    # Сортируем по времени
    data = data.sort_values(time_column)
    
    # Создаем полный временной ряд
    date_range = pd.date_range(start=data[time_column].min(),
                             end=data[time_column].max(),
                             freq='D')
    
    # Создаем DataFrame с полным временным рядом
    full_data = pd.DataFrame({time_column: date_range})
    
    # Объединяем с исходными данными
    merged_data = pd.merge(full_data, data, on=time_column, how='left')
    
    # Находим пропуски
    gaps = merged_data[value_column].isna()
    
    # Интерполируем только небольшие пропуски
    for i in range(len(merged_data)):
        if gaps.iloc[i]:
            # Ищем ближайшие непустые значения
            left_idx = i - 1
            right_idx = i + 1
            
            while left_idx >= 0 and gaps.iloc[left_idx]:
                left_idx -= 1
            while right_idx < len(merged_data) and gaps.iloc[right_idx]:
                right_idx += 1
            
            # Проверяем размер пропуска
            if left_idx >= 0 and right_idx < len(merged_data):
                gap_size = (merged_data[time_column].iloc[right_idx] - 
                          merged_data[time_column].iloc[left_idx]).days
                
                if gap_size <= max_gap_days:
                    # Линейная интерполяция
                    left_val = merged_data[value_column].iloc[left_idx]
                    right_val = merged_data[value_column].iloc[right_idx]
                    left_time = merged_data[time_column].iloc[left_idx]
                    right_time = merged_data[time_column].iloc[right_idx]
                    
                    for j in range(left_idx + 1, right_idx):
                        t = (merged_data[time_column].iloc[j] - left_time) / (right_time - left_time)
                        merged_data.loc[merged_data.index[j], value_column] = left_val + t * (right_val - left_val)
    
    return merged_data

def extract_image_patches(image_path: str, 
                         patch_size: Tuple[int, int] = (224, 224),
                         stride: Tuple[int, int] = (112, 112)) -> List[np.ndarray]:
    """
    Извлекает патчи из изображения для обучения модели
    
    Args:
        image_path: Путь к изображению
        patch_size: Размер патча (высота, ширина)
        stride: Шаг для извлечения патчей
        
    Returns:
        Список патчей
    """
    patches = []
    
    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        
        for y in range(0, height - patch_size[0] + 1, stride[0]):
            for x in range(0, width - patch_size[1] + 1, stride[1]):
                window = Window(x, y, patch_size[1], patch_size[0])
                patch = src.read(window=window)
                patches.append(patch)
    
    return patches 