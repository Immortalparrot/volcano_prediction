import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

def collect_historical_data(data_dir: str = ".", start_year: int = 2018, end_year: int = 2024):
    """
    Сбор исторических данных (снимки за 2018-2024 годы) и добавление их в датасет.
    
    Args:
        data_dir (str): Директория с данными.
        start_year (int): Начальный год.
        end_year (int): Конечный год.
    """
    data_dir = Path(data_dir)
    
    # Загружаем текущий датасет
    with open(data_dir / "dataset.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Генерируем даты для каждого года
    for year in range(start_year, end_year + 1):
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        current_date = start_date
        
        while current_date <= end_date:
            # Генерируем имя файла (пример)
            file_name = f"Klyuchevskoy_{current_date.strftime('%Y%m%d')}_1200.tiff"
            
            # Проверяем, существует ли файл
            if (data_dir / file_name).exists():
                # Добавляем запись в датасет
                dataset.append({
                    'file_name': file_name,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'period': 'background',  # Предполагаем, что это фоновый период
                    'smoke_detected': False,  # Предполагаем, что дыма нет
                    'smoke_percentage': 0.0,
                    'preview_path': f"{file_name.replace('.tiff', '.png')}",
                    'smoke_index_path': f"{file_name.replace('.tiff', '_smoke_index.png')}"
                })
            
            current_date += timedelta(days=1)
    
    # Сохраняем обновленный датасет
    with open(data_dir / "dataset.json", 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"Исторические данные добавлены в {data_dir / 'dataset.json'}")

if __name__ == "__main__":
    collect_historical_data() 