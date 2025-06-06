import os
import json
import pandas as pd
from pathlib import Path

def prepare_tft_dataset(data_dir: str = ".", output_file: str = "tft_dataset.csv"):
    """
    Подготовка данных к обучению Temporal Fusion Transformer (TFT).
    
    Args:
        data_dir (str): Директория с данными.
        output_file (str): Имя выходного CSV-файла.
    """
    data_dir = Path(data_dir)
    
    # Загружаем датасет
    with open(data_dir / "dataset.json", 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Преобразуем в DataFrame
    df = pd.DataFrame(dataset)
    
    # Добавляем столбец с временем (предполагаем, что время съемки - полдень)
    df['timestamp'] = pd.to_datetime(df['date'] + ' 12:00:00')
    
    # Сортируем по времени
    df = df.sort_values('timestamp')
    
    # Сохраняем в CSV
    df.to_csv(data_dir / output_file, index=False)
    
    print(f"Данные для TFT сохранены в {data_dir / output_file}")

if __name__ == "__main__":
    prepare_tft_dataset() 