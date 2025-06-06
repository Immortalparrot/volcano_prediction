import os
import json
import pandas as pd
from pathlib import Path

def prepare_dataset(data_dir: str = ".", output_file: str = "dataset.json"):
    """
    Подготовка датасета для обучения нейросети.
    
    Args:
        data_dir (str): Директория с данными.
        output_file (str): Имя выходного JSON-файла.
    """
    data_dir = Path(data_dir)
    dataset = []
    
    # Загружаем CSV с периодами извержений
    eruptions_df = pd.read_csv(data_dir / "klyuchevskoy_eruptions.csv")
    
    # Обрабатываем каждый TIFF-файл
    for tiff_file in data_dir.glob("*.tiff"):
        if "RGB" in tiff_file.name:
            continue
        
        # Извлекаем дату из имени файла
        try:
            file_date = pd.to_datetime(tiff_file.stem.split('_')[1], format='%Y%m%d')
            
            # Определяем период (pre-eruption, eruption, post-eruption, background)
            period = "background"
            for _, row in eruptions_df.iterrows():
                eruption_start = pd.to_datetime(row['start_date'])
                eruption_end = pd.to_datetime(row['end_date'])
                if file_date < eruption_start:
                    period = "pre-eruption"
                elif file_date <= eruption_end:
                    period = "eruption"
                elif file_date <= eruption_end + pd.Timedelta(days=14):
                    period = "post-eruption"
            
            # Проверяем наличие дыма
            smoke_detected = False
            smoke_percentage = 0.0
            smoke_index_path = tiff_file.with_name(f"{tiff_file.stem}_smoke_index.png")
            if smoke_index_path.exists():
                smoke_detected = True
                smoke_percentage = 3.33  # Примерное значение, можно уточнить
            
            # Добавляем запись в датасет
            dataset.append({
                'file_name': tiff_file.name,
                'date': file_date.strftime('%Y-%m-%d'),
                'period': period,
                'smoke_detected': smoke_detected,
                'smoke_percentage': smoke_percentage,
                'preview_path': str(tiff_file.with_suffix('.png')),
                'smoke_index_path': str(smoke_index_path)
            })
        except (ValueError, IndexError):
            print(f"Не удалось обработать файл: {tiff_file.name}")
            continue
    
    # Сохраняем датасет в JSON
    with open(data_dir / output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    print(f"Датасет сохранен в {data_dir / output_file}")

if __name__ == "__main__":
    prepare_dataset() 