from sentinel_api import SentinelDataCollector
from datetime import datetime, timedelta
import os
from pathlib import Path

def test_sentinel_collection():
    # Создаем директорию для тестовых данных
    test_dir = Path("./data/test/optical")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем коллектор с тестовыми параметрами
    collector = SentinelDataCollector(
        config_path="config/sentinel_credentials.json",
        cache_dir=str(test_dir)
    )
    
    # Тестовый период: последние 7 дней
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"Тестирование сбора данных за период: {start_date.date()} - {end_date.date()}")
    
    try:
        # Получаем данные с разными параметрами облачности
        print("\n1. Тест с максимальной облачностью 20%")
        result_20 = collector.get_sentinel2_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            resolution=10,
            max_cloud_cover=20
        )
        print(f"Успешно сохранено: {result_20['filename']}")
        
        # Если не удалось получить данные с 20% облачностью, пробуем с 30%
        if not os.path.exists(result_20['filename']):
            print("\n2. Тест с максимальной облачностью 30%")
            result_30 = collector.get_sentinel2_data(
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                resolution=10,
                max_cloud_cover=30
            )
            print(f"Успешно сохранено: {result_30['filename']}")
        
        print("\nПроверка метаданных:")
        if os.path.exists(result_20['metadata_file']):
            with open(result_20['metadata_file'], 'r') as f:
                import json
                metadata = json.load(f)
                print(f"Размер изображения: {metadata['size']}")
                print(f"Разрешение: {metadata['resolution']}м")
                print(f"Облачность: {metadata['cloud_cover']}%")
                print(f"Каналы: {', '.join(metadata['channels'])}")
        
    except Exception as e:
        print(f"Ошибка при тестировании: {str(e)}")
        raise

if __name__ == "__main__":
    test_sentinel_collection() 