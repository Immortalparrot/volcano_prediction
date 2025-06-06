from sentinel_api import SentinelDataCollector
from datetime import datetime, timedelta
from pathlib import Path
import time

# Параметры сбора
DAYS = 14
MAX_CLOUD_COVER = 20
RESOLUTION = 10
CACHE_DIR = "data/period_collection/optical"
CONFIG_PATH = "config/sentinel_credentials.json"

# Создаем директорию для сохранения
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

collector = SentinelDataCollector(
    config_path=CONFIG_PATH,
    cache_dir=CACHE_DIR
)

end_date = datetime.now()
start_date = end_date - timedelta(days=DAYS)

print(f"Сбор данных за период: {start_date.date()} - {end_date.date()} (bbox: {collector._create_bbox(collector.volcano_coords['lon'], collector.volcano_coords['lat'], collector.default_params['bbox_size'])})")

for i in range(DAYS):
    day_start = (start_date + timedelta(days=i)).strftime("%Y-%m-%d")
    day_end = (start_date + timedelta(days=i+1)).strftime("%Y-%m-%d")
    print(f"\nПробуем собрать за {day_start}...")
    try:
        result = collector.get_sentinel2_data(
            start_date=day_start,
            end_date=day_end,
            resolution=RESOLUTION,
            max_cloud_cover=MAX_CLOUD_COVER
        )
        print(f"Успешно: {result['filename']}")
        print(f"  Метаданные: {result['metadata_file']}")
    except Exception as e:
        print(f"Нет подходящих данных за {day_start}: {e}")
    time.sleep(1)  # чтобы не перегружать API

print("\nСбор завершён!") 