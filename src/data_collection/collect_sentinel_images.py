import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions
import math
import time

def collect_sentinel_images(data_dir: str = ".", start_year: int = 2018, end_year: int = 2024):
    """
    Скачивание снимков Sentinel-2 за 2018-2024 годы.
    
    Args:
        data_dir (str): Директория для сохранения снимков.
        start_year (int): Начальный год.
        end_year (int): Конечный год.
    """
    data_dir = Path(data_dir)
    print(f"Снимки будут сохранены в директорию: {data_dir.absolute()}")
    
    # Настройка Sentinel Hub
    config = SHConfig()
    config.sh_client_id = "a1bb9063-ccb5-436d-b1c6-044ea5bdcf5d"
    config.sh_client_secret = "GFwuiBocqBBocBE5U8g9EnR5muoYCvNM"
    
    # Координаты вулкана Ключевской
    bbox = BBox(bbox=[160.5, 56.0, 161.0, 56.5], crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=10)
    MAX_IMAGE_SIZE = 2500
    if size[0] > MAX_IMAGE_SIZE or size[1] > MAX_IMAGE_SIZE:
        scale_factor = max(size[0] / MAX_IMAGE_SIZE, size[1] / MAX_IMAGE_SIZE)
        new_size = (
            math.ceil(size[0] / scale_factor),
            math.ceil(size[1] / scale_factor)
        )
        print(f"Коррекция размера: {size} -> {new_size}")
        size = new_size
    print(f"Размер изображения: {size[0]}x{size[1]} пикселей")
    
    # Генерируем даты для каждого года
    for year in range(start_year, end_year + 1):
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        current_date = start_date
        
        while current_date <= end_date:
            # Создаем запрос на скачивание снимка
            request = SentinelHubRequest(
                evalscript="""
                    //VERSION=3
                    function setup() {
                        return {
                            input: ["B02", "B03", "B04", "B08", "B11"],
                            output: { bands: 5 }
                        };
                    }
                    function evaluatePixel(sample) {
                        return [sample.B02, sample.B03, sample.B04, sample.B08, sample.B11];
                    }
                """,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A,
                        time_interval=(current_date, current_date + timedelta(days=1)),
                        mosaicking_order='leastCC'
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=bbox,
                size=size,
                config=config,
                data_folder=data_dir
            )
            
            # Скачиваем снимок
            try:
                request.get_data()
                print(f"Снимок за {current_date.strftime('%Y-%m-%d')} скачан успешно.")
                # Выводим полный путь сохранения снимка
                tiff_path = data_dir / f"Klyuchevskoy_{current_date.strftime('%Y%m%d')}.tiff"
                print(f"Сохранено в: {tiff_path.absolute()}")
                # Добавляем паузу между запросами
                time.sleep(5)  # Пауза 5 секунд между запросами
            except Exception as e:
                print(f"Ошибка при скачивании снимка за {current_date.strftime('%Y-%m-%d')}: {e}")
                # При ошибке делаем более длительную паузу
                time.sleep(30)  # Пауза 30 секунд при ошибке
            
            current_date += timedelta(days=1)
    
    print(f"Снимки за {start_year}-{end_year} годы скачаны в {data_dir}")

if __name__ == "__main__":
    collect_sentinel_images() 