from sentinelhub import (
    SHConfig,
    SentinelHubRequest,
    DataCollection,
    MimeType,
    BBox,
    CRS,
    bbox_to_dimensions
)
from datetime import datetime
import rasterio
from rasterio.transform import from_origin
import numpy as np
import math
from affine import Affine
import os
import json
from pathlib import Path
import logging

class SentinelDataCollector:
    def __init__(self, config_path=None, cache_dir="./data/raw/optical"):
        """
        Инициализация коллектора данных Sentinel
        
        Args:
            config_path: путь к файлу конфигурации с учетными данными
            cache_dir: директория для кэширования данных
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Загрузка конфигурации
        self.config = self._load_config(config_path)
        
        # Координаты вулкана Ключевской
        self.volcano_coords = {
            'lat': 56.06,
            'lon': 160.64
        }
        
        # Параметры по умолчанию
        self.default_params = {
            'resolution': 10,  # метров на пиксель
            'max_cloud_cover': 20,  # процент
            'max_image_size': 2500,  # пикселей
            'bbox_size': 0.45  # градусов (примерно 50 км)
        }
        
    def _load_config(self, config_path):
        """Загрузка конфигурации из файла или переменных окружения"""
        config = SHConfig()
        
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                creds = json.load(f)
                config.sh_client_id = creds['client_id']
                config.sh_client_secret = creds['client_secret']
        else:
            config.sh_client_id = os.getenv('SENTINEL_CLIENT_ID')
            config.sh_client_secret = os.getenv('SENTINEL_CLIENT_SECRET')
        
        if not config.sh_client_id or not config.sh_client_secret:
            raise ValueError("Необходимо указать учетные данные Sentinel Hub")
            
        return config
    
    def _create_bbox(self, center_lon, center_lat, size_degrees):
        """Создание bounding box вокруг точки"""
        half_size = size_degrees / 2
        return [
            center_lon - half_size,
            center_lat - half_size,
            center_lon + half_size,
            center_lat + half_size
        ]
    
    def _calculate_image_size(self, bbox, resolution):
        """Расчет размера изображения с проверкой ограничений"""
        bbox_obj = BBox(bbox=bbox, crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox_obj, resolution=resolution)
        
        if size[0] > self.default_params['max_image_size'] or \
           size[1] > self.default_params['max_image_size']:
            scale_factor = max(
                size[0] / self.default_params['max_image_size'],
                size[1] / self.default_params['max_image_size']
            )
            new_size = (
                math.ceil(size[0] / scale_factor),
                math.ceil(size[1] / scale_factor)
            )
            self.logger.info(f"Коррекция размера: {size} -> {new_size}")
            return new_size
            
        return size
    
    def _get_evalscript(self):
        """Генерация EVALSCRIPT для обработки данных"""
        return """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04", "B08", "B11", "B12"],
                output: { bands: 6, sampleType: "FLOAT32" }
            };
        }

        function evaluatePixel(sample) {
            function norm(val) { return val / 10000.0; }
            let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
            let ndbi = (sample.B11 - sample.B08) / (sample.B11 + sample.B08);
            let ndbai = (sample.B12 - sample.B08) / (sample.B12 + sample.B08);
            return [
                norm(sample.B04), // Red
                norm(sample.B03), // Green
                norm(sample.B02), // Blue
                ndvi,
                ndbi,
                ndbai
            ];
        }
        """
    
    def get_sentinel2_data(self, start_date, end_date, resolution=None, max_cloud_cover=None):
        """
        Получение и сохранение данных Sentinel-2
        
        Args:
            start_date: начальная дата
            end_date: конечная дата
            resolution: разрешение в метрах
            max_cloud_cover: максимальный процент облачности
            
        Returns:
            dict: информация о сохраненных данных
        """
        # Использование параметров по умолчанию, если не указаны
        resolution = resolution or self.default_params['resolution']
        max_cloud_cover = max_cloud_cover or self.default_params['max_cloud_cover']
        
        # Создание bounding box
        bbox = self._create_bbox(
            self.volcano_coords['lon'],
            self.volcano_coords['lat'],
            self.default_params['bbox_size']
        )
        
        # Расчет размера изображения
        size = self._calculate_image_size(bbox, resolution)
        
        # Создание запроса
        request = SentinelHubRequest(
            evalscript=self._get_evalscript(),
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(start_date, end_date),
                    maxcc=max_cloud_cover / 100,
                    mosaicking_order='leastCC'
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=BBox(bbox=bbox, crs=CRS.WGS84),
            size=size,
            config=self.config
        )
        
        # Загрузка данных
        self.logger.info("Загрузка данных Sentinel-2...")
        try:
            data = request.get_data()
            self.logger.info("Данные успешно получены")
        except Exception as e:
            self.logger.error(f"Ошибка при загрузке: {str(e)}")
            raise
        
        # Подготовка к сохранению
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        filename = self.cache_dir / f"Klyuchevskoy_{timestamp}.tiff"
        metadata_file = self.cache_dir / f"Klyuchevskoy_{timestamp}_metadata.json"
        
        try:
            # Получаем параметры геопреобразования
            transform = Affine.from_gdal(*BBox(bbox=bbox, crs=CRS.WGS84).get_transform_vector(size[0], size[1]))
            
            # Сохранение изображения
            with rasterio.open(
                filename,
                "w",
                driver="GTiff",
                width=size[0],
                height=size[1],
                count=6,  # 6 каналов: RGB, NDVI, NDBI, NDBaI
                dtype=np.float32,
                crs=CRS.WGS84.pyproj_crs(),
                transform=transform,
                nodata=-9999
            ) as dst:
                # Проверка размерности
                if data[0].shape != (size[1], size[0], 6):
                    raise ValueError("Несоответствие размеров данных")
                
                # Запись каналов
                for i in range(6):
                    dst.write(data[0][:, :, i], i + 1)
                
                # Метаданные
                metadata = {
                    'bbox': bbox,
                    'size': size,
                    'resolution': resolution,
                    'date_acquired': datetime.now().isoformat(),
                    'cloud_cover': max_cloud_cover,
                    'channels': [
                        'Red (B04)',
                        'Green (B03)',
                        'Blue (B02)',
                        'NDVI',
                        'NDBI',
                        'NDBaI'
                    ],
                    'volcano_coords': self.volcano_coords,
                    'processing_info': {
                        'normalized': True,
                        'units': 'reflectance',
                        'nodata_value': -9999
                    }
                }
                
                # Сохранение метаданных
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Добавление метаданных в GeoTIFF
                dst.update_tags(**metadata)
            
            self.logger.info(f"Данные сохранены: {filename}")
            return {
                'filename': str(filename),
                'metadata_file': str(metadata_file),
                'metadata': metadata
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении: {str(e)}")
            raise 