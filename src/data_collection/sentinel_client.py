import os
from datetime import datetime
from pathlib import Path
import requests
from sentinelsat import SentinelAPI
from typing import List, Dict, Tuple
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

class SentinelHubClient:
    def __init__(self, username: str = None, password: str = None):
        """Инициализация клиента Sentinel Hub"""
        self.username = username or os.getenv('SENTINEL_USERNAME')
        self.password = password or os.getenv('SENTINEL_PASSWORD')
        
        if not self.username or not self.password:
            raise ValueError("Необходимо указать учетные данные Sentinel Hub")
        
        self.api = SentinelAPI(self.username, self.password, 'https://scihub.copernicus.eu/dhus')
    
    def search_images(self, 
                     start_date: datetime,
                     end_date: datetime,
                     coordinates: Tuple[float, float, float, float],
                     cloud_cover: float = 20.0) -> List[Dict]:
        """Поиск снимков за указанный период"""
        # Поиск снимков Sentinel-2
        products = self.api.query(
            coordinates,
            date=(start_date, end_date),
            platformname='Sentinel-2',
            cloudcoverpercentage=(0, cloud_cover)
        )
        
        return self.api.to_dataframe(products)
    
    def download_image(self, 
                      product_id: str,
                      output_dir: Path,
                      bands: List[str] = ['B02', 'B03', 'B04', 'B08']) -> Dict[str, Path]:
        """Загрузка снимка и его обработка"""
        # Создаем директорию для временных файлов
        temp_dir = output_dir / 'temp'
        temp_dir.mkdir(exist_ok=True)
        
        # Загружаем продукт
        self.api.download(product_id, temp_dir)
        
        # Получаем пути к файлам
        downloaded_files = list(temp_dir.glob(f'*{product_id}*.SAFE'))
        if not downloaded_files:
            raise FileNotFoundError(f"Не удалось найти загруженные файлы для {product_id}")
        
        # Обрабатываем каждый канал
        processed_files = {}
        for band in bands:
            band_file = next(downloaded_files[0].glob(f'*_{band}_*.jp2'), None)
            if band_file:
                # Конвертируем в GeoTIFF
                output_file = output_dir / f"{product_id}_{band}.tiff"
                self._convert_to_tiff(band_file, output_file)
                processed_files[band] = output_file
        
        # Удаляем временные файлы
        for file in downloaded_files:
            self._remove_directory(file)
        
        return processed_files
    
    def _convert_to_tiff(self, input_file: Path, output_file: Path):
        """Конвертация JP2 в GeoTIFF"""
        with rasterio.open(input_file) as src:
            # Читаем данные
            data = src.read()
            
            # Сохраняем как GeoTIFF
            with rasterio.open(
                output_file,
                'w',
                driver='GTiff',
                height=data.shape[1],
                width=data.shape[2],
                count=data.shape[0],
                dtype=data.dtype,
                crs=src.crs,
                transform=src.transform
            ) as dst:
                dst.write(data)
    
    def _remove_directory(self, directory: Path):
        """Рекурсивное удаление директории"""
        for item in directory.iterdir():
            if item.is_dir():
                self._remove_directory(item)
            else:
                item.unlink()
        directory.rmdir() 