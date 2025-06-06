import rasterio
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from PIL import Image

class SatelliteProcessor:
    def __init__(self, config):
        self.config = config
        self.image_size = config['satellite']['image_size']
        self.channels = config['satellite']['channels']
        self.normalize = config['satellite']['normalize']
        
        # Трансформации для изображений
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]) if self.normalize else transforms.Lambda(lambda x: x)
        ])
        
    def load_satellite_image(self, image_path):
        """Загрузка спутникового изображения"""
        with rasterio.open(image_path) as src:
            image = src.read()
            metadata = src.meta
        return image, metadata
    
    def calculate_indices(self, image):
        """Расчет индексов (NDVI, NDBI, NDBaI)"""
        # Нормализация значений в диапазон [0, 1]
        image = image.astype(np.float32) / 10000.0
        
        # Расчет NDVI
        nir = image[3]  # NIR канал
        red = image[2]  # Red канал
        ndvi = (nir - red) / (nir + red + 1e-6)
        
        # Расчет NDBI
        swir = image[4]  # SWIR канал
        ndbi = (swir - nir) / (swir + nir + 1e-6)
        
        # Расчет NDBaI
        ndbai = (swir - red) / (swir + red + 1e-6)
        
        return {
            'ndvi': ndvi,
            'ndbi': ndbi,
            'ndbai': ndbai
        }
    
    def detect_anomalies(self, image, indices):
        """Обнаружение аномалий на изображении"""
        anomalies = {}
        
        # Обнаружение термальных аномалий
        thermal = image[5]  # Термальный канал
        thermal_threshold = self.config.get('thermal_threshold', 300)
        anomalies['thermal'] = thermal > thermal_threshold
        
        # Обнаружение дымовых шлейфов
        smoke_threshold = self.config.get('smoke_threshold', 0.3)
        anomalies['smoke'] = indices['ndbai'] > smoke_threshold
        
        return anomalies
    
    def prepare_for_model(self, image, indices, anomalies):
        """Подготовка данных для модели"""
        # Объединение всех признаков
        features = np.concatenate([
            image,
            np.expand_dims(indices['ndvi'], axis=0),
            np.expand_dims(indices['ndbi'], axis=0),
            np.expand_dims(indices['ndbai'], axis=0),
            np.expand_dims(anomalies['thermal'].astype(np.float32), axis=0),
            np.expand_dims(anomalies['smoke'].astype(np.float32), axis=0)
        ])
        
        # Преобразование в тензор
        features = torch.from_numpy(features).float()
        
        return features
    
    def process_image(self, image_path):
        """Обработка спутникового изображения"""
        try:
            # Чтение изображения с помощью rasterio
            with rasterio.open(image_path) as src:
                # Чтение данных
                image = src.read()
                
                # Преобразование в формат (channels, height, width)
                if image.shape[0] != self.channels:
                    raise ValueError(f"Ожидалось {self.channels} каналов, получено {image.shape[0]}")
                
                # Нормализация значений
                if self.normalize:
                    image = image.astype(np.float32) / 255.0
                
                # Преобразование в тензор
                image_tensor = torch.from_numpy(image).float()
                
                # Изменение размера
                image_tensor = self.transform(image_tensor)
                
                # Расчет индексов
                ndvi = self.calculate_ndvi(image_tensor)
                ndbi = self.calculate_ndbi(image_tensor)
                ndbai = self.calculate_ndbai(image_tensor)
                
                # Объединение признаков
                features = torch.cat([
                    image_tensor,
                    ndvi.unsqueeze(0),
                    ndbi.unsqueeze(0),
                    ndbai.unsqueeze(0)
                ], dim=0)
                
                return {
                    'features': features,
                    'metadata': {
                        'transform': src.transform,
                        'crs': src.crs,
                        'bounds': src.bounds
                    }
                }
                
        except Exception as e:
            raise Exception(f"Ошибка при обработке изображения {image_path}: {str(e)}")
    
    def calculate_ndvi(self, image):
        """Расчет индекса NDVI"""
        nir = image[2]  # Ближний инфракрасный канал
        red = image[0]  # Красный канал
        
        ndvi = (nir - red) / (nir + red + 1e-6)
        return ndvi
    
    def calculate_ndbi(self, image):
        """Расчет индекса NDBI"""
        swir = image[1]  # Коротковолновый инфракрасный канал
        nir = image[2]   # Ближний инфракрасный канал
        
        ndbi = (swir - nir) / (swir + nir + 1e-6)
        return ndbi
    
    def calculate_ndbai(self, image):
        """Расчет индекса NDBAI"""
        swir = image[1]  # Коротковолновый инфракрасный канал
        tir = image[3]   # Тепловой инфракрасный канал
        
        ndbai = (swir - tir) / (swir + tir + 1e-6)
        return ndbai
    
    def detect_thermal_anomaly(self, image):
        """Обнаружение термальных аномалий"""
        tir = image[3]  # Тепловой инфракрасный канал
        threshold = 300  # Порог температуры в Кельвинах
        
        return (tir > threshold).float()
    
    def detect_smoke(self, image):
        """Обнаружение дыма"""
        # Используем комбинацию каналов для обнаружения дыма
        blue = image[0]
        green = image[1]
        red = image[2]
        
        # Расчет индекса дыма
        smoke_index = (blue - red) / (blue + red + 1e-6)
        threshold = 0.3
        
        return (smoke_index > threshold).float() 