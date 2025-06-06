import unittest
import os
import numpy as np
from pathlib import Path
import rasterio
from rasterio.windows import Window
from PIL import Image
import json

class TestTiffProcessing(unittest.TestCase):
    def setUp(self):
        # Путь к тестовому TIFF-файлу
        self.test_tiff_path = Path("Klyuchevskoy_20250527_1859.tiff")
        self.test_rgb_path = Path("Klyuchevskoy_20250527_1859_RGB8_stretch.tiff")
        
    def test_tiff_loading(self):
        """Тест загрузки TIFF-файла"""
        self.assertTrue(self.test_tiff_path.exists(), "Тестовый TIFF-файл не найден")
        
        with rasterio.open(self.test_tiff_path) as src:
            # Проверяем основные параметры
            self.assertEqual(src.count, 4, "Неверное количество каналов")
            self.assertEqual(src.dtypes[0], 'float32', "Неверный тип данных")
            
            # Проверяем размеры
            self.assertGreater(src.width, 0, "Ширина должна быть положительной")
            self.assertGreater(src.height, 0, "Высота должна быть положительной")
            
    def test_cloud_processing(self):
        """Тест обработки облачности"""
        with rasterio.open(self.test_tiff_path) as src:
            # Читаем маску облаков (4-й канал)
            cloud_mask = src.read(4)
            
            # Проверяем, что маска содержит только 0 и 1
            unique_values = np.unique(cloud_mask)
            self.assertTrue(all(x in [0, 1] for x in unique_values), 
                          "Маска облаков должна содержать только 0 и 1")
            
            # Проверяем, что маска имеет правильную форму
            self.assertEqual(cloud_mask.shape, (src.height, src.width),
                           "Неверная форма маски облаков")
            
    def test_preview_creation(self):
        """Тест создания превью"""
        # Создаем превью, если его нет
        preview_path = self.test_tiff_path.with_suffix('.png')
        if not preview_path.exists():
            with rasterio.open(self.test_tiff_path) as src:
                # Читаем RGB каналы
                rgb = src.read([1, 2, 3])
                # Нормализуем значения
                rgb = np.clip(rgb / 65535.0 * 255, 0, 255).astype(np.uint8)
                # Транспонируем для PIL
                rgb = np.transpose(rgb, (1, 2, 0))
                # Создаем изображение
                img = Image.fromarray(rgb)
                img.save(preview_path)
        
        # Проверяем, что превью существует
        self.assertTrue(preview_path.exists(), "Превью не было создано")
        
        # Проверяем формат превью
        with Image.open(preview_path) as img:
            self.assertEqual(img.format, 'PNG', "Превью должно быть в формате PNG")
            self.assertEqual(img.mode, 'RGB', "Превью должно быть в режиме RGB")
            
    def test_rgb_processing(self):
        """Тест обработки RGB-версии"""
        self.assertTrue(self.test_rgb_path.exists(), "RGB-версия не найдена")
        
        with rasterio.open(self.test_rgb_path) as src:
            # Проверяем, что это RGB-изображение
            self.assertEqual(src.count, 3, "RGB-изображение должно иметь 3 канала")
            self.assertEqual(src.dtypes[0], 'uint8', "RGB-изображение должно быть в формате uint8")
            
            # Проверяем, что значения находятся в правильном диапазоне
            data = src.read()
            self.assertTrue(np.all(data >= 0) and np.all(data <= 255),
                          "Значения должны быть в диапазоне 0-255")

if __name__ == '__main__':
    unittest.main() 