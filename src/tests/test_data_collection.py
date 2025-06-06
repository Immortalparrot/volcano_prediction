import unittest
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import shutil
import tempfile

from src.data_collection.image_analyzer import ImageAnalyzer

class TestDataCollection(unittest.TestCase):
    def setUp(self):
        """Подготовка тестового окружения"""
        # Создаем временную директорию для тестов
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Копируем тестовые файлы во временную директорию
        test_files = [
            "Klyuchevskoy_20250525_1544.tiff",
            "Klyuchevskoy_20250527_1854.tiff",
            "Klyuchevskoy_20250527_1859.tiff"
        ]
        
        for file in test_files:
            if Path(file).exists():
                shutil.copy2(file, self.test_dir / file)
        
        self.analyzer = ImageAnalyzer(data_dir=str(self.test_dir))
    
    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.test_dir)
    
    def test_tiff_loading(self):
        """Тест загрузки TIFF-файлов"""
        tiff_files = list(self.test_dir.glob("*.tiff"))
        self.assertGreater(len(tiff_files), 0, "Тестовые TIFF-файлы не найдены")
        
        for tiff_file in tiff_files:
            result = self.analyzer.load_tiff(tiff_file)
            
            # Проверяем структуру результата
            self.assertIn('metadata', result)
            self.assertIn('channel_info', result)
            self.assertIn('data', result)
            
            # Проверяем данные
            self.assertIsInstance(result['data'], np.ndarray)
            self.assertEqual(result['data'].shape[0], 4, "Должно быть 4 канала")
            
            # Проверяем информацию о каналах
            for i in range(4):
                channel_key = f'channel_{i+1}'
                self.assertIn(channel_key, result['channel_info'])
                channel_data = result['channel_info'][channel_key]
                self.assertIn('min', channel_data)
                self.assertIn('max', channel_data)
                self.assertIn('mean', channel_data)
                self.assertIn('std', channel_data)
    
    def test_smoke_detection(self):
        """Тест обнаружения дымов"""
        tiff_file = next(self.test_dir.glob("*.tiff"))
        image_data = self.analyzer.load_tiff(tiff_file)
        smoke_info = self.analyzer.check_for_smoke(image_data['data'])
        
        # Проверяем структуру результата
        self.assertIn('has_smoke', smoke_info)
        self.assertIn('smoke_percentage', smoke_info)
        self.assertIn('smoke_mask', smoke_info)
        self.assertIn('smoke_index', smoke_info)
        
        # Проверяем типы данных
        self.assertIsInstance(smoke_info['has_smoke'], (bool, np.bool_))
        self.assertIsInstance(smoke_info['smoke_percentage'], float)
        self.assertIsInstance(smoke_info['smoke_mask'], np.ndarray)
        self.assertIsInstance(smoke_info['smoke_index'], np.ndarray)
        
        # Проверяем диапазоны значений
        self.assertGreaterEqual(smoke_info['smoke_percentage'], 0)
        self.assertLessEqual(smoke_info['smoke_percentage'], 100)
    
    def test_preview_creation(self):
        """Тест создания превью"""
        tiff_file = next(self.test_dir.glob("*.tiff"))
        image_data = self.analyzer.load_tiff(tiff_file)
        smoke_info = self.analyzer.check_for_smoke(image_data['data'])
        
        # Создаем превью
        preview_path = self.test_dir / f"{tiff_file.stem}.png"
        self.analyzer.create_preview(image_data['data'], preview_path, smoke_info)
        
        # Проверяем, что файлы созданы
        self.assertTrue(preview_path.exists(), "Превью не создано")
        smoke_index_path = self.test_dir / f"{tiff_file.stem}_smoke_index.png"
        self.assertTrue(smoke_index_path.exists(), "Карта индекса дымов не создана")
    
    def test_recent_images_analysis(self):
        """Тест анализа последних снимков"""
        results = self.analyzer.analyze_recent_images(days=7)
        
        # Проверяем, что есть результаты
        self.assertGreater(len(results), 0, "Нет результатов анализа")
        
        for result in results:
            # Проверяем структуру результата
            self.assertIn('file_name', result)
            self.assertIn('date', result)
            self.assertIn('smoke_detected', result)
            self.assertIn('smoke_percentage', result)
            self.assertIn('preview_path', result)
            self.assertIn('smoke_index_path', result)
            
            # Проверяем типы данных
            self.assertIsInstance(result['date'], datetime)
            self.assertIsInstance(result['smoke_detected'], (bool, np.bool_))
            self.assertIsInstance(result['smoke_percentage'], float)
            
            # Проверяем существование файлов
            self.assertTrue(Path(result['preview_path']).exists(), 
                          f"Превью не найдено: {result['preview_path']}")
            if result['smoke_detected']:
                self.assertTrue(Path(result['smoke_index_path']).exists(),
                              f"Карта индекса дымов не найдена: {result['smoke_index_path']}")

if __name__ == '__main__':
    unittest.main() 