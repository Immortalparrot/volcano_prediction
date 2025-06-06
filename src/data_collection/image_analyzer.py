import rasterio
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

class ImageAnalyzer:
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        
    def load_tiff(self, file_path: Path) -> dict:
        """Загрузка и анализ TIFF-файла"""
        with rasterio.open(file_path) as src:
            # Читаем все каналы
            data = src.read()
            metadata = src.meta
            
            # Анализ каждого канала
            channel_info = {}
            for i in range(src.count):
                channel_data = data[i]
                channel_info[f'channel_{i+1}'] = {
                    'min': float(channel_data.min()),
                    'max': float(channel_data.max()),
                    'mean': float(channel_data.mean()),
                    'std': float(channel_data.std()),
                    'shape': channel_data.shape
                }
            
            return {
                'metadata': metadata,
                'channel_info': channel_info,
                'data': data
            }
    
    def check_for_smoke(self, data: np.ndarray) -> dict:
        """Проверка наличия дымов на снимке"""
        # Используем комбинацию каналов для обнаружения дымов
        # SWIR (канал 3) и NIR (канал 2) обычно хорошо показывают дымы
        swir = data[2]
        nir = data[1]
        
        # Нормализация каналов
        swir_range = np.max(swir) - np.min(swir)
        swir_norm = (swir - np.min(swir)) / swir_range if swir_range != 0 else np.zeros_like(swir)
        nir_range = np.max(nir) - np.min(nir)
        nir_norm = (nir - np.min(nir)) / nir_range if nir_range != 0 else np.zeros_like(nir)
        
        # Вычисляем индекс дымов (можно настроить коэффициенты)
        smoke_index = 2 * swir_norm - nir_norm
        
        # Применяем пороговое значение
        smoke_threshold = np.mean(smoke_index) + 1.5 * np.std(smoke_index)
        smoke_mask = smoke_index > smoke_threshold
        
        # Применяем морфологические операции для удаления шума
        from scipy import ndimage
        smoke_mask = ndimage.binary_opening(smoke_mask, structure=np.ones((3,3)))
        smoke_mask = ndimage.binary_closing(smoke_mask, structure=np.ones((3,3)))
        
        smoke_percentage = (np.sum(smoke_mask) / smoke_mask.size) * 100
        
        return {
            'has_smoke': smoke_percentage > 2,  # Уменьшаем порог до 2%
            'smoke_percentage': smoke_percentage,
            'smoke_mask': smoke_mask,
            'smoke_index': smoke_index
        }
    
    def create_preview(self, data: np.ndarray, output_path: Path, smoke_info: dict = None):
        """Создание превью снимка с возможностью наложения маски дымов"""
        # Нормализуем RGB каналы
        rgb = data[:3]  # Берем первые 3 канала
        rgb = np.clip(rgb / 65535.0 * 255, 0, 255).astype(np.uint8)
        rgb = np.transpose(rgb, (1, 2, 0))
        
        if smoke_info and smoke_info['has_smoke']:
            # Создаем маску дымов в красном цвете
            smoke_overlay = np.zeros_like(rgb)
            smoke_overlay[smoke_info['smoke_mask']] = [255, 0, 0]
            
            # Накладываем маску с прозрачностью
            alpha = 0.3
            rgb = np.where(smoke_info['smoke_mask'][..., None], 
                          (1 - alpha) * rgb + alpha * smoke_overlay,
                          rgb)
        
        # Приводим к uint8 после всех преобразований
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        # Создаем и сохраняем изображение
        img = Image.fromarray(rgb)
        img.save(output_path)
        
        # Создаем дополнительное изображение с индексом дымов
        if smoke_info:
            plt.figure(figsize=(10, 10))
            plt.imshow(smoke_info['smoke_index'], cmap='hot')
            plt.colorbar(label='Индекс дымов')
            plt.title('Карта индекса дымов')
            plt.savefig(output_path.with_name(f"{output_path.stem}_smoke_index.png"))
            plt.close()
    
    def analyze_recent_images(self, days: int = 7):
        """Анализ снимков за последние N дней"""
        current_date = datetime.now()
        start_date = current_date - timedelta(days=days)
        
        results = []
        for tiff_file in self.data_dir.glob("*.tiff"):
            # Пропускаем RGB-версии
            if "RGB" in tiff_file.name:
                continue
                
            # Извлекаем дату из имени файла
            try:
                file_date = datetime.strptime(tiff_file.stem.split('_')[1], '%Y%m%d')
                if file_date >= start_date:
                    print(f"Анализ файла: {tiff_file.name}")
                    
                    # Загружаем и анализируем снимок
                    image_data = self.load_tiff(tiff_file)
                    smoke_info = self.check_for_smoke(image_data['data'])
                    
                    # Создаем превью
                    preview_path = tiff_file.with_suffix('.png')
                    self.create_preview(image_data['data'], preview_path, smoke_info)
                    
                    results.append({
                        'file_name': tiff_file.name,
                        'date': file_date,
                        'smoke_detected': smoke_info['has_smoke'],
                        'smoke_percentage': smoke_info['smoke_percentage'],
                        'preview_path': str(preview_path),
                        'smoke_index_path': str(preview_path.with_name(f"{preview_path.stem}_smoke_index.png"))
                    })
            except (ValueError, IndexError):
                print(f"Не удалось обработать файл: {tiff_file.name}")
                continue
        
        return results

def main():
    analyzer = ImageAnalyzer()
    results = analyzer.analyze_recent_images()
    
    print("\nРезультаты анализа:")
    for result in results:
        print(f"\nФайл: {result['file_name']}")
        print(f"Дата: {result['date'].strftime('%Y-%m-%d')}")
        print(f"Обнаружен дым: {'Да' if result['smoke_detected'] else 'Нет'}")
        print(f"Процент площади с дымом: {result['smoke_percentage']:.2f}%")
        print(f"Превью: {result['preview_path']}")
        if result['smoke_detected']:
            print(f"Карта индекса дымов: {result['smoke_index_path']}")

if __name__ == "__main__":
    main() 