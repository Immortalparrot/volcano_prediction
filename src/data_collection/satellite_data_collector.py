import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import rasterio
from rasterio.windows import Window
from pathlib import Path
import shutil
import json
from eq_eruption_parser import load_events, filter_events_by_periods
from PIL import Image

class SatelliteDataCollector:
    def __init__(self, 
                 eruptions_file: str = 'data/satellite_images/klyuchevskoy_eruptions.csv',
                 raw_data_dir: str = 'data/raw',
                 processed_data_dir: str = 'data/processed',
                 satellite_data_dir: str = 'data/satellite_images',
                 days_before_eruption: int = 14,
                 days_after_eruption: int = 14,
                 max_cloud_coverage: float = 0.3,
                 pre_eruption_days: int = 30):
        """
        Инициализация коллектора спутниковых данных
        
        Args:
            eruptions_file: Путь к файлу с данными об извержениях
            raw_data_dir: Директория для исходных данных
            processed_data_dir: Директория для обработанных данных
            satellite_data_dir: Директория для финальных данных
            days_before_eruption: Количество дней до извержения для сбора данных
            days_after_eruption: Количество дней после извержения для сбора данных
            max_cloud_coverage: Максимально допустимый процент облачности
            pre_eruption_days: Длина периода до извержения для прогноза (по умолчанию 30)
        """
        self.eruptions_file = eruptions_file
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.satellite_data_dir = Path(satellite_data_dir)
        self.days_before = days_before_eruption
        self.days_after = days_after_eruption
        self.max_cloud_coverage = max_cloud_coverage
        self.pre_eruption_days = pre_eruption_days
        
        # Создаем директории если они не существуют
        for directory in [self.raw_data_dir, self.processed_data_dir, self.satellite_data_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def load_eruption_data(self) -> pd.DataFrame:
        """Загрузка и проверка данных об извержениях"""
        df = pd.read_csv(self.eruptions_file)
        
        # Проверяем и конвертируем даты
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
        
        # Фильтруем только последние 10 лет
        current_year = datetime.now().year
        df = df[df['start_date'].dt.year >= current_year - 10]
        
        # Удаляем строки с некорректными датами
        df = df.dropna(subset=['start_date'])
        
        return df
    
    def get_date_ranges(self, eruption_data: pd.DataFrame) -> list:
        """Получение диапазонов дат для сбора данных"""
        date_ranges = []
        
        for _, row in eruption_data.iterrows():
            start_date = row['start_date']
            end_date = row['end_date'] if pd.notna(row['end_date']) else start_date + timedelta(days=30)
            
            # Добавляем период до извержения
            pre_eruption_start = start_date - timedelta(days=self.days_before)
            date_ranges.append((pre_eruption_start, start_date))
            
            # Добавляем период извержения
            date_ranges.append((start_date, end_date))
            
            # Добавляем период после извержения
            post_eruption_end = end_date + timedelta(days=self.days_after)
            date_ranges.append((end_date, post_eruption_end))
        
        return date_ranges
    
    def get_background_date_ranges(self, eruption_data: pd.DataFrame, interval_days: int = 14) -> list:
        """
        Генерация контрольных (фоновых) периодов между извержениями
        Args:
            eruption_data: DataFrame с датами извержений
            interval_days: Длина одного фонового периода (например, 14 дней)
        Returns:
            list: Список кортежей (start_date, end_date) для фоновых периодов
        """
        # Сортируем по дате начала
        eruption_data = eruption_data.sort_values('start_date').reset_index(drop=True)
        background_periods = []
        # Начало наблюдений — 10 лет назад
        observation_start = eruption_data['start_date'].min().replace(month=1, day=1)
        observation_end = datetime.now()
        # Собираем интервалы между извержениями
        prev_end = observation_start
        for _, row in eruption_data.iterrows():
            start = row['start_date']
            if prev_end < start:
                # Разбиваем длинный интервал на куски по interval_days
                curr = prev_end
                while curr + timedelta(days=interval_days) < start:
                    background_periods.append((curr, curr + timedelta(days=interval_days)))
                    curr += timedelta(days=interval_days)
            prev_end = max(prev_end, row['end_date'] if pd.notna(row['end_date']) else row['start_date']+timedelta(days=30))
        # Добавляем период после последнего извержения
        if prev_end < observation_end:
            curr = prev_end
            while curr + timedelta(days=interval_days) < observation_end:
                background_periods.append((curr, curr + timedelta(days=interval_days)))
                curr += timedelta(days=interval_days)
        return background_periods
    
    def get_pre_eruption_periods(self, eruption_data: pd.DataFrame) -> list:
        """
        Генерирует pre-eruption периоды для каждого извержения
        Returns: список кортежей (start_date, end_date)
        """
        pre_periods = []
        for _, row in eruption_data.iterrows():
            start_date = row['start_date']
            pre_start = start_date - timedelta(days=self.pre_eruption_days)
            pre_periods.append((pre_start, start_date))
        return pre_periods
    
    def process_satellite_image(self, image_path: str) -> dict:
        """
        Обработка одного спутникового снимка
        Args:
            image_path: Путь к спутниковому снимку
        Returns:
            dict: Метаданные обработанного снимка или None если снимок не подходит
        """
        try:
            with rasterio.open(image_path) as src:
                tags = src.tags()
                # 1. Пробуем взять облачность из тега CLOUD_COVER
                if 'CLOUD_COVER' in tags:
                    try:
                        cloud_coverage = float(tags['CLOUD_COVER']) / 100
                    except Exception:
                        cloud_coverage = 1.0
                # 2. Если нет — вычисляем по cloud mask (4-й канал)
                elif src.count >= 4:
                    clm = src.read(4)
                    cloud_coverage = np.mean(clm > 0)
                else:
                    cloud_coverage = 1.0
                print(f"[DEBUG] Файл: {image_path}, Облачность: {cloud_coverage:.2%}, Порог: {self.max_cloud_coverage:.2%}")
                metadata = {
                    'filename': Path(image_path).name,
                    'cloud_coverage': cloud_coverage,
                    'acquisition_time': tags.get('DATE', ''),
                    'acquisition_date': tags.get('DATE', ''),
                    'processing_level': tags.get('PROCESSING_LEVEL', ''),
                    'sensor': tags.get('SENSOR', ''),
                    'resolution': src.meta.get('resolution', ''),
                    'status': 'rejected'
                }
                # Проверяем облачность
                if metadata['cloud_coverage'] > self.max_cloud_coverage:
                    return metadata
                # Проверяем время съемки (если есть)
                if metadata['acquisition_time']:
                    try:
                        hour = int(metadata['acquisition_time'][11:13])
                        if not (6 <= hour <= 12):
                            return metadata
                    except Exception:
                        pass
                # Копируем снимок в директорию обработанных данных
                output_path = self.processed_data_dir / Path(image_path).name
                # Копируем только если исходный и целевой путь различаются
                if Path(image_path).resolve() != output_path.resolve():
                    shutil.copy2(image_path, output_path)
                # Генерируем PNG-превью
                try:
                    rgb = src.read([1, 2, 3])
                    # Растяжка по каждому каналу (2-98 перцентили)
                    rgb_stretched = np.zeros_like(rgb)
                    for i in range(3):
                        band = rgb[i]
                        band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
                        min_val = np.percentile(band, 2)
                        max_val = np.percentile(band, 98)
                        if max_val - min_val < 1e-6:
                            band = np.zeros_like(band)
                        else:
                            band = np.clip((band - min_val) / (max_val - min_val), 0, 1)
                        rgb_stretched[i] = (band * 255).astype(np.uint8)
                    img = np.transpose(rgb_stretched, (1, 2, 0))
                    png_path = self.processed_data_dir / (Path(image_path).stem + '_preview.png')
                    Image.fromarray(img).save(png_path)
                except Exception as e:
                    print(f"[WARNING] Не удалось создать PNG-превью для {image_path}: {e}")
                metadata['status'] = 'accepted'
                return metadata
        except Exception as e:
            print(f"Ошибка при обработке {image_path}: {str(e)}")
            return None
    
    def process_test_images(self, test_images_dir: str = None) -> dict:
        """
        Обработка тестовых спутниковых снимков
        
        Args:
            test_images_dir: Директория с тестовыми снимками
            
        Returns:
            dict: Статистика обработки снимков
        """
        if test_images_dir is None:
            test_images_dir = self.raw_data_dir
        
        test_images_dir = Path(test_images_dir)
        if not test_images_dir.exists():
            print(f"Директория {test_images_dir} не существует")
            return {}
        
        # Собираем все TIFF файлы
        image_files = list(test_images_dir.glob('*.tiff')) + list(test_images_dir.glob('*.tif'))
        
        if not image_files:
            print(f"В директории {test_images_dir} не найдено TIFF файлов")
            return {}
        
        # Обрабатываем каждый снимок
        results = {
            'total_images': len(image_files),
            'processed_images': [],
            'rejected_images': [],
            'cloud_coverage_stats': {
                'min': float('inf'),
                'max': 0,
                'mean': 0,
                'total': 0
            }
        }
        
        for image_path in image_files:
            metadata = self.process_satellite_image(str(image_path))
            if metadata:
                if metadata['status'] == 'accepted':
                    results['processed_images'].append(metadata)
                else:
                    results['rejected_images'].append(metadata)
                
                # Обновляем статистику облачности
                cloud_coverage = metadata['cloud_coverage']
                results['cloud_coverage_stats']['min'] = min(results['cloud_coverage_stats']['min'], cloud_coverage)
                results['cloud_coverage_stats']['max'] = max(results['cloud_coverage_stats']['max'], cloud_coverage)
                results['cloud_coverage_stats']['total'] += cloud_coverage
        
        # Вычисляем среднюю облачность
        total_processed = len(results['processed_images']) + len(results['rejected_images'])
        if total_processed > 0:
            results['cloud_coverage_stats']['mean'] = results['cloud_coverage_stats']['total'] / total_processed
        
        # Сохраняем результаты
        results_path = self.satellite_data_dir / 'test_processing_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def collect_data(self, earthquake_file: str = 'Data2018.xlsx'):
        """Основной метод для сбора и обработки данных"""
        # Загружаем данные об извержениях
        eruption_data = self.load_eruption_data()
        # Загружаем данные о землетрясениях
        earthquakes_df = load_events(earthquake_file)
        # Получаем периоды
        eruption_date_ranges = self.get_date_ranges(eruption_data)
        background_date_ranges = self.get_background_date_ranges(eruption_data)
        pre_eruption_periods = self.get_pre_eruption_periods(eruption_data)
        # Фильтруем землетрясения по периодам
        eqs_pre = filter_events_by_periods(earthquakes_df, pre_eruption_periods)
        eqs_eruption = filter_events_by_periods(earthquakes_df, eruption_date_ranges)
        eqs_background = filter_events_by_periods(earthquakes_df, background_date_ranges)
        # Метаданные
        metadata = {
            'eruption_periods': [],
            'background_periods': [],
            'pre_eruption_periods': [],
            'processed_images': [],
            'cloud_coverage_stats': {
                'min': float('inf'),
                'max': 0,
                'mean': 0,
                'total_images': 0
            }
        }
        # pre-eruption
        for i, (start_date, end_date) in enumerate(pre_eruption_periods):
            period_images = []
            # Здесь должен быть код для получения спутниковых снимков
            # period_images = self.get_images_for_period(start_date, end_date)
            earthquakes = eqs_pre[i].to_dict(orient='records')
            metadata['pre_eruption_periods'].append({
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'images': period_images,
                'earthquakes': earthquakes,
                'period_type': 'pre-eruption'
            })
        # eruption
        for i, (start_date, end_date) in enumerate(eruption_date_ranges):
            period_images = []
            # period_images = self.get_images_for_period(start_date, end_date)
            earthquakes = eqs_eruption[i].to_dict(orient='records')
            metadata['eruption_periods'].append({
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'images': period_images,
                'earthquakes': earthquakes,
                'period_type': 'eruption'
            })
        # background
        for i, (start_date, end_date) in enumerate(background_date_ranges):
            period_images = []
            # period_images = self.get_images_for_period(start_date, end_date)
            earthquakes = eqs_background[i].to_dict(orient='records')
            metadata['background_periods'].append({
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'images': period_images,
                'earthquakes': earthquakes,
                'period_type': 'background'
            })
        # Сохраняем метаданные
        metadata_path = self.satellite_data_dir / 'metadata.json'
        pd.DataFrame(metadata['eruption_periods'] + metadata['background_periods'] + metadata['pre_eruption_periods']).to_json(metadata_path, orient='records')
        return metadata

if __name__ == '__main__':
    # Создаем экземпляр коллектора
    collector = SatelliteDataCollector()
    
    # Обрабатываем тестовые снимки
    test_results = collector.process_test_images()
    
    # Выводим статистику
    print("\nСтатистика обработки тестовых снимков:")
    print(f"Всего снимков: {test_results.get('total_images', 0)}")
    print(f"Обработано успешно: {len(test_results.get('processed_images', []))}")
    print(f"Отклонено: {len(test_results.get('rejected_images', []))}")
    
    if test_results.get('cloud_coverage_stats'):
        stats = test_results['cloud_coverage_stats']
        print("\nСтатистика облачности:")
        print(f"Минимальная облачность: {stats['min']:.2%}")
        print(f"Максимальная облачность: {stats['max']:.2%}")
        print(f"Средняя облачность: {stats['mean']:.2%}") 