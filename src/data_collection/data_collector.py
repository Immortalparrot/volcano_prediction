import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import os
from typing import List, Dict, Tuple
import requests
from tqdm import tqdm
import json
import rasterio
from src.data_collection.image_analyzer import ImageAnalyzer
from src.data_collection.sentinel_oauth_client import SentinelHubOAuthClient

class DataCollector:
    def __init__(self, data_dir: str = "data", config_path: str = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        self.metadata_dir = self.data_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        self.image_analyzer = ImageAnalyzer(data_dir=str(self.images_dir))
        # Новый OAuth2 клиент
        self.sentinel_client = SentinelHubOAuthClient(config_path=config_path)
        self.volcano_bbox = self.sentinel_client.volcano_bbox

    def load_eruption_dates(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        return df

    def generate_labeled_periods(self, eruption_df: pd.DataFrame, pre_days: int = 14, post_days: int = 14) -> list:
        periods = []
        eruption_df = eruption_df.sort_values('start_date').reset_index(drop=True)
        for idx, row in eruption_df.iterrows():
            start = row['start_date']
            end = row['end_date'] if pd.notna(row['end_date']) else row['start_date']
            pre_start = start - pd.Timedelta(days=pre_days)
            periods.append((pre_start, start, 'pre-eruption'))
            periods.append((start, end, 'eruption'))
            post_end = end + pd.Timedelta(days=post_days)
            periods.append((end, post_end, 'post-eruption'))
            if idx < len(eruption_df) - 1:
                next_start = eruption_df.loc[idx + 1, 'start_date']
                bg_start = post_end
                bg_end = min(bg_start + pd.Timedelta(days=pre_days + (end - start).days + post_days), next_start - pd.Timedelta(days=1))
                if bg_start < bg_end:
                    periods.append((bg_start, bg_end, 'background'))
        return periods

    def collect_labeled_images(self, periods: list):
        results = []
        for start_date, end_date, label in tqdm(periods, desc="Сбор снимков по периодам"):
            try:
                # Скачиваем снимок через OAuth2 клиент
                tiff_path = self.sentinel_client.download_image(
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    output_dir=self.images_dir,
                    bbox=self.volcano_bbox
                )
                if not tiff_path.exists():
                    print(f"[ERROR] Файл не был создан: {tiff_path}")
                    continue
                print(f"\nОбработка файла: {tiff_path.name} [{label}]")
                image_data = self.image_analyzer.load_tiff(tiff_path)
                smoke_info = self.image_analyzer.check_for_smoke(image_data['data'])
                preview_path = tiff_path.with_suffix('.png')
                try:
                    self.image_analyzer.create_preview(image_data['data'], preview_path, smoke_info)
                except Exception as e:
                    print(f"[WARNING] Не удалось создать превью для {tiff_path.name}: {e}")
                # Чтение облачности из метаданных TIFF
                with rasterio.open(tiff_path) as src:
                    tags = src.tags()
                    cloud_cover = float(tags.get('CLOUD_COVER', 1.0))
                    date_tag = tags.get('DATE', str(start_date))
                results.append({
                    'file_name': tiff_path.name,
                    'date': date_tag,
                    'band': 'RGB',
                    'cloud_cover': cloud_cover,
                    'smoke_detected': bool(smoke_info['has_smoke']),
                    'smoke_percentage': float(smoke_info['smoke_percentage']),
                    'preview_path': str(preview_path),
                    'smoke_index_path': str(preview_path.with_name(f"{preview_path.stem}_smoke_index.png")),
                    'channel_info': image_data['channel_info'],
                    'label': label
                })
            except Exception as e:
                print(f"Ошибка при обработке периода {start_date} - {end_date} [{label}]: {str(e)}")
                continue
        results_file = self.metadata_dir / "analysis_results_labeled.json"
        print(f"[DEBUG] Сохраняю {len(results)} результатов в {results_file}")
        try:
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4, default=str)
            print(f"[DEBUG] Файл успешно сохранён: {results_file}")
        except Exception as e:
            print(f"[ERROR] Не удалось сохранить файл {results_file}: {e}") 