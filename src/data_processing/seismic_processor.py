import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Union

class SeismicProcessor:
    def __init__(self, config):
        self.config = config
        self.window_size = config['seismic']['window_size']
        self.features = config['seismic']['features']
    
    def load_seismic_data(self, file_path: str) -> pd.DataFrame:
        """Загрузка сейсмических данных из CSV файла"""
        try:
            df = pd.read_csv(file_path)
            
            # Проверка наличия необходимых колонок
            required_columns = ['timestamp', 'magnitude', 'depth', 'latitude', 'longitude']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Отсутствуют необходимые колонки: {missing_columns}")
            
            # Преобразование timestamp в datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Сортировка по времени
            df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            raise Exception(f"Ошибка при загрузке сейсмических данных: {str(e)}")
    
    def process_time_series(self, 
                          data: pd.DataFrame,
                          start_time: datetime,
                          end_time: datetime) -> pd.DataFrame:
        """Обработка временных рядов сейсмических данных"""
        try:
            # Фильтрация по временному диапазону
            mask = (data['timestamp'] >= start_time) & (data['timestamp'] <= end_time)
            filtered_data = data[mask].copy()
            
            # Создание временного индекса с часовым интервалом
            time_index = pd.date_range(start=start_time, end=end_time, freq='H')
            
            # Инициализация DataFrame для признаков
            features_df = pd.DataFrame(index=time_index)
            
            # Расчет признаков для каждого временного окна
            for feature in self.features:
                # Скользящее окно для каждого признака
                features_df[f'{feature}_mean'] = self._calculate_rolling_mean(
                    filtered_data, feature, self.window_size
                )
                features_df[f'{feature}_std'] = self._calculate_rolling_std(
                    filtered_data, feature, self.window_size
                )
                features_df[f'{feature}_max'] = self._calculate_rolling_max(
                    filtered_data, feature, self.window_size
                )
            
            # Расчет дополнительных признаков
            features_df['event_count'] = self._calculate_event_count(
                filtered_data, self.window_size
            )
            features_df['energy_release'] = self._calculate_energy_release(
                filtered_data, self.window_size
            )
            
            # Заполнение пропущенных значений
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            return features_df
            
        except Exception as e:
            raise Exception(f"Ошибка при обработке временных рядов: {str(e)}")
    
    def _calculate_rolling_mean(self, 
                              data: pd.DataFrame,
                              feature: str,
                              window: int) -> pd.Series:
        """Расчет скользящего среднего"""
        return data.set_index('timestamp')[feature].rolling(
            window=window,
            min_periods=1
        ).mean()
    
    def _calculate_rolling_std(self,
                             data: pd.DataFrame,
                             feature: str,
                             window: int) -> pd.Series:
        """Расчет скользящего стандартного отклонения"""
        return data.set_index('timestamp')[feature].rolling(
            window=window,
            min_periods=1
        ).std()
    
    def _calculate_rolling_max(self,
                             data: pd.DataFrame,
                             feature: str,
                             window: int) -> pd.Series:
        """Расчет скользящего максимума"""
        return data.set_index('timestamp')[feature].rolling(
            window=window,
            min_periods=1
        ).max()
    
    def _calculate_event_count(self,
                             data: pd.DataFrame,
                             window: int) -> pd.Series:
        """Расчет количества событий в окне"""
        return data.set_index('timestamp').resample('H').size().rolling(
            window=window,
            min_periods=1
        ).sum()
    
    def _calculate_energy_release(self,
                                data: pd.DataFrame,
                                window: int) -> pd.Series:
        """Расчет выделенной энергии"""
        # Формула для расчета энергии: E = 10^(1.5 * M + 4.8)
        data['energy'] = 10 ** (1.5 * data['magnitude'] + 4.8)
        return data.set_index('timestamp')['energy'].rolling(
            window=window,
            min_periods=1
        ).sum()
    
    def detect_swarms(self,
                     data: pd.DataFrame,
                     min_events: int = 10,
                     time_window: int = 24,
                     magnitude_threshold: float = 2.0) -> List[Dict]:
        """Обнаружение сейсмических роев"""
        swarms = []
        
        # Фильтрация по магнитуде
        filtered_data = data[data['magnitude'] >= magnitude_threshold].copy()
        
        # Группировка по временным окнам
        for i in range(len(filtered_data) - min_events):
            window_data = filtered_data.iloc[i:i + min_events]
            
            # Проверка временного окна
            time_diff = (window_data['timestamp'].max() - 
                        window_data['timestamp'].min()).total_seconds() / 3600
            
            if time_diff <= time_window:
                # Расчет параметров роя
                swarm = {
                    'start_time': window_data['timestamp'].min(),
                    'end_time': window_data['timestamp'].max(),
                    'event_count': len(window_data),
                    'max_magnitude': window_data['magnitude'].max(),
                    'mean_magnitude': window_data['magnitude'].mean(),
                    'mean_depth': window_data['depth'].mean(),
                    'center_lat': window_data['latitude'].mean(),
                    'center_lon': window_data['longitude'].mean()
                }
                swarms.append(swarm)
        
        return swarms 