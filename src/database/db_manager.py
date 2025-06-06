import os
import psycopg2
from psycopg2.extras import DictCursor
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

class DatabaseManager:
    """Класс для управления базой данных"""
    
    def __init__(self):
        """Инициализация менеджера базы данных"""
        self.db_params = {
            'dbname': os.getenv('POSTGRES_DB', 'volcano_prediction'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432')
        }
        self.logger = logging.getLogger(__name__)
    
    def get_connection(self):
        """Получение соединения с базой данных"""
        return psycopg2.connect(**self.db_params)
    
    def save_satellite_data(self, data: Dict[str, Any]) -> int:
        """Сохранение спутниковых данных"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO satellite_data 
                    (timestamp, ndvi, ndbi, ndbai, thermal_anomaly, 
                     smoke_detected, image_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    data['timestamp'],
                    data.get('ndvi'),
                    data.get('ndbi'),
                    data.get('ndbai'),
                    data.get('thermal_anomaly', False),
                    data.get('smoke_detected', False),
                    data.get('image_path')
                ))
                return cursor.fetchone()[0]
    
    def save_seismic_data(self, data: Dict[str, Any]) -> int:
        """Сохранение сейсмических данных"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO seismic_data 
                    (timestamp, magnitude, depth, latitude, longitude, event_type)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    data['timestamp'],
                    data.get('magnitude'),
                    data.get('depth'),
                    data.get('latitude'),
                    data.get('longitude'),
                    data.get('event_type')
                ))
                return cursor.fetchone()[0]
    
    def save_eruption(self, data: Dict[str, Any]) -> int:
        """Сохранение данных об извержении"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO eruptions 
                    (start_time, end_time, magnitude, description)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (
                    data['start_time'],
                    data.get('end_time'),
                    data.get('magnitude'),
                    data.get('description')
                ))
                return cursor.fetchone()[0]
    
    def save_prediction(self, data: Dict[str, Any]) -> int:
        """Сохранение предсказания модели"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO predictions 
                    (timestamp, probability, threshold, is_eruption_predicted,
                     feature_importance, temporal_importance)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    data['timestamp'],
                    data['probability'],
                    data['threshold'],
                    data['is_eruption_predicted'],
                    json.dumps(data.get('feature_importance', {})),
                    json.dumps(data.get('temporal_importance', {}))
                ))
                return cursor.fetchone()[0]
    
    def get_satellite_data(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Получение спутниковых данных за период"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM satellite_data
                    WHERE timestamp BETWEEN %s AND %s
                    ORDER BY timestamp
                """, (start_time, end_time))
                return [dict(row) for row in cursor.fetchall()]
    
    def get_seismic_data(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Получение сейсмических данных за период"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM seismic_data
                    WHERE timestamp BETWEEN %s AND %s
                    ORDER BY timestamp
                """, (start_time, end_time))
                return [dict(row) for row in cursor.fetchall()]
    
    def get_eruptions(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Получение данных об извержениях за период"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM eruptions
                    WHERE start_time BETWEEN %s AND %s
                    ORDER BY start_time
                """, (start_time, end_time))
                return [dict(row) for row in cursor.fetchall()]
    
    def get_predictions(self, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Получение предсказаний за период"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM predictions
                    WHERE timestamp BETWEEN %s AND %s
                    ORDER BY timestamp
                """, (start_time, end_time))
                return [dict(row) for row in cursor.fetchall()]
    
    def get_latest_prediction(self) -> Optional[Dict]:
        """Получение последнего предсказания"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute("""
                    SELECT * FROM predictions
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                row = cursor.fetchone()
                return dict(row) if row else None 