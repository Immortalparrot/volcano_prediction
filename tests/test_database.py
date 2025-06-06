import pytest
import os
from datetime import datetime, timedelta
from src.database.db_manager import DatabaseManager

@pytest.fixture
def db_manager():
    """Фикстура для создания менеджера базы данных"""
    return DatabaseManager()

@pytest.fixture
def sample_satellite_data():
    """Фикстура с тестовыми спутниковыми данными"""
    return {
        'timestamp': datetime.now(),
        'ndvi': 0.5,
        'ndbi': 0.3,
        'ndbai': 0.2,
        'thermal_anomaly': True,
        'smoke_detected': False,
        'image_path': 'data/satellite_images/test.jpg'
    }

@pytest.fixture
def sample_seismic_data():
    """Фикстура с тестовыми сейсмическими данными"""
    return {
        'timestamp': datetime.now(),
        'magnitude': 4.5,
        'depth': 10.0,
        'latitude': 56.0,
        'longitude': 160.0,
        'event_type': 'volcanic'
    }

@pytest.fixture
def sample_eruption():
    """Фикстура с тестовыми данными об извержении"""
    return {
        'start_time': datetime.now(),
        'end_time': datetime.now() + timedelta(hours=2),
        'magnitude': 3.0,
        'description': 'Тестовое извержение'
    }

@pytest.fixture
def sample_prediction():
    """Фикстура с тестовыми данными предсказания"""
    return {
        'timestamp': datetime.now(),
        'probability': 0.8,
        'threshold': 0.5,
        'is_eruption_predicted': True,
        'feature_importance': {'ndvi': 0.3, 'magnitude': 0.7},
        'temporal_importance': {'last_24h': 0.6, 'last_48h': 0.4}
    }

def test_save_satellite_data(db_manager, sample_satellite_data):
    """Тест сохранения спутниковых данных"""
    data_id = db_manager.save_satellite_data(sample_satellite_data)
    assert data_id is not None
    
    # Проверка получения данных
    start_time = sample_satellite_data['timestamp'] - timedelta(minutes=1)
    end_time = sample_satellite_data['timestamp'] + timedelta(minutes=1)
    data = db_manager.get_satellite_data(start_time, end_time)
    
    assert len(data) == 1
    assert data[0]['ndvi'] == sample_satellite_data['ndvi']
    assert data[0]['thermal_anomaly'] == sample_satellite_data['thermal_anomaly']

def test_save_seismic_data(db_manager, sample_seismic_data):
    """Тест сохранения сейсмических данных"""
    data_id = db_manager.save_seismic_data(sample_seismic_data)
    assert data_id is not None
    
    # Проверка получения данных
    start_time = sample_seismic_data['timestamp'] - timedelta(minutes=1)
    end_time = sample_seismic_data['timestamp'] + timedelta(minutes=1)
    data = db_manager.get_seismic_data(start_time, end_time)
    
    assert len(data) == 1
    assert data[0]['magnitude'] == sample_seismic_data['magnitude']
    assert data[0]['event_type'] == sample_seismic_data['event_type']

def test_save_eruption(db_manager, sample_eruption):
    """Тест сохранения данных об извержении"""
    data_id = db_manager.save_eruption(sample_eruption)
    assert data_id is not None
    
    # Проверка получения данных
    start_time = sample_eruption['start_time'] - timedelta(minutes=1)
    end_time = sample_eruption['start_time'] + timedelta(minutes=1)
    data = db_manager.get_eruptions(start_time, end_time)
    
    assert len(data) == 1
    assert data[0]['magnitude'] == sample_eruption['magnitude']
    assert data[0]['description'] == sample_eruption['description']

def test_save_prediction(db_manager, sample_prediction):
    """Тест сохранения предсказания"""
    data_id = db_manager.save_prediction(sample_prediction)
    assert data_id is not None
    
    # Проверка получения данных
    start_time = sample_prediction['timestamp'] - timedelta(minutes=1)
    end_time = sample_prediction['timestamp'] + timedelta(minutes=1)
    data = db_manager.get_predictions(start_time, end_time)
    
    assert len(data) == 1
    assert data[0]['probability'] == sample_prediction['probability']
    assert data[0]['is_eruption_predicted'] == sample_prediction['is_eruption_predicted']

def test_get_latest_prediction(db_manager, sample_prediction):
    """Тест получения последнего предсказания"""
    # Сохранение предсказания
    db_manager.save_prediction(sample_prediction)
    
    # Получение последнего предсказания
    latest = db_manager.get_latest_prediction()
    assert latest is not None
    assert latest['probability'] == sample_prediction['probability']
    assert latest['is_eruption_predicted'] == sample_prediction['is_eruption_predicted']

def test_empty_data_retrieval(db_manager):
    """Тест получения данных за период без записей"""
    start_time = datetime.now() - timedelta(days=1)
    end_time = datetime.now()
    
    satellite_data = db_manager.get_satellite_data(start_time, end_time)
    seismic_data = db_manager.get_seismic_data(start_time, end_time)
    eruptions = db_manager.get_eruptions(start_time, end_time)
    predictions = db_manager.get_predictions(start_time, end_time)
    
    assert len(satellite_data) == 0
    assert len(seismic_data) == 0
    assert len(eruptions) == 0
    assert len(predictions) == 0 