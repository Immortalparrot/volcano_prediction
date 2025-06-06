import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
from src.data_processing.satellite_processor import SatelliteProcessor
from src.data_processing.seismic_processor import SeismicProcessor

@pytest.fixture
def config():
    """Фикстура с конфигурацией обработки данных"""
    config_path = Path('config/model_config.json')
    if not config_path.exists():
        raise FileNotFoundError("Файл конфигурации не найден")
    
    with open(config_path) as f:
        return json.load(f)

@pytest.fixture
def satellite_processor():
    """Фикстура для SatelliteProcessor"""
    config = {
        'image_size': 256,
        'channels': 6,
        'normalize': True,
        'thermal_threshold': 300,
        'smoke_threshold': 0.3
    }
    return SatelliteProcessor(config)

@pytest.fixture
def seismic_processor():
    """Фикстура для SeismicProcessor"""
    config = {
        'window_size': 24,
        'min_magnitude': 1.0,
        'max_depth': 30
    }
    return SeismicProcessor(config)

@pytest.fixture
def sample_satellite_data():
    """Фикстура для тестовых спутниковых данных"""
    # Создание тестового изображения
    image = np.random.rand(6, 256, 256) * 10000
    return image.astype(np.uint16)

@pytest.fixture
def sample_seismic_data():
    """Фикстура для тестовых сейсмических данных"""
    # Создание тестовых данных
    data = {
        'timestamp': pd.date_range(
            start=datetime(2020, 1, 1),
            periods=100,
            freq='H'
        ),
        'magnitude': np.random.uniform(1.0, 5.0, 100),
        'depth': np.random.uniform(0, 30, 100),
        'latitude': np.random.uniform(56.0, 56.5, 100),
        'longitude': np.random.uniform(160.5, 161.0, 100)
    }
    return pd.DataFrame(data)

def test_satellite_processor_initialization(satellite_processor):
    """Тест инициализации SatelliteProcessor"""
    assert satellite_processor.config['image_size'] == 256
    assert satellite_processor.config['channels'] == 6
    assert satellite_processor.config['normalize'] is True

def test_seismic_processor_initialization(seismic_processor):
    """Тест инициализации SeismicProcessor"""
    assert seismic_processor.config['window_size'] == 24
    assert seismic_processor.config['min_magnitude'] == 1.0
    assert seismic_processor.config['max_depth'] == 30

def test_satellite_data_processing(satellite_processor, sample_satellite_data):
    """Тест обработки спутниковых данных"""
    # Обработка данных
    processed_data = satellite_processor.process_image(sample_satellite_data)
    
    # Проверка результатов
    assert isinstance(processed_data, np.ndarray)
    assert processed_data.shape[0] == 10  # Количество признаков
    assert not np.isnan(processed_data).any()

def test_seismic_data_processing(seismic_processor, sample_seismic_data):
    """Тест обработки сейсмических данных"""
    # Обработка данных
    processed_data = seismic_processor.process_data(sample_seismic_data)
    
    # Проверка результатов
    assert isinstance(processed_data, np.ndarray)
    assert processed_data.shape[1] == 12  # Количество признаков
    assert not np.isnan(processed_data).any()

def test_anomaly_detection(satellite_processor, sample_satellite_data):
    """Тест обнаружения аномалий в спутниковых данных"""
    # Добавление аномалии (горячая точка)
    sample_satellite_data[0, 100:150, 100:150] = 5000
    
    # Обнаружение аномалий
    anomalies = satellite_processor.detect_anomalies(sample_satellite_data)
    
    # Проверка результатов
    assert isinstance(anomalies, dict)
    assert 'thermal' in anomalies
    assert 'smoke' in anomalies
    assert len(anomalies['thermal']) > 0

def test_seismic_swarm_detection(seismic_processor, sample_seismic_data):
    """Тест обнаружения сейсмических роев"""
    # Добавление роя землетрясений
    sample_seismic_data.loc[10:20, 'magnitude'] = 4.5
    
    # Обнаружение роев
    swarms = seismic_processor.detect_swarms(sample_seismic_data)
    
    # Проверка результатов
    assert isinstance(swarms, list)
    assert len(swarms) > 0
    assert all(isinstance(swarm, dict) for swarm in swarms)

def test_data_normalization(satellite_processor, sample_satellite_data):
    """Тест нормализации данных"""
    # Нормализация данных
    normalized_data = satellite_processor.normalize_data(sample_satellite_data)
    
    # Проверка результатов
    assert isinstance(normalized_data, np.ndarray)
    assert normalized_data.min() >= 0
    assert normalized_data.max() <= 1

def test_feature_extraction(seismic_processor, sample_seismic_data):
    """Тест извлечения признаков из сейсмических данных"""
    # Извлечение признаков
    features = seismic_processor.extract_features(sample_seismic_data)
    
    # Проверка результатов
    assert isinstance(features, np.ndarray)
    assert features.shape[1] == 12  # Количество признаков
    assert not np.isnan(features).any()

def test_data_validation(satellite_processor, sample_satellite_data):
    """Тест валидации данных"""
    # Валидация данных
    is_valid = satellite_processor.validate_data(sample_satellite_data)
    
    # Проверка результатов
    assert isinstance(is_valid, bool)
    assert is_valid is True

def test_error_handling(satellite_processor):
    """Тест обработки ошибок"""
    # Тест с некорректными данными
    invalid_data = np.random.rand(3, 100, 100)  # Неправильное количество каналов
    
    with pytest.raises(ValueError):
        satellite_processor.process_image(invalid_data)

def test_data_saving(satellite_processor, sample_satellite_data, tmp_path):
    """Тест сохранения данных"""
    # Сохранение данных
    output_path = tmp_path / "test_output.npy"
    satellite_processor.save_data(sample_satellite_data, output_path)
    
    # Проверка результатов
    assert output_path.exists()
    loaded_data = np.load(output_path)
    assert np.array_equal(sample_satellite_data, loaded_data)

def test_satellite_processor(config, sample_satellite_image):
    """Тест обработки спутниковых данных"""
    from src.data_processing.satellite_processor import SatelliteProcessor
    
    processor = SatelliteProcessor(config['data']['satellite'])
    
    # Тест обработки изображения
    processed_image = processor.process_image(sample_satellite_image)
    assert processed_image.shape == (256, 256, 4)
    assert np.all(processed_image >= 0) and np.all(processed_image <= 1)
    
    # Тест расчета индексов
    ndvi = processor.calculate_ndvi(sample_satellite_image)
    assert ndvi.shape == (256, 256)
    assert np.all(ndvi >= -1) and np.all(ndvi <= 1)
    
    ndbi = processor.calculate_ndbi(sample_satellite_image)
    assert ndbi.shape == (256, 256)
    assert np.all(ndbi >= -1) and np.all(ndbi <= 1)
    
    # Тест обнаружения тепловых аномалий
    thermal_anomaly = processor.detect_thermal_anomaly(sample_satellite_image)
    assert isinstance(thermal_anomaly, bool)

def test_seismic_processor(config, sample_seismic_data):
    """Тест обработки сейсмических данных"""
    from src.data_processing.seismic_processor import SeismicProcessor
    
    processor = SeismicProcessor(config['data']['seismic'])
    
    # Тест загрузки данных
    loaded_data = processor.load_seismic_data(sample_seismic_data)
    assert isinstance(loaded_data, pd.DataFrame)
    assert all(col in loaded_data.columns for col in ['timestamp', 'magnitude', 'depth'])
    
    # Тест обработки временного ряда
    window_size = config['data']['seismic']['window_size']
    processed_data = processor.process_time_series(loaded_data, window_size)
    assert isinstance(processed_data, pd.DataFrame)
    assert len(processed_data) > 0
    
    # Тест расчета признаков
    features = processor.calculate_features(loaded_data)
    assert isinstance(features, dict)
    assert all(key in features for key in ['mean_magnitude', 'std_magnitude', 'max_magnitude'])
    
    # Тест обнаружения роев
    swarms = processor.detect_swarms(loaded_data)
    assert isinstance(swarms, list)

def test_data_combination(config, sample_satellite_image, sample_seismic_data):
    """Тест комбинирования данных"""
    from src.data_processing.satellite_processor import SatelliteProcessor
    from src.data_processing.seismic_processor import SeismicProcessor
    
    satellite_processor = SatelliteProcessor(config['data']['satellite'])
    seismic_processor = SeismicProcessor(config['data']['seismic'])
    
    # Обработка спутниковых данных
    processed_image = satellite_processor.process_image(sample_satellite_image)
    satellite_features = {
        'ndvi_mean': np.mean(satellite_processor.calculate_ndvi(processed_image)),
        'ndbi_mean': np.mean(satellite_processor.calculate_ndbi(processed_image)),
        'thermal_anomaly': satellite_processor.detect_thermal_anomaly(processed_image)
    }
    
    # Обработка сейсмических данных
    processed_seismic = seismic_processor.process_time_series(sample_seismic_data, 
                                                            config['data']['seismic']['window_size'])
    seismic_features = seismic_processor.calculate_features(processed_seismic)
    
    # Комбинирование признаков
    combined_features = {**satellite_features, **seismic_features}
    assert isinstance(combined_features, dict)
    assert len(combined_features) > 0

def test_data_normalization(config, sample_satellite_image, sample_seismic_data):
    """Тест нормализации данных"""
    from src.data_processing.satellite_processor import SatelliteProcessor
    from src.data_processing.seismic_processor import SeismicProcessor
    
    satellite_processor = SatelliteProcessor(config['data']['satellite'])
    seismic_processor = SeismicProcessor(config['data']['seismic'])
    
    # Нормализация спутниковых данных
    normalized_image = satellite_processor.normalize_image(sample_satellite_image)
    assert np.all(normalized_image >= 0) and np.all(normalized_image <= 1)
    
    # Нормализация сейсмических данных
    normalized_seismic = seismic_processor.normalize_data(sample_seismic_data)
    assert isinstance(normalized_seismic, pd.DataFrame)
    assert all(col in normalized_seismic.columns for col in sample_seismic_data.columns)

def test_error_handling(config):
    """Тест обработки ошибок"""
    from src.data_processing.satellite_processor import SatelliteProcessor
    from src.data_processing.seismic_processor import SeismicProcessor
    
    satellite_processor = SatelliteProcessor(config['data']['satellite'])
    seismic_processor = SeismicProcessor(config['data']['seismic'])
    
    # Тест обработки некорректного изображения
    invalid_image = np.random.rand(100, 100, 3)  # Неправильное количество каналов
    with pytest.raises(ValueError):
        satellite_processor.process_image(invalid_image)
    
    # Тест обработки некорректных сейсмических данных
    invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
    with pytest.raises(ValueError):
        seismic_processor.load_seismic_data(invalid_data) 