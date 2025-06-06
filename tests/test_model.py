import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
from src.models.temporal_fusion_transformer import TemporalFusionTransformer

@pytest.fixture
def config():
    """Фикстура с конфигурацией модели"""
    config_path = Path('config/model_config.json')
    if not config_path.exists():
        raise FileNotFoundError("Файл конфигурации не найден")
    
    with open(config_path) as f:
        return json.load(f)

@pytest.fixture
def sample_input_data():
    """Фикстура с тестовыми входными данными"""
    # Создаем тестовые данные
    batch_size = 32
    seq_length = 24
    num_features = 10
    
    # Создаем временной ряд
    dates = pd.date_range(start='2023-01-01', periods=seq_length, freq='H')
    
    # Создаем тензор с признаками
    features = torch.randn(batch_size, seq_length, num_features)
    
    # Создаем маску для временного ряда
    mask = torch.ones(batch_size, seq_length)
    
    return {
        'features': features,
        'dates': dates,
        'mask': mask
    }

@pytest.fixture
def model_config():
    """Фикстура с конфигурацией модели"""
    return {
        'model': {
            'hidden_size': 256,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1
        },
        'satellite': {
            'num_features': 10
        },
        'seismic': {
            'num_features': 12
        },
        'time': {
            'num_features': 8
        }
    }

@pytest.fixture
def sample_batch():
    """Фикстура с тестовым батчем данных"""
    batch_size = 32
    seq_len = 24
    
    return {
        'satellite': torch.randn(batch_size, seq_len, 10),
        'seismic': torch.randn(batch_size, seq_len, 12),
        'time': torch.randn(batch_size, seq_len, 8),
        'target': torch.randint(0, 2, (batch_size, 1)).float()
    }

@pytest.fixture
def model(model_config):
    """Фикстура с моделью"""
    return TemporalFusionTransformer(model_config)

def test_model_initialization(model, model_config):
    """Тест инициализации модели"""
    assert model.hidden_size == model_config['model']['hidden_size']
    assert model.num_heads == model_config['model']['num_heads']
    assert model.num_layers == model_config['model']['num_layers']
    assert model.dropout == model_config['model']['dropout']

def test_model_forward_pass(model, sample_batch):
    """Тест прямого прохода модели"""
    # Прямой проход
    output = model(
        sample_batch['satellite'],
        sample_batch['seismic'],
        sample_batch['time']
    )
    
    # Проверка результатов
    assert isinstance(output, torch.Tensor)
    assert output.shape == (32, 1)  # [batch_size, 1]
    assert torch.all((output >= 0) & (output <= 1))  # Проверка диапазона

def test_model_prediction(model, sample_batch):
    """Тест предсказания модели"""
    # Получение предсказаний
    predictions, probabilities = model.predict(
        sample_batch['satellite'],
        sample_batch['seismic'],
        sample_batch['time']
    )
    
    # Проверка результатов
    assert isinstance(predictions, torch.Tensor)
    assert isinstance(probabilities, torch.Tensor)
    assert predictions.shape == (32, 1)
    assert probabilities.shape == (32, 1)
    assert torch.all((predictions == 0) | (predictions == 1))
    assert torch.all((probabilities >= 0) & (probabilities <= 1))

def test_model_encoders(model, sample_batch):
    """Тест энкодеров модели"""
    # Проверка спутникового энкодера
    satellite_encoded = model.satellite_encoder(sample_batch['satellite'])
    assert satellite_encoded.shape == (32, 24, 256)  # [batch_size, seq_len, hidden_size]
    
    # Проверка сейсмического энкодера
    seismic_encoded = model.seismic_encoder(sample_batch['seismic'])
    assert seismic_encoded.shape == (32, 24, 256)
    
    # Проверка временного энкодера
    time_encoded = model.time_encoder(sample_batch['time'])
    assert time_encoded.shape == (32, 24, 256)

def test_model_transformer(model, sample_batch):
    """Тест трансформера"""
    # Подготовка входных данных
    satellite_encoded = model.satellite_encoder(sample_batch['satellite'])
    seismic_encoded = model.seismic_encoder(sample_batch['seismic'])
    time_encoded = model.time_encoder(sample_batch['time'])
    
    # Объединение признаков
    combined = torch.cat([
        satellite_encoded,
        seismic_encoded,
        time_encoded
    ], dim=-1)
    
    # Применение трансформера
    transformer_out = model.transformer(combined)
    
    # Проверка результатов
    assert transformer_out.shape == (32, 24, 256)

def test_model_output_layer(model, sample_batch):
    """Тест выходного слоя"""
    # Подготовка входных данных
    satellite_encoded = model.satellite_encoder(sample_batch['satellite'])
    seismic_encoded = model.seismic_encoder(sample_batch['seismic'])
    time_encoded = model.time_encoder(sample_batch['time'])
    
    # Объединение признаков
    combined = torch.cat([
        satellite_encoded,
        seismic_encoded,
        time_encoded
    ], dim=-1)
    
    # Применение трансформера
    transformer_out = model.transformer(combined)
    
    # Взятие последнего временного шага
    last_step = transformer_out[:, -1, :]
    
    # Применение выходного слоя
    output = model.output_layer(last_step)
    
    # Проверка результатов
    assert output.shape == (32, 1)
    assert torch.all((output >= 0) & (output <= 1))

def test_model_different_batch_sizes(model):
    """Тест модели с разными размерами батча"""
    batch_sizes = [1, 16, 32, 64]
    seq_len = 24
    
    for batch_size in batch_sizes:
        # Создание тестового батча
        batch = {
            'satellite': torch.randn(batch_size, seq_len, 10),
            'seismic': torch.randn(batch_size, seq_len, 12),
            'time': torch.randn(batch_size, seq_len, 8)
        }
        
        # Прямой проход
        output = model(
            batch['satellite'],
            batch['seismic'],
            batch['time']
        )
        
        # Проверка результатов
        assert output.shape == (batch_size, 1)

def test_model_different_sequence_lengths(model):
    """Тест модели с разными длинами последовательности"""
    batch_size = 32
    seq_lengths = [12, 24, 48]
    
    for seq_len in seq_lengths:
        # Создание тестового батча
        batch = {
            'satellite': torch.randn(batch_size, seq_len, 10),
            'seismic': torch.randn(batch_size, seq_len, 12),
            'time': torch.randn(batch_size, seq_len, 8)
        }
        
        # Прямой проход
        output = model(
            batch['satellite'],
            batch['seismic'],
            batch['time']
        )
        
        # Проверка результатов
        assert output.shape == (batch_size, 1)

def test_model_gradient_flow(model, sample_batch):
    """Тест потока градиентов"""
    # Включение режима обучения
    model.train()
    
    # Прямой проход
    output = model(
        sample_batch['satellite'],
        sample_batch['seismic'],
        sample_batch['time']
    )
    
    # Расчет потерь
    criterion = torch.nn.BCELoss()
    loss = criterion(output, sample_batch['target'])
    
    # Обратное распространение
    loss.backward()
    
    # Проверка градиентов
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

def test_model_save_load(model, tmp_path):
    """Тест сохранения и загрузки модели"""
    # Сохранение модели
    save_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), save_path)
    
    # Создание новой модели
    new_model = TemporalFusionTransformer(model.config)
    
    # Загрузка весов
    new_model.load_state_dict(torch.load(save_path))
    
    # Проверка параметров
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)

def test_model_interpretation(config, sample_input_data):
    """Тест интерпретации предсказаний модели"""
    from src.models.tft_model import TemporalFusionTransformer
    
    model = TemporalFusionTransformer(config['model'])
    
    # Получаем предсказания и интерпретацию
    output, attention_weights = model(sample_input_data['features'], 
                                    sample_input_data['mask'],
                                    return_attention=True)
    
    # Проверка интерпретации
    interpretation = model.interpret_prediction(output, attention_weights)
    
    assert isinstance(interpretation, dict)
    assert 'feature_importance' in interpretation
    assert 'temporal_importance' in interpretation
    assert len(interpretation['feature_importance']) == sample_input_data['features'].shape[2]
    assert len(interpretation['temporal_importance']) == sample_input_data['features'].shape[1]

def test_model_training(config, sample_input_data):
    """Тест процесса обучения модели"""
    from src.models.tft_model import TemporalFusionTransformer
    from src.training.trainer import ModelTrainer
    
    # Создаем модель и тренер
    model = TemporalFusionTransformer(config['model'])
    trainer = ModelTrainer(model, config['training'])
    
    # Создаем тестовые данные
    train_data = {
        'features': sample_input_data['features'],
        'mask': sample_input_data['mask'],
        'target': torch.rand(sample_input_data['features'].shape[0], 1)
    }
    
    # Обучаем модель
    train_loss = trainer.train_epoch(train_data)
    
    assert isinstance(train_loss, float)
    assert train_loss >= 0

def test_model_validation(config, sample_input_data):
    """Тест валидации модели"""
    from src.models.tft_model import TemporalFusionTransformer
    from src.training.trainer import ModelTrainer
    
    # Создаем модель и тренер
    model = TemporalFusionTransformer(config['model'])
    trainer = ModelTrainer(model, config['training'])
    
    # Создаем тестовые данные
    val_data = {
        'features': sample_input_data['features'],
        'mask': sample_input_data['mask'],
        'target': torch.rand(sample_input_data['features'].shape[0], 1)
    }
    
    # Валидируем модель
    val_loss, metrics = trainer.validate_epoch(val_data)
    
    assert isinstance(val_loss, float)
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics 