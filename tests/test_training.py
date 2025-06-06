import pytest
import torch
import numpy as np
from pathlib import Path
from src.training.trainer import ModelTrainer
from src.models.temporal_fusion_transformer import TemporalFusionTransformer
import pandas as pd

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
        },
        'training': {
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'batch_size': 32,
            'num_epochs': 2,
            'early_stopping_patience': 2,
            'threshold': 0.5
        },
        'paths': {
            'checkpoints': 'checkpoints',
            'logs': 'logs',
            'tensorboard_logs': 'logs/tensorboard',
            'results': 'results'
        }
    }

@pytest.fixture
def sample_data():
    """Фикстура с тестовыми данными"""
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

@pytest.fixture
def trainer(model, model_config):
    """Фикстура с тренером"""
    return ModelTrainer(model, model_config)

def test_trainer_initialization(trainer, model_config):
    """Тест инициализации тренера"""
    assert trainer.config == model_config
    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert isinstance(trainer.criterion, torch.nn.BCELoss)
    assert trainer.device in ['cuda', 'cpu']

def test_train_epoch(trainer, sample_data):
    """Тест обучения на одной эпохе"""
    # Создание DataLoader
    train_loader = torch.utils.data.DataLoader(
        [sample_data],
        batch_size=1,
        shuffle=True
    )
    
    # Обучение на одной эпохе
    metrics = trainer.train_epoch(train_loader)
    
    # Проверка результатов
    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert all(0 <= v <= 1 for v in metrics.values())

def test_validate(trainer, sample_data):
    """Тест валидации"""
    # Создание DataLoader
    val_loader = torch.utils.data.DataLoader(
        [sample_data],
        batch_size=1,
        shuffle=False
    )
    
    # Валидация
    metrics = trainer.validate(val_loader)
    
    # Проверка результатов
    assert isinstance(metrics, dict)
    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert all(0 <= v <= 1 for v in metrics.values())

def test_calculate_metrics(trainer):
    """Тест расчета метрик"""
    # Создание тестовых данных
    predictions = np.array([0.1, 0.7, 0.3, 0.9, 0.2])
    targets = np.array([0, 1, 0, 1, 0])
    
    # Расчет метрик
    metrics = trainer.calculate_metrics(predictions, targets)
    
    # Проверка результатов
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert all(0 <= v <= 1 for v in metrics.values())

def test_save_load_checkpoint(trainer, tmp_path):
    """Тест сохранения и загрузки чекпоинта"""
    # Сохранение чекпоинта
    epoch = 1
    metrics = {'loss': 0.5, 'accuracy': 0.8}
    checkpoint_path = tmp_path / "checkpoint.pt"
    
    trainer.save_checkpoint(epoch, metrics)
    
    # Проверка существования файла
    assert Path(trainer.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt').exists()
    
    # Загрузка чекпоинта
    loaded_epoch = trainer.load_checkpoint(
        str(trainer.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
    )
    
    # Проверка результатов
    assert loaded_epoch == epoch

def test_early_stopping(trainer, sample_data):
    """Тест раннего останова"""
    # Создание DataLoader
    train_loader = torch.utils.data.DataLoader(
        [sample_data],
        batch_size=1,
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        [sample_data],
        batch_size=1,
        shuffle=False
    )
    
    # Обучение с ранним остановом
    trainer.train(
        train_loader,
        val_loader,
        num_epochs=5,
        early_stopping_patience=2
    )
    
    # Проверка, что обучение остановилось раньше
    assert len(trainer.writer.log_dir.glob('events.out.tfevents.*')) > 0

def test_log_metrics(trainer):
    """Тест логирования метрик"""
    # Создание тестовых метрик
    epoch = 1
    train_metrics = {
        'loss': 0.5,
        'accuracy': 0.8,
        'precision': 0.7,
        'recall': 0.6,
        'f1': 0.65
    }
    val_metrics = {
        'loss': 0.4,
        'accuracy': 0.85,
        'precision': 0.75,
        'recall': 0.7,
        'f1': 0.72
    }
    
    # Логирование метрик
    trainer.log_metrics(epoch, train_metrics, val_metrics)
    
    # Проверка, что метрики записаны в TensorBoard
    assert len(trainer.writer.log_dir.glob('events.out.tfevents.*')) > 0

def test_plot_training_history(trainer, tmp_path):
    """Тест построения графиков"""
    # Создание тестовой истории
    history = {
        'train_loss': [0.5, 0.4, 0.3],
        'val_loss': [0.45, 0.35, 0.25],
        'val_metrics': pd.DataFrame({
            'accuracy': [0.8, 0.85, 0.9],
            'precision': [0.7, 0.75, 0.8],
            'recall': [0.6, 0.65, 0.7],
            'f1': [0.65, 0.7, 0.75]
        })
    }
    
    # Построение графиков
    save_dir = tmp_path / "plots"
    save_dir.mkdir()
    trainer.plot_training_history(history, save_dir)
    
    # Проверка существования файлов
    assert (save_dir / 'loss_history.png').exists()
    assert (save_dir / 'metrics_history.png').exists()

def test_evaluate(trainer, sample_data):
    """Тест оценки модели"""
    # Создание DataLoader
    test_loader = torch.utils.data.DataLoader(
        [sample_data],
        batch_size=1,
        shuffle=False
    )
    
    # Оценка модели
    metrics = trainer.evaluate(test_loader)
    
    # Проверка результатов
    assert isinstance(metrics, dict)
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    assert all(0 <= v <= 1 for v in metrics.values())
    assert Path('pr_curve.png').exists() 