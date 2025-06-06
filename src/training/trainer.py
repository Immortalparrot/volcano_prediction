import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self, 
                 model: nn.Module,
                 config: Dict,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Оптимизатор и функция потерь
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        self.criterion = nn.BCELoss()
        
        # Настройка логирования
        self.setup_logging()
        
        # Настройка TensorBoard
        self.writer = SummaryWriter(
            log_dir=Path(config['paths']['tensorboard_logs'])
        )
        
        # Создание директорий
        self.checkpoint_dir = Path(config['paths']['checkpoints'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Настройка логирования"""
        log_dir = Path(self.config['paths']['logs'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, 
                   train_loader: DataLoader) -> Dict[str, float]:
        """Обучение на одной эпохе"""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch in train_loader:
            # Распаковка батча
            satellite_data = batch['satellite'].to(self.device)
            seismic_data = batch['seismic'].to(self.device)
            time_data = batch['time'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Прямой проход
            self.optimizer.zero_grad()
            output = self.model(satellite_data, seismic_data, time_data)
            loss = self.criterion(output, target)
            
            # Обратное распространение
            loss.backward()
            self.optimizer.step()
            
            # Сбор метрик
            total_loss += loss.item()
            predictions.extend(output.cpu().detach().numpy())
            targets.extend(target.cpu().numpy())
        
        # Расчет метрик
        metrics = self.calculate_metrics(
            np.array(predictions),
            np.array(targets)
        )
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    def validate(self, 
                val_loader: DataLoader) -> Dict[str, float]:
        """Валидация модели"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Распаковка батча
                satellite_data = batch['satellite'].to(self.device)
                seismic_data = batch['seismic'].to(self.device)
                time_data = batch['time'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Прямой проход
                output = self.model(satellite_data, seismic_data, time_data)
                loss = self.criterion(output, target)
                
                # Сбор метрик
                total_loss += loss.item()
                predictions.extend(output.cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        # Расчет метрик
        metrics = self.calculate_metrics(
            np.array(predictions),
            np.array(targets)
        )
        metrics['loss'] = total_loss / len(val_loader)
        
        return metrics
    
    def calculate_metrics(self,
                         predictions: np.ndarray,
                         targets: np.ndarray) -> Dict[str, float]:
        """Расчет метрик"""
        threshold = self.config['training']['threshold']
        pred_labels = (predictions > threshold).astype(int)
        
        # Расчет метрик
        tp = np.sum((pred_labels == 1) & (targets == 1))
        fp = np.sum((pred_labels == 1) & (targets == 0))
        fn = np.sum((pred_labels == 0) & (targets == 1))
        tn = np.sum((pred_labels == 0) & (targets == 0))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_checkpoint(self, 
                       epoch: int,
                       metrics: Dict[str, float],
                       is_best: bool = False):
        """Сохранение чекпоинта"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Сохранение последнего чекпоинта
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Сохранение лучшей модели
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            
        # Сохранение конфигурации
        config_path = self.checkpoint_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def load_checkpoint(self, 
                       checkpoint_path: str) -> int:
        """Загрузка чекпоинта"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch']
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: int,
             early_stopping_patience: int = 5):
        """Обучение модели"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Обучение
            train_metrics = self.train_epoch(train_loader)
            
            # Валидация
            val_metrics = self.validate(val_loader)
            
            # Логирование
            self.log_metrics(epoch, train_metrics, val_metrics)
            
            # Сохранение чекпоинта
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f'Early stopping at epoch {epoch}')
                break
    
    def log_metrics(self,
                   epoch: int,
                   train_metrics: Dict[str, float],
                   val_metrics: Dict[str, float]):
        """Логирование метрик"""
        # Логирование в TensorBoard
        for name, value in train_metrics.items():
            self.writer.add_scalar(f'train/{name}', value, epoch)
        
        for name, value in val_metrics.items():
            self.writer.add_scalar(f'val/{name}', value, epoch)
        
        # Логирование в файл
        self.logger.info(
            f'Epoch {epoch}: '
            f'Train Loss: {train_metrics["loss"]:.4f}, '
            f'Val Loss: {val_metrics["loss"]:.4f}, '
            f'Val F1: {val_metrics["f1"]:.4f}'
        )
    
    def plot_training_history(self, history: Dict, save_dir: Path):
        """Построение графиков процесса обучения"""
        # График потерь
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(save_dir / 'loss_history.png')
        plt.close()
        
        # График метрик
        plt.figure(figsize=(10, 6))
        metrics = pd.DataFrame(history['val_metrics'])
        sns.lineplot(data=metrics)
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Validation Metrics')
        plt.savefig(save_dir / 'metrics_history.png')
        plt.close()
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Оценка модели на тестовом наборе"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                all_predictions.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Расчет метрик
        metrics = {
            'f1': f1_score(targets, predictions > 0.5),
            'roc_auc': roc_auc_score(targets, predictions)
        }
        
        # Построение PR-кривой
        precision, recall, thresholds = precision_recall_curve(targets, predictions)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig('pr_curve.png')
        plt.close()
        
        return metrics 