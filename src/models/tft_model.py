import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class TemporalFusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.attention_dropout = config['attention_dropout']
        self.hidden_continuous_size = config['hidden_continuous_size']
        
        # Энкодер для временных рядов
        self.temporal_encoder = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Механизм внимания
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.attention_dropout
        )
        
        # Полносвязные слои
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_continuous_size)
        self.fc2 = nn.Linear(self.hidden_continuous_size, 1)
        
        # Нормализация
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_len, _ = x.size()
        
        # Энкодирование временных рядов
        temporal_features, _ = self.temporal_encoder(x)
        
        # Применение механизма внимания
        temporal_features = temporal_features.permute(1, 0, 2)  # (seq_len, batch_size, hidden_size)
        attn_output, attn_weights = self.attention(
            temporal_features, temporal_features, temporal_features
        )
        attn_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, hidden_size)
        
        # Нормализация и прореживание
        attn_output = self.layer_norm(attn_output)
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)
        
        # Получение финального представления
        final_features = attn_output[:, -1, :]  # Берем последний временной шаг
        
        # Полносвязные слои
        x = F.relu(self.fc1(final_features))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = torch.sigmoid(self.fc2(x))
        
        return x, attn_weights
    
    def interpret_prediction(self, prediction, attention_weights):
        """Интерпретация предсказания модели"""
        # Нормализация весов внимания
        attention_weights = attention_weights.mean(dim=0)  # Усредняем по головам
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Получаем важность временных интервалов
        temporal_importance = attention_weights[-1].cpu().numpy()
        
        # Получаем важность признаков
        feature_importance = {
            f"feature_{i}": float(importance)
            for i, importance in enumerate(temporal_importance)
        }
        
        return {
            'feature_importance': feature_importance,
            'temporal_importance': {
                f"t-{i}": float(importance)
                for i, importance in enumerate(temporal_importance)
            }
        }
    
    @classmethod
    def load(cls, path):
        """Загрузка модели из файла"""
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def save(self, path):
        """Сохранение модели в файл"""
        torch.save({
            'config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_heads': self.num_heads,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'attention_dropout': self.attention_dropout,
                'hidden_continuous_size': self.hidden_continuous_size
            },
            'state_dict': self.state_dict()
        }, path)

    def predict(self, x: torch.Tensor, 
               threshold: float = 0.5) -> Tuple[torch.Tensor, Dict]:
        """
        Получение прогноза и его интерпретации
        
        Args:
            x: Входные данные
            threshold: Порог для бинарной классификации
            
        Returns:
            predictions: Бинарные прогнозы
            interpretation: Интерпретация прогноза
        """
        self.eval()
        with torch.no_grad():
            probs, attention_dict = self(x)
            predictions = (probs > threshold).float()
            interpretation = self.interpret_prediction(x, attention_dict['attention_weights'])
            
        return predictions, interpretation 