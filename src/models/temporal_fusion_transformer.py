import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class TemporalFusionTransformer(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Параметры модели
        self.hidden_size = config['model']['hidden_size']
        self.num_heads = config['model']['num_heads']
        self.num_layers = config['model']['num_layers']
        self.dropout = config['model']['dropout']
        
        # Размерности входных данных
        self.satellite_features = config['satellite']['num_features']
        self.seismic_features = config['seismic']['num_features']
        self.time_features = config['time']['num_features']
        
        # Энкодеры для разных типов данных
        self.satellite_encoder = nn.Sequential(
            nn.Linear(self.satellite_features, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.seismic_encoder = nn.Sequential(
            nn.Linear(self.seismic_features, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        self.time_encoder = nn.Sequential(
            nn.Linear(self.time_features, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Transformer слои
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.dropout
            ),
            num_layers=self.num_layers
        )
        
        # Выходной слой
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                satellite_data: torch.Tensor,
                seismic_data: torch.Tensor,
                time_data: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели
        
        Args:
            satellite_data: Спутниковые данные [batch_size, seq_len, satellite_features]
            seismic_data: Сейсмические данные [batch_size, seq_len, seismic_features]
            time_data: Временные признаки [batch_size, seq_len, time_features]
            
        Returns:
            Предсказания [batch_size, 1]
        """
        # Кодирование входных данных
        satellite_encoded = self.satellite_encoder(satellite_data)
        seismic_encoded = self.seismic_encoder(seismic_data)
        time_encoded = self.time_encoder(time_data)
        
        # Объединение признаков
        combined = torch.cat([
            satellite_encoded,
            seismic_encoded,
            time_encoded
        ], dim=-1)
        
        # Применение Transformer
        transformer_out = self.transformer(combined)
        
        # Взятие последнего временного шага
        last_step = transformer_out[:, -1, :]
        
        # Получение предсказания
        prediction = self.output_layer(last_step)
        
        return prediction
    
    def predict(self,
                satellite_data: torch.Tensor,
                seismic_data: torch.Tensor,
                time_data: torch.Tensor,
                threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Получение предсказаний и вероятностей
        
        Args:
            satellite_data: Спутниковые данные
            seismic_data: Сейсмические данные
            time_data: Временные признаки
            threshold: Порог для бинарной классификации
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (предсказания, вероятности)
        """
        self.eval()
        with torch.no_grad():
            probabilities = self(satellite_data, seismic_data, time_data)
            predictions = (probabilities > threshold).float()
        return predictions, probabilities 