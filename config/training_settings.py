from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TrainingSettings:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    early_stopping_patience: int = 15
    model_save_path: str = 'models'
    data_dir: str = 'yf_data'
    save_checkpoints_dir: str = 'checkpoints'
    save_figs_dir: str = 'figs'
    save_metrics_dir: str = 'metrics'

    @classmethod
    def default(cls):
        return cls(
            device='cuda' if torch.cuda.is_available() else 'cpu',
            epochs=100,
            learning_rate=1e-4,
            weight_decay=1e-5,
            batch_size=32,
            early_stopping_patience=15,
            model_save_path='models',
            data_dir='data',
            save_checkpoints_dir='checkpoints',
            save_figs_dir='figs',
            save_metrics_dir='metrics'
        )
