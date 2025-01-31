import torch
from torch import nn
import torch.nn.functional as F
from config.data_settings import DataSettings
from config.model_settings import ModelSettings
from config.training_settings import TrainingSettings
from utils.data_related.data_loader import load_data
from train.trainer import Trainer
from TFT.TFT import TemporalFusionTransformer


def main():
    # Get settings
    data_settings = DataSettings.default()
    model_settings = ModelSettings.default()
    training_settings = TrainingSettings.default()

    # Load data
    train_loader, val_loader, test_loader = load_data(
        data_settings,
        model_settings,
        training_settings
    )

    # Initialize model
    model = TemporalFusionTransformer(
        num_vars=len(data_settings.factors),
        d_model=model_settings.d_model,
        original_dim=model_settings.original_dim,
        num_heads=model_settings.num_heads,
        dropout=model_settings.dropout,
        forecast_len=model_settings.forecast_len,
        backcast_len=model_settings.backcast_len,
        quantiles=model_settings.quantiles
    )

    # Initialize trainer with settings
    trainer = Trainer(model, training_settings, model_settings)

    # Train model using settings
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Test model
    test_loss, test_metrics = trainer.test(test_loader)
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
