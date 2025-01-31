from config.training_settings import TrainingSettings
from config.model_settings import ModelSettings
from TFT.TFT import TemporalFusionTransformer as TFT
from TFT.loss import training_loss, q_risk
import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from typing import Tuple, Dict


class Trainer:
    def __init__(self, model: nn.Module, training_settings: TrainingSettings = TrainingSettings.default(), model_settings: ModelSettings = ModelSettings.default()):
        """
        Initialize trainer
        Args:
            model: PyTorch model
            training_settings: Training configuration
        """
        self.model = model
        self.device = training_settings.device
        self.model.to(self.device)
        self.training_settings = training_settings
        self.model_settings = model_settings


    def train_step(self, x_past: torch.Tensor, x_future: torch.Tensor, y: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Single training step
        Args:
            x_past: Past input tensor [batch_size, past_len, num_factors]
            x_future: Future input tensor [batch_size, forecast_len, num_factors-1]
            y: Target tensor [batch_size, 1]
        Returns:
            loss: Training loss
            y_pred: Predicted values
        """
        self.model.train()
        # Forward pass
        y_pred = self.model(x_past, x_future)

        # Calculate loss
        loss = training_loss(
            y_true=y,
            y_pred=y_pred,
            quantiles=self.model_settings.quantiles,
            T_max=self.model_settings.forecast_len
        )

        return loss, y_pred

    def validate(self, val_loader) -> Tuple[float, Dict[str, float]]:
        """
        Validation step
        Args:
            val_loader: Validation data loader
        Returns:
            avg_loss: Average validation loss
            metrics: Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for x_past, x_future, y in val_loader:
                x_past = x_past.to(self.device)
                x_future = x_future.to(self.device)
                y = y.to(self.device)

                # Forward pass
                y_pred = self.model(x_past, x_future)

                # Calculate loss
                loss = torch.nn.MSELoss()(y_pred, y)
                total_loss += loss.item()

                predictions.extend(y_pred.cpu().numpy())
                targets.extend(y.cpu().numpy())

        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - targets))

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

        return total_loss / len(val_loader), metrics

    def train(self, train_loader, val_loader) -> Dict:
        """
        Complete training process using settings from TrainingSettings
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        Returns:
            history: Training history
        """
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_settings.learning_rate,
            weight_decay=self.training_settings.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            verbose=True
        )

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }

        best_val_loss = float('inf')

        for epoch in range(self.training_settings.epochs):
            # Training loop
            train_loss = 0
            self.model.train()

            with tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.training_settings.epochs}') as pbar:
                for x_past, x_future, y in pbar:
                    optimizer.zero_grad()

                    loss, _ = self.train_step(x_past, x_future, y)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    pbar.set_postfix({'train_loss': loss.item()})

            avg_train_loss = train_loss / len(train_loader)

            # Validation loop
            val_loss, val_metrics = self.validate(val_loader)

            # Update learning rate
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')

            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)

            # Print epoch results
            print(f'Epoch {epoch+1}/{self.training_settings.epochs}:')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print('Validation Metrics:')
            for metric, value in val_metrics.items():
                print(f'  {metric}: {value:.4f}')
            print('-' * 50)

        return history

    def test(self, test_loader) -> Tuple[float, Dict[str, float]]:
        """
        Test the model
        Args:
            test_loader: Test data loader
        Returns:
            test_loss: Test loss
            metrics: Test metrics
        """
        return self.validate(test_loader)

    def save_model(self, path):
        """
        Save model state
        """
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """
        Load model state
        """
        self.model.load_state_dict(torch.load(path))
