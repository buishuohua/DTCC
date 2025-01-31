import torch.nn as nn
from .modules.TFD import TemporalFusionDecoder


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        num_vars,           # Number of input variables
        d_model,
        original_dim,
        num_heads,          # Number of attention heads
        dropout=0.1,        # Dropout rate
        forecast_len=1,     # Number of steps to forecast
        backcast_len=10,    # Number of past steps to consider
        quantiles=[0.1, 0.5, 0.9]  # Quantiles to forecast
    ):
        super().__init__()

        self.d_model = d_model
        self.original_dim = original_dim
        self.forecast_len = forecast_len
        self.backcast_len = backcast_len
        self.quantiles = quantiles

        # Temporal Fusion Decoder (includes LSTM encoder/decoder and attention mechanisms)
        self.temporal_decoder = TemporalFusionDecoder(
            d_model=d_model,
            original_dim=original_dim,
            num_vars=num_vars,
            num_heads=num_heads,
            forecast_len=forecast_len,
            quantiles=quantiles,
            dropout=dropout
        )

    def forward(self, x_past, x_future):
        """
        Args:
            x_past: Past inputs [batch_size, backcast_len, num_vars, input_size]
        Returns:
            forecasts: Predictions for each quantile [batch_size, forecast_len, num_quantiles]
        """

        forecasts = self.temporal_decoder(
            x_past=x_past,
            x_future=x_future,
            mask=None
        )

        return forecasts
