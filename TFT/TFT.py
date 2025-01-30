import torch
import torch.nn as nn
from .modules.TFD import TemporalFusionDecoder
from .modules.VSN import VariableSelectionNetwork


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        num_vars,           # Number of input variables
		d_model,
        num_heads,          # Number of attention heads
        dropout=0.1,        # Dropout rate
        forecast_len=1,     # Number of steps to forecast
        backcast_len=10,    # Number of past steps to consider
        quantiles=[0.1, 0.5, 0.9]  # Quantiles to forecast
    ):
        super().__init__()

        self.d_model = d_model
        self.forecast_len = forecast_len
        self.backcast_len = backcast_len
        self.quantiles = quantiles

        # Temporal Fusion Decoder (includes LSTM encoder/decoder and attention mechanisms)
        self.temporal_decoder = TemporalFusionDecoder(
            d_model=d_model,
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


def quantile_loss(y_true, y_pred, q):
    """
    Calculates the quantile loss for a single quantile
    According to equation 25: QL(y, ŷ, q) = q(y - ŷ)₊ + (1-q)(ŷ - y)₊
    """
    error = y_true - y_pred
    return torch.max(q * error, (q - 1) * error)


def training_loss(y_true, y_pred, quantiles, T_max):
    """
    Calculates the training loss according to equation 24:
    L(Ω,W) = Σ(yt∈Ω) Σ(q∈Q) Σ(τ=1 to Tmax) QL(yt, ŷ(q,t-τ,τ), q) / MTmax
    """
    batch_size = y_true.size(0)
    loss = 0

    # Sum over all samples in batch (Σ(yt∈Ω))
    for t in range(T_max):
        # Sum over all quantiles (Σ(q∈Q))
        for i, q in enumerate(quantiles):
            # Get predictions for current quantile
            y_pred_q = y_pred[:, t, i]

            # Calculate quantile loss
            ql = quantile_loss(y_true[:, t], y_pred_q, q)

            # Sum over time steps and normalize
            loss += ql.sum() / (batch_size * T_max)

    return loss


def q_risk(y_true, y_pred, quantiles, T_max):
    """
    Calculates the q-Risk metric according to equation 26:
    q-Risk = 2Σ(yt∈Ω̃)Σ(τ=1 to Tmax) QL(yt, ŷ(q,t-τ,τ), q) / Σ(yt∈Ω̃)Σ(τ=1 to Tmax)|yt|
    """
    # Calculate numerator (sum of quantile losses)
    numerator = 0
    for t in range(T_max):
        for i, q in enumerate(quantiles):
            y_pred_q = y_pred[:, t, i]
            ql = quantile_loss(y_true[:, t], y_pred_q, q)
            numerator += ql.sum()

    # Calculate denominator (sum of absolute true values)
    denominator = torch.abs(y_true[:, :T_max]).sum()

    # Compute q-Risk (multiply by 2 as per equation)
    risk = 2 * numerator / denominator

    return risk


def train_step(model, optimizer, x_past, y_true, quantiles=[0.1, 0.5, 0.9]):
    """
    Single training step for the TFT model using the quantile loss
    """
    optimizer.zero_grad()

    # Forward pass
    y_pred = model(x_past)

    # Calculate training loss (equation 24)
    loss = training_loss(
        y_true=y_true,
        y_pred=y_pred,
        quantiles=quantiles,
        T_max=y_pred.size(1)  # forecast horizon
    )

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(model, x_past, y_true, quantiles=[0.1, 0.5, 0.9]):
    """
    Evaluate model using q-Risk metric
    """
    model.eval()
    with torch.no_grad():
        # Get predictions
        y_pred = model(x_past)

        # Calculate q-Risk (equation 26)
        risk = q_risk(
            y_true=y_true,
            y_pred=y_pred,
            quantiles=quantiles,
            T_max=y_pred.size(1)
        )

    return risk.item()
