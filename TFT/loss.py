import torch

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
            loss += ql.sum(dim=0) / (batch_size * T_max)

    return loss


def q_risk(y_true, y_pred, quantile, T_max):
    """
    Calculates the q-Risk metric according to equation 26:
    q-Risk = 2Σ(yt∈Ω̃)Σ(τ=1 to Tmax) QL(yt, ŷ(q,t-τ,τ), q) / Σ(yt∈Ω̃)Σ(τ=1 to Tmax)|yt|
    """
    # Calculate numerator (sum of quantile losses)
    numerator = 0
    for t in range(T_max):
        y_pred_t = y_pred[:, t, :]
        ql = quantile_loss(y_true[:, t], y_pred_t, quantile)
        numerator += ql.sum(dim=0)

    # Calculate denominator (sum of absolute true values)
    denominator = torch.abs(y_true).sum(dim=0)

    # Compute q-Risk (multiply by 2 as per equation)
    risk = 2 * numerator / denominator

    return risk
