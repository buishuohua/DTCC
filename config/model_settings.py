from dataclasses import dataclass
from typing import Literal, List


@dataclass
class ModelSettings:
    num_heads: int
    quantiles: List[float]
    num_vars: int
    original_dim: int
    d_model: int
    past_len: int
    backcast_len: int
    forecast_len: int
    init_method: Literal['normal', 'xavier_uniform',
                         'xavier_normal', 'kaiming_uniform', 'kaiming_normal']
    dropout: float

    @classmethod
    def default(cls):
        return cls(
            num_heads=8,
            quantiles=[0.1, 0.5, 0.9],
            num_vars=10,
            original_dim=1,
            d_model=128,
            past_len=10,
            backcast_len=10,
            forecast_len=1,
            init_method='kaiming_normal',
            dropout=0.1
        )
