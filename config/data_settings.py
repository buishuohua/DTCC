from dataclasses import dataclass
from typing import List
from utils.data_related.portfolio import Portfolio


@dataclass
class DataSettings:
    data_dir: str
    factors: List[str]
    portfolio_data: Portfolio
    val_ratio: float
    test_ratio: float

    @classmethod
    def default(cls):
        return cls(
            data_dir="data",
            factors=["Open", "High", "Low", "Close", "Volume"],
            portfolio_data=Portfolio.default(),
            val_ratio=0.1,
            test_ratio=0.1
        )
