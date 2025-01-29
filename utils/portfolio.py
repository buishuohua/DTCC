import dataclasses
from datetime import datetime

@dataclasses.dataclass
class Portfolio:
	index: str
	tickers: list[str]
	period: str
	interval: str

	@classmethod
	def default(cls) -> 'Portfolio':
		return cls(index='DIA', tickers=['AAPL', 'NVDA'], period='10y', interval='1d')
