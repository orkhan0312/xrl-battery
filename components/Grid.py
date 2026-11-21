import numpy as np
from config.types import GridConfig

class Grid:
    """Simple grid model for imports (buy) and exports (sell)."""
    def __init__(self, grid_config: GridConfig) -> None:
        self.config_ = grid_config
        first_col = grid_config.import_prices.iloc[:, 0]
        self.import_prices_ = np.asarray(first_col.values, dtype=float)
        self.export_prices_ = self.import_prices_
        self.import_price_factor = grid_config.import_price_factor
        self.export_price_factor = grid_config.export_price_factor

        if len(self.import_prices) != len(self.export_prices):
            raise ValueError(
                f"Price timeseries are not of the same length \
                    \n import : {len(self.import_prices)} \
                    \n export : {len(self.export_prices)} "
            )

    @property
    def config(self) -> GridConfig:
        return GridConfig.parse_obj(
            {
                "import_prices": self.config_.import_prices_,
                "export_prices": self.config_.export_prices_,
                "import_price_factor": self.import_price_factor,
                "export_price_factor": self.export_price_factor,
            }
        )

    def get_cost(self, t: int, energy: float) -> float:
        if energy >= 0:
            cost = self.export_prices[t] * energy
        else:
            cost = self.import_prices[t] * energy
        return cost

    @property
    def __len__(self) -> int:
        return len(self.import_prices)

    @property
    def import_prices(self) -> np.ndarray:
        return self.import_prices_ * self.import_price_factor

    @property
    def export_prices(self) -> np.ndarray:
        return self.export_prices_ * self.export_price_factor

    def get_price(self, t: int) -> float:
        return self.import_prices[t]
