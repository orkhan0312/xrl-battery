import numpy as np
from config.types import PvConfig

class PV:
    def __init__(self, pv_config: PvConfig) -> None:
        self.config_ = pv_config
        first_col = pv_config.pv_production_ts.iloc[:, 0]
        self.pv_production_ts_ = np.asarray(first_col.values, dtype=float)
        self.production_factor = pv_config.production_factor

    @property
    def config(self) -> PvConfig:
        return PvConfig.parse_obj(
            {
                "pv_production_ts": self.config_.pv_production_ts,
                "production_factor": self.production_factor,
            }
        )

    @property
    def pv_production_ts(self) -> np.ndarray:
        return self.pv_production_ts_ * self.production_factor

    @property
    def __len__(self) -> int:
        return len(self.pv_production_ts)

    @property
    def __mean__(self) -> int:
        return np.mean(self.pv_production_ts)

    def get_power(self, t: int) -> float:
        return self.pv_production_ts[t]

