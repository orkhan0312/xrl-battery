from typing import Any

import numpy as np
from numpy import floating

from config.types import LoadConfig

class Load:
    def __init__(self, load_config: LoadConfig) -> None:
        self.config_ = load_config

        first_col = load_config.load_ts.iloc[:, 0]
        self.load_ts_ = np.asarray(first_col.values, dtype=float)
        self.load_factor = 1.0

    @property
    def config(self) -> LoadConfig:
        return LoadConfig.parse_obj(
            {
                "load_ts": self.config_.load_ts_,
                "load_factor": self.load_factor,
            }
        )

    @property
    def load_ts(self) -> np.ndarray:
        return self.load_ts_ * self.load_factor

    @property
    def __len__(self) -> int:
        return len(self.load_ts)

    @property
    def __mean__(self) -> float:
        return np.mean(self.load_ts)

    def get_load(self, t: int) -> float:
        return self.load_ts[t]