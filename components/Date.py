from datetime import datetime
import numpy as np
from config.types import DateConfig

class Date:
    def __init__(self, date_config: DateConfig) -> None:
        self.config_ = date_config

        first_col = date_config.current_date.iloc[:, 0]
        self.current_date_ = np.asarray(first_col.values, dtype=datetime)

        # first_col_weekday = date_config.weekday.iloc[:, 0]
        # self.weekday_ = np.asarray(first_col_weekday.values, dtype=int)

    @property
    def config(self) -> DateConfig:
        return DateConfig.parse_obj(
            {
                "current_date": self.config_.current_date_,
                "weekday": self.weekday_,
            }
        )

    def get_date(self, t: int) -> datetime:
        return self.current_date_[t]

