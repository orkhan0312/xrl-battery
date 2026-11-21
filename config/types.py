"""
Type helpers for the project
"""
from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel

MODEL_CONFIG = {"arbitrary_types_allowed": True}


class BatteryConfig(BaseModel):
    capacity: float
    high_capacity: float
    low_capacity: float
    max_output: float
    min_output: float
    initial_energy: Optional[float] = 0
    overcharge_penalty: float
    eta_c: Optional[float] = 1.0
    eta_d: Optional[float] = 1.0


class GridConfig(BaseModel):
    import_prices: pd.DataFrame
    export_prices: pd.DataFrame
    import_price_factor: Optional[float] = 1.0
    export_price_factor: Optional[float] = 0.75

    model_config = MODEL_CONFIG


class PvConfig(BaseModel):
    pv_production_ts: pd.DataFrame
    production_factor: Optional[float] = 1.0

    model_config = MODEL_CONFIG


class LoadConfig(BaseModel):
    load_ts: pd.DataFrame
    load_factor: Optional[float] = 1.0

    model_config = MODEL_CONFIG

class DateConfig(BaseModel):
    current_date: pd.DataFrame
    weekday: Optional[pd.DataFrame] = 1

    model_config = MODEL_CONFIG


class MicrogridConfig(BaseModel):
    max_timestep: int
    overprod_penalty: float
    underprod_penalty: float
    pv: PvConfig
    load: LoadConfig
    grid: GridConfig
    battery: BatteryConfig
    current_date: DateConfig
    weekday: DateConfig
