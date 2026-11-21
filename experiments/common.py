"""
Shared helpers for the camera-ready experiment suite: builds a disjoint
train/held-out-eval split from the full-year dataset (the original Config.py
only ever used one contiguous slice for both training and evaluation), and
provides a small evaluation routine reused by every training script.
"""
import os
import numpy as np
import pandas as pd

from config.types import BatteryConfig, GridConfig, LoadConfig, MicrogridConfig, PvConfig, DateConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(REPO_ROOT, "data", "Annual_load_PV_with_price.csv")
RESAMPLE_RULE = "15min"

DURATION = 8
CAPACITY = 10
# Charge/discharge efficiency (Eq. 5 of the paper). Round-trip ~90%.
ETA_C = 0.95
ETA_D = 0.95

# Same starting point/length as the original config.py, giving the same
# ~156-day (2023-04-15 to 2023-09-18) window used throughout the original repo.
START_TIMESTEP = 10000
LEN_TIMESTEP = 15000

# Held-out fraction taken from the *end* of that window, used only for
# evaluation and never seen during training (the original repo trained and
# evaluated on the same continuous trajectory -- this fixes that).
TRAIN_FRACTION = 0.80


def _load_full_df():
    df = pd.read_csv(DATA_FILE, parse_dates=["time"])
    df = df.set_index("time").sort_index()
    df = df.resample(RESAMPLE_RULE).agg({"generation": "mean", "consumption": "mean", "price": "mean"})
    return df


def _slice_to_config(df_slice: pd.DataFrame) -> MicrogridConfig:
    consumption_df = df_slice[["consumption"]].astype(float)
    generation_df = df_slice[["generation"]].astype(float)
    price_df = df_slice[["price"]].astype(float) / 1000
    time_df = pd.DataFrame({"time": df_slice.index}).reset_index(drop=True)

    battery_config = BatteryConfig.parse_obj(
        {
            "capacity": CAPACITY,
            "high_capacity": 0.9 * CAPACITY,
            "low_capacity": 0.1 * CAPACITY,
            "max_output": int(np.ceil(CAPACITY / DURATION)),
            "min_output": 0,
            "initial_energy": 0.5 * CAPACITY,
            "overcharge_penalty": 1,
            "eta_c": ETA_C,
            "eta_d": ETA_D,
        }
    )
    pv_config = PvConfig.parse_obj({"pv_production_ts": generation_df.reset_index(drop=True)})
    grid_config = GridConfig.parse_obj(
        {"import_prices": price_df.reset_index(drop=True), "export_prices": price_df.reset_index(drop=True)}
    )
    load_config = LoadConfig.parse_obj({"load_ts": consumption_df.reset_index(drop=True)})
    date_config = DateConfig.parse_obj({"current_date": time_df})

    return MicrogridConfig.parse_obj(
        {
            "max_timestep": len(df_slice),
            "overprod_penalty": 1,
            "underprod_penalty": 1,
            "pv": pv_config,
            "load": load_config,
            "battery": battery_config,
            "grid": grid_config,
            "current_date": date_config,
            "weekday": date_config,
        }
    )


def build_train_eval_configs():
    """Disjoint, time-ordered train/eval split from the same window the
    original repo used, so results stay comparable while eval is now genuinely
    held out."""
    df = _load_full_df()
    full_slice = df.iloc[START_TIMESTEP:START_TIMESTEP + LEN_TIMESTEP]
    n_train = int(len(full_slice) * TRAIN_FRACTION)

    train_df = full_slice.iloc[:n_train].copy()
    eval_df = full_slice.iloc[n_train:].copy()

    return _slice_to_config(train_df), _slice_to_config(eval_df), train_df, eval_df


def eval_windows(eval_df: pd.DataFrame, n_windows: int = 4, days: int = 7):
    """Non-overlapping `days`-long windows spread across the held-out period,
    returned as (start, end) pandas Timestamps."""
    start0 = eval_df.index[0]
    end0 = eval_df.index[-1]
    total_days = (end0 - start0).days
    span = pd.Timedelta(days=days)
    usable_days = total_days - days
    if usable_days <= 0 or n_windows <= 1:
        return [(start0, start0 + span)]
    step = usable_days / (n_windows - 1)
    windows = []
    for i in range(n_windows):
        s = start0 + pd.Timedelta(days=i * step)
        windows.append((s, s + span))
    return windows
