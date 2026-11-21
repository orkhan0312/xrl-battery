"""
'Realistic' config to be used for testing
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config.data_utils import prepare_data, get_row_range, get_row_numbers_from_dates
# from config.data_utils import DATA_FOLDER

from config.types import BatteryConfig, GridConfig, LoadConfig, MicrogridConfig, PvConfig, DateConfig


DURATION = 8
CAPACITY = 10 # 5
START_TIMESTEP = 10000 #10000
LEN_TIMESTEP = 15000 # 15000 | 34000
RESAMPLE_RULE = "15min"
DATA_FOLDER = os.path.dirname(__file__)



# data_file = os.path.join(DATA_FOLDER, "..", "data", "dataset.csv")
#
# df = pd.read_csv(data_file, parse_dates=["time"])
#
# df.drop(["active_power", "voltage"], inplace=True, axis=1)
# df["consumption"].fillna(method="ffill", inplace=True)
# df["generation"].fillna(method="ffill", inplace=True)
#
# pv_size = 4
# battery_capacity = 10000
# df['generation'] = df['generation'] * pv_size
#
#     # dataset['date'] = dataset.index.date
# df['time'] = pd.to_datetime(df['time'])
#
#     # Extract date from the 'time' column
# df['date'] = df['time'].dt.date
# daily_avg = df.groupby('date')['price'].transform('mean')
# df['avg_price'] = daily_avg

# df = prepare_data()
# print(len(df))





data_file = os.path.join(DATA_FOLDER, "..", "data", "Annual_load_PV_with_price.csv")
df = pd.read_csv(data_file, parse_dates=["time"])

df_raw = df.set_index("time").sort_index()

df = df_raw.resample(RESAMPLE_RULE).agg({
    "generation": "mean",
    "consumption": "mean",
    "price": "mean"
})

# start_row, end_row = get_row_numbers_from_dates(
#     df,
#     "2022-04-01 00:00:00",
#     "2022-04-05 00:00:00"
# )
# print("Start", start_row, "end", end_row)

consumption_df = df[["consumption"]].astype(float) if "consumption" in df.columns else None
generation_df = df[["generation"]].astype(float) if "generation" in df.columns else None
time_df = pd.DataFrame({"time": df.index})
price_df = df[["price"]].astype(float) if "price" in df.columns else None

pv_slice = generation_df[START_TIMESTEP:START_TIMESTEP + LEN_TIMESTEP]
load_slice = consumption_df[START_TIMESTEP:START_TIMESTEP + LEN_TIMESTEP]
price_slice = price_df[START_TIMESTEP:START_TIMESTEP + LEN_TIMESTEP] / 1000
time_slice = time_df[START_TIMESTEP:START_TIMESTEP + LEN_TIMESTEP]




battery_config = BatteryConfig.parse_obj(
    {
        "capacity": CAPACITY,
        "high_capacity": 0.9 * CAPACITY,
        "low_capacity": 0.1 * CAPACITY,
        "max_output": int(np.ceil(CAPACITY / DURATION)),
        "min_output": 0,
        "initial_energy": 0.5 * CAPACITY,
        "overcharge_penalty": 1,
    }
)

pv_config = PvConfig.parse_obj(
    {
        "pv_production_ts": pv_slice,
    }
)

grid_config = GridConfig.parse_obj(
    {
        "import_prices": price_slice,
        "export_prices": price_slice,
    }
)

load_config = LoadConfig.parse_obj(
    {
        "load_ts": load_slice,
    }
)

date_config = DateConfig.parse_obj(
    {
        "current_date": time_slice,
        # "weekday": weekday_slice,
    }
)

mg_config = MicrogridConfig.parse_obj(
    {
        "max_timestep": LEN_TIMESTEP,
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
