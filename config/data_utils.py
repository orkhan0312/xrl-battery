"""
External data handling helpers
"""
import json
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

# DATA_FOLDER = os.path.dirname(__file__)


# def load_data(path: Path) -> np.ndarray:
#     """
#     Read the data based on the file path
#
#     Args:
#         file (path to the correct file): _description_
#
#     Raises:
#         ValueError: _description_
#
#     Returns:
#         list: _description_
#     """
#     if not str(path).endswith(".csv"):
#         raise ValueError(f"The given path to file is not a csv file : {path}")
#     data = pd.read_csv(path)
#     assert len(data.columns) == 1
#     return np.array(data.values.flatten())


# def get_indexes(data_folder) -> dict:
#     """
#     Get the indexes for all data folders that allow the user to see what data \
#     is available
#
#     Args:
#         data_folder (str): the folder where the package data is stored
#
#     Returns:
#         dict: A dict with the different data types (pv, load, co2) as keys \
#             and available data in list as values
#     """
#     dirs = [
#         os.path.join(data_folder, d)
#         for d in os.listdir(data_folder)
#         if os.path.isdir(os.path.join(data_folder, d))
#     ]
#     indexes = {}
#     for directory in dirs:
#         indexes[directory.split("/")[-1]] = get_index(directory)
#     return indexes
#
#
# def get_index(data_folder) -> list:
#     """
#     Create the index that lists the different files available
#     Returns:
#         list: _description_
#     """
#     return [
#         os.path.join(data_folder, f)
#         for f in os.listdir(data_folder)
#         if f.endswith(".csv")
#     ]

# def get_weekday(path: Path) -> np.ndarray:
#     if not str(path).endswith(".csv"):
#         raise ValueError(f"The given path to file is not a csv file : {path}")
#     data = pd.read_csv(path)
#     assert len(data.columns) == 1
#     return data.weekday()
#
#
#
# def load_date(path: Union[str, Path]) -> np.ndarray:
#     """
#     Read a CSV containing a single datetime column and return numpy array of datetimes.
#
#     Accepts:
#       - a CSV with one column of ISO-like datetimes (with or without header)
#       - a CSV with columns ['year','month','day'] (pandas will parse if present)
#
#     Returns:
#       np.ndarray of numpy.datetime64 values (flattened 1D)
#     """
#     path = Path(path)
#     if not str(path).endswith(".csv"):
#         raise ValueError(f"The given path to file is not a csv file : {path}")
#
#     # Read without parsing first so we can inspect columns
#     df = pd.read_csv(path)
#
#     # If file contains one column, convert that column
#     if df.shape[1] == 1:
#         # pick the single Series and convert to datetime (handles strings/numbers)
#         series = df.iloc[:, 0]
#         dt = pd.to_datetime(series, errors="raise", infer_datetime_format=True)
#         return np.array(dt.values.flatten())
#
#     # If it has explicit year/month/day columns, let pandas assemble them
#     lower_cols = [c.lower() for c in df.columns]
#     if set(("year", "month", "day")).issubset(lower_cols):
#         # rename columns to lowercase for safe access
#         df_lower = df.copy()
#         df_lower.columns = lower_cols
#         dt = pd.to_datetime(df_lower[["year", "month", "day"]])
#         return np.array(dt.values.flatten())
#
#     # If multiple columns but one likely contains datetimes (e.g., 'date' or 'timestamp'), try to find it:
#     for candidate in ("date", "datetime", "timestamp", "time"):
#         for col in df.columns:
#             if candidate in col.lower():
#                 dt = pd.to_datetime(df[col], errors="raise", infer_datetime_format=True)
#                 return np.array(dt.values.flatten())
#
#     # Last resort: raise with helpful message
#     raise ValueError(
#         f"CSV at {path} must contain a single datetime column or year/month/day columns. "
#         f"Found columns: {list(df.columns)}"
#     )

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def prepare_data(
    price_xlsx: str = "./data/market_price_202122.xlsx",
    price_sheet: str = "Sheet1",
    solar_json: str = "./data/solar_panel_data2022.json",
    load_xlsx: str = "./data/load.xlsx",
    pv_size: float = 7.5,
    demand_max: float = 4.85,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load price, PV and load data, apply the same processing as your original code,
    and return a DataFrame indexed at 15-minute resolution with columns:
      - generation  (kW)  -> scaled PV profile
      - consumption (kW)  -> load with preserved hourly avg randomness
      - price        (tariff)

    Behaviour preserved from original:
    - Price file: keep rows for 2022, compute hourly mean then forward-fill to 5-min resolution.
    - Solar JSON: list of [timestamp_ms, power], convert to datetime, make 5-min freq (asfreq), fill NaNs with 0,
      limit to 2022-01-01 through 2022-12-30, normalize PV to 0..1 then scale by pv_size.
    - Load: use 'Periood' column, hourly mean then forward-fill to 5-min, then shift index by -5 hours (same as original),
      fill remaining NaNs forward.
    - Add randomness per-hour preserving the hourly average (same method), then replace demand with randomized version.
    - Final result is obtained by taking the first value in each 15-minute block (same as original resample('15min').first()).
    """
    # optional reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # ---- PRICES ----
    price_path = Path(price_xlsx)
    price = pd.read_excel(price_path, sheet_name=price_sheet)
    # ensure date/time columns are string-like, then combine
    price["date"] = price["date"].astype(str)
    price["time"] = price["time"].astype(str)
    price["datetime"] = pd.to_datetime(price["date"] + " " + price["time"])
    price = price.set_index("datetime")
    price = price.drop(columns=["date", "time"])
    # restrict to 2022 and follow original: hourly mean then upsample to 5min by forward-fill
    price_2022 = price.loc["2022"]
    price_hourly = price_2022.resample("H").mean()
    price_5min = price_hourly.resample("5T").ffill()  # '5T' == 5 minutes
    # price_5min.to_csv("price_5min_2022.csv")  # optional save (kept commented, as original)

    # ---- SOLAR / PV ----
    with open(solar_json, "r") as f:
        data = json.load(f)

    # data expected as list of [timestamp_ms, power]
    # unpack efficiently
    timestamps_ms = [entry[0] for entry in data]
    powers = [entry[1] for entry in data]

    # convert ms -> seconds and to datetime index
    timestamps_s = [t / 1000 for t in timestamps_ms]
    dt_index = pd.to_datetime(timestamps_s, unit="s")

    pv_df = pd.DataFrame({"pv": powers}, index=dt_index)
    # convert to regular 5-minute index (asfreq keeps original values where present; NaN otherwise)
    pv_df = pv_df.resample("5T").asfreq()
    pv_df = pv_df.fillna(0)
    # limit to 2022-01-01 through 2022-12-30 as original
    pv_df = pv_df.loc["2022-01-01":"2022-12-30"]

    # normalize (0..1) then scale by pv_size
    if pv_df["pv"].max() > 0:
        pv_df["pv"] = pv_df["pv"] / pv_df["pv"].max()
    else:
        pv_df["pv"] = 0.0
    pv_df["pv"] = pv_df["pv"] * pv_size

    # ---- LOAD ----
    load = pd.read_excel(load_xlsx)
    # original code dropped 'hour' column; keep same behaviour if present
    if "hour" in load.columns:
        load = load.drop(columns=["hour"])
    # 'Periood' -> datetime index
    load["Periood"] = pd.to_datetime(load["Periood"])
    load = load.set_index("Periood")
    load_hourly = load.resample("H").mean()
    load_5min = load_hourly.resample("5T").ffill()

    # original code did: load_5min = load_res.resample('5min').ffill()
    # then: load_5min = load_5min.shift(periods=-5, freq='H')
    # reproduce exactly (this shifts the index by -5 hours)
    load_5min = load_5min.shift(periods=-5, freq="H")
    load_5min.fillna(method="ffill", inplace=True)

    # make sure the load has a 'demand' column (original code uses load.Periood and .demand)
    if "demand" not in load_5min.columns:
        # try common alternatives or raise informative error
        possible = [c for c in load_5min.columns if "demand" in c.lower()]
        if possible:
            load_5min = load_5min.rename(columns={possible[0]: "demand"})
        else:
            raise KeyError("load data must contain a 'demand' column (or similar).")

    # scale demand similar to original code: normalize then multiply by demand_max
    if load_5min["demand"].max() > 0:
        load_5min["demand"] = load_5min["demand"] / load_5min["demand"].max()
    else:
        load_5min["demand"] = 0.0
    load_5min["demand"] = load_5min["demand"] * demand_max

    # ---- COMBINE into single 5-min DataFrame ----
    # start from pv_df index (which is 5-minute) and align others to it via reindex/ffill as needed
    # build a combined frame to exactly mirror original assignment:
    df = pd.DataFrame(index=pv_df.index)
    # attach pv
    df["pv"] = pv_df["pv"]
    # attach tariff (price) aligned to pv 5-min index
    # price_5min index may have full 2022 coverage; reindex to our pv_idx and forward-fill if necessary
    df["tariff"] = price_5min.reindex(df.index).ffill()["tariff"]
    # attach demand aligned to this index
    df["demand"] = load_5min.reindex(df.index).ffill()["demand"]

    # ---- RANDOMNESS: keep same hourly avg ----
    def add_randomness_with_same_hourly_avg(group: pd.Series) -> pd.Series:
        """Preserves the hourly mean while adding small random fluctuations (as original)."""
        mean_val = group.mean()
        random_fluctuations = np.random.rand(len(group)) - 0.5
        random_fluctuations -= random_fluctuations.mean()
        fluctuation_scale = 0.3
        group = group + random_fluctuations * fluctuation_scale
        adjustment = mean_val - group.mean()
        group = group + adjustment
        return group

    df["demand_rand"] = df.groupby(df.index.hour)["demand"].transform(
        add_randomness_with_same_hourly_avg
    )
    df["demand"] = df["demand_rand"]
    df = df.drop(columns=["demand_rand"])

    # ---- FINAL RESAMPLE to 15 minutes (same as original: .resample('15min').first()) ----
    df_15 = df.resample("15T").first()
    # keep only required columns and rename them as requested
    result = df_15[["pv", "demand", "tariff"]].rename(
        columns={"pv": "generation", "demand": "consumption", "tariff": "price"}
    )

    # ensure the index name is DatetimeIndex and sorted
    result = result.sort_index()
    result.index.name = None

    return result


from typing import Tuple, Union

def get_row_range(
    df: pd.DataFrame,
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
) -> Tuple[int, int]:
    """
    Given a dataframe with DatetimeIndex (15-min resolution),
    return the integer row positions corresponding to start_date and end_date.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a DatetimeIndex (sorted).
    start_date : str or pd.Timestamp
    end_date   : str or pd.Timestamp

    Returns
    -------
    (start_idx, end_idx) : tuple[int, int]
        Integer row positions in df.
    """

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DatetimeIndex.")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    if start_date > end_date:
        raise ValueError("start_date must be earlier than or equal to end_date.")

    # Find closest valid positions inside index
    start_pos = df.index.searchsorted(start_date, side="left")
    end_pos = df.index.searchsorted(end_date, side="right") - 1

    if start_pos >= len(df) or end_pos < 0:
        raise ValueError("Requested date range is outside dataframe index range.")

    return int(start_pos), int(end_pos)

def get_row_numbers_from_dates(df, start_date, end_date):
    """
    Returns (start_row, end_row) inclusive,
    based on datetime index.
    """

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Get integer positions
    start_row = df.index.get_loc(start_date)
    end_row = df.index.get_loc(end_date)

    return start_row, end_row