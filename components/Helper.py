from typing import Union, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from components.Grid import Grid
from components.Load import LoadProfile
from components.MicrogridEnv import MicrogridEnv
from components.PV import PVProfile
from components.Storage import BatteryPybamm


def run_rl_episode(env,
                   policy: Union[str, object] = "random",
                   deterministic: bool = True,
                   max_steps: Optional[int] = None,
                   start_date: Optional[int] = None,
                   end_date: Optional[int] = None):
    """
    Run single episode and return validation DataFrame.

    New arguments:
    --------------
    start_date : str or pd.Timestamp
        Example: "2023-06-01"
    end_date : str or pd.Timestamp
        Example: "2023-06-03"

    If provided, a temporary validation environment will be created
    using only that date slice.
    """

    # -------------------------------------------------
    # 1️⃣  If date range provided → build validation env
    # -------------------------------------------------
    if start_date is not None or end_date is not None:

        if not hasattr(env.load_profile.series, "index"):
            raise ValueError("Validation slicing requires time-indexed data.")

        full_index = env.load_profile.series.index
        print(full_index)

        # start_date = pd.to_datetime(start_date) if start_date is not None else full_index.min()
        # end_date = pd.to_datetime(end_date) if end_date is not None else full_index.max()

        mask = (full_index >= start_date) & (full_index <= end_date)

        if mask.sum() == 0:
            raise ValueError("Selected validation range contains no data.")

        # Slice all time series
        load_slice = env.load_profile.series.loc[mask]
        pv_slice = env.pv_profile.series.loc[mask]
        price_slice = env.grid.buy_price.loc[mask] if hasattr(env.grid.buy_price, "loc") else env.grid.buy_price

        # Rebuild components for validation
        timestep_h = env.load_profile.dt_h

        load_profile = LoadProfile(load_slice, timestep_h)
        pv_profile = PVProfile(pv_slice, env.pv_profile.capacity_kw, timestep_h)
        grid = Grid(buy_price=price_slice,
                    sell_price=price_slice * 0.8,
                    timestep_hours=timestep_h)

        # Fresh battery for fair validation
        battery = BatteryPybamm(
            nominal_capacity_kwh=env.battery.nominal_capacity_kwh,
            max_charge_kw=env.battery.max_charge_kw,
            max_discharge_kw=env.battery.max_discharge_kw,
            efficiency=env.battery.efficiency,
            dt_hours=timestep_h
        )

        env = MicrogridEnv(load_profile, pv_profile, grid, battery,
                           price_series=price_slice,
                           timestep_hours=timestep_h)

        print(f"Validation range: {start_date} → {end_date} "
              f"({len(load_slice)} steps)")

    model = None
    if isinstance(policy, str) and policy not in ("random", "heuristic"):

        model = PPO.load(policy)


    elif hasattr(policy, "predict") and not isinstance(policy, str):
        model = policy

    def heuristic_action_fn(obs):
        # assume obs: [load_kW, pv_kW, buy_price, soc, capacity_kwh]
        load_kW, pv_kW, buy_price, soc, capacity = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3]), float(obs[4])
        # if PV surplus: charge up to available fraction
        if pv_kW > load_kW:
            # fraction of max_charge required to absorb PV surplus
            frac = (pv_kW - load_kW) / max(1e-6, env.battery.max_charge_kw)
            return np.array([float(min(1.0, frac))], dtype=np.float32)
        else:
            # discharge when price high or load > pv
            try:
                # try to find a reasonable threshold from env.grid (if series)
                if hasattr(env.grid, "buy_price") and not isinstance(env.grid.buy_price, (int, float)):
                    price_thresh = float(np.nanmedian(env.grid.buy_price))
                else:
                    price_thresh = 0.18
            except Exception:
                price_thresh = 0.18
            if buy_price > price_thresh and soc > 0.1:
                return np.array([-1.0], dtype=np.float32)
            elif (load_kW - pv_kW) > 1e-3 and soc > 0.05:
                frac = (load_kW - pv_kW) / max(1e-6, env.battery.max_discharge_kw)
                return np.array([ -float(min(1.0, frac)) ], dtype=np.float32)
            else:
                return np.array([0.0], dtype=np.float32)

    obs = env.reset()
    done = False
    steps = 0

    records = {
        "time_h": [],
        "battery_soc": [],
        "battery_capacity_kwh": [],
        "battery_energy_charged_kwh": [],
        "battery_energy_discharged_kwh": [],
        "grid_import_kwh": [],
        "grid_export_kwh": [],
        "price_eur_per_kwh": []
    }

    while not done:
        # Select action
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            if policy == "random":
                action = env.action_space.sample()
            elif policy == "heuristic":
                action = heuristic_action_fn(obs)
            elif callable(policy):
                action = policy(obs)
            else:
                raise ValueError("Unknown policy argument")

        next_obs, reward, done, info = env.step(action)

        # t index - use env.current_step-1 because env increments after step
        t_index = max(0, getattr(env, "current_step", steps) - 1)

        records["time_h"].append(t_index)
        # Info keys in MicrogridEnv from blueprint: batt_soc, batt_capacity_kwh, batt_energy_charged_kwh, batt_energy_discharged_kwh, grid_import_kwh, pv_export_kwh
        records["battery_soc"].append(info.get("batt_soc", env.battery.get_soc() if hasattr(env, "battery") else np.nan))
        records["battery_capacity_kwh"].append(info.get("batt_capacity_kwh", env.battery.get_capacity_kwh() if hasattr(env, "battery") else np.nan))
        records["battery_energy_charged_kwh"].append(info.get("batt_energy_charged_kwh", 0.0))
        records["battery_energy_discharged_kwh"].append(info.get("batt_energy_discharged_kwh", 0.0))
        grid_imp = info.get("grid_import_kwh", info.get("grid_import", 0.0))
        pv_exp = info.get("pv_export_kwh", info.get("grid_export_kwh", 0.0))
        # ensure export is positive in dataframe
        grid_export_positive = pv_exp if pv_exp >= 0 else -pv_exp
        records["grid_import_kwh"].append(grid_imp)
        records["grid_export_kwh"].append(grid_export_positive)

        # price: try get from env.grid
        try:
            buy_price, _ = env.grid.price_at(t_index)
        except Exception:
            buy_price = np.nan
        records["price_eur_per_kwh"].append(buy_price)

        obs = next_obs
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break

    df = pd.DataFrame.from_dict(records)
    # ensure numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df
