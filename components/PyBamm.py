# simulate_with_pybamm.py
# Requires: pip install pybamm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pybamm
from typing import Optional, Tuple


def simulate_pybamm_from_df(
    df: pd.DataFrame,
    series_col: str = "power_kw",
    series_is_power: bool = True,
    pack_voltage_v: float = 400.0,
    model_name: str = "SPM",
    t_eval: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> Tuple["pybamm.Solution", "pybamm.Simulation", pd.Timestamp, np.ndarray, np.ndarray]:
    """
    Run a PyBaMM simulation driven by a time series (power_kW or current A).
    Returns (solution, sim, t0, current_a, t_seconds).

    See docstring in previous versions for parameter meanings.
    """

    # --- Input validation and preparation -----------------------------------
    if series_col not in df.columns:
        raise ValueError(f"Series column '{series_col}' not found in DataFrame.")

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.set_index(pd.to_datetime(df.index))
        except Exception:
            raise ValueError("df must have a pandas.DatetimeIndex or convertible index.")

    if len(df.index) < 2:
        raise ValueError("df must contain at least two timestamps to build an interpolant for PyBaMM.")

    df = df.sort_index()

    # Reference start time and seconds array
    t0 = df.index[0]
    t_seconds = (df.index - t0).total_seconds().astype(float)  # numpy array (1D)

    # Build numeric current array (A)
    if series_is_power:
        # Convert power (kW) -> W -> current (A): I = P_W / V_pack
        power_kw = df[series_col].astype(float).values
        power_w = power_kw * 1000.0
        current_a = power_w / float(pack_voltage_v)
    else:
        current_a = df[series_col].astype(float).values

    # --- Robust array shaping & validation for Interpolant -------------------
    t_seconds = np.asarray(t_seconds, dtype=float).ravel()
    current_a = np.asarray(current_a, dtype=float).ravel()

    if t_seconds.ndim != 1 or current_a.ndim != 1:
        raise ValueError("Internal error: t_seconds and current_a must be 1-D arrays.")

    if t_seconds.shape[0] != current_a.shape[0]:
        raise ValueError("Length mismatch: time vector and series vector must have same length.")

    n = t_seconds.shape[0]
    if n < 2:
        raise ValueError("Need at least two time points to create an interpolant for PyBaMM.")

    # Ensure strictly non-decreasing times. If duplicates exist, collapse them by averaging the current.
    diffs = np.diff(t_seconds)
    if not np.all(diffs >= 0):
        order = np.argsort(t_seconds)
        t_seconds = t_seconds[order]
        current_a = current_a[order]
        diffs = np.diff(t_seconds)

    # Handle repeated times by averaging the values for identical timestamps
    if np.any(diffs == 0):
        # find unique times and average corresponding currents
        uniq_times, inverse_idx = np.unique(t_seconds, return_inverse=True)
        agg_vals = np.zeros_like(uniq_times, dtype=float)
        counts = np.zeros_like(uniq_times, dtype=float)
        for i, v in enumerate(current_a):
            agg_vals[inverse_idx[i]] += v
            counts[inverse_idx[i]] += 1
        agg_vals /= counts
        t_seconds = uniq_times
        current_a = agg_vals

    # Recompute n
    n = t_seconds.shape[0]
    if n < 2:
        raise ValueError("After collapsing duplicate timestamps, fewer than 2 unique times remain; can't interpolate.")

    # Optionally, make sure arrays are floats (already done), and prepare y as shape (n, 1)
    # PyBaMM Interpolant accepts y as shape (n,) or (n, m). We'll pass (n, 1) to be robust.
    y_for_pybamm = current_a.reshape((n, 1))

    # Optional t_eval
    if t_eval is None:
        t_eval = t_seconds
    else:
        t_eval = np.asarray(t_eval, dtype=float).ravel()

    if verbose:
        print("simulate_pybamm_from_df:")
        print(f"  model_name = {model_name}, series_is_power = {series_is_power}, pack_voltage_v = {pack_voltage_v}")
        print(f"  input unique timesteps = {n}, start = {t0}, end = {pd.to_datetime(t0) + pd.to_timedelta(t_seconds[-1], unit='s')}")
        if n >= 2:
            print(f"  first interval (s) = {t_seconds[1] - t_seconds[0]:.2f}")

    # --- Build PyBaMM model --------------------------------------------------
    model_name_up = model_name.strip().upper()
    if model_name_up == "SPM":
        model = pybamm.lithium_ion.SPM()
    elif model_name_up == "DFN":
        model = pybamm.lithium_ion.DFN()
    else:
        raise ValueError("Unsupported model_name; use 'SPM' or 'DFN'")

    param = model.default_parameter_values

    # --- Create PyBaMM Interpolant (symbolic) --------------------------------
    # Use pybamm.Interpolant(pybamm.t, x, y). Here x = t_seconds (1D), y = (n,1) numeric array.
    # The Interpolant will be processed symbolically by PyBaMM and evaluated numerically during solve.
    try:
        current_interpolant = pybamm.Interpolant(pybamm.t, t_seconds, y_for_pybamm)
    except Exception as e:
        # Provide an informative message if Interpolant construction fails
        raise RuntimeError(
            "Failed to create pybamm.Interpolant. Ensure t_seconds is 1D numeric and current array has matching length.\n"
            f"Details: {e}"
        )

    # PyBaMM expects the parameter "Current function [A]" to be either a callable numeric or a PyBaMM expression.
    # Here we assign the symbolic interpolant.
    param["Current function [A]"] = current_interpolant

    # --- Create Simulation and solve ----------------------------------------
    sim = pybamm.Simulation(model, parameter_values=param)

    sol = sim.solve(t_eval=t_eval)

    if verbose:
        print("PyBaMM solve finished. Solution timepoints:", len(sol.t))
        try:
            keys = list(sol.keys())
            print(f"  solution contains variables: {keys[:8]}{'...' if len(keys)>8 else ''}")
        except Exception:
            pass

    return sol, sim, t0, current_a, t_seconds

