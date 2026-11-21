# tools/reward_policy_explainability.py
import os
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _safe_get(logs: Dict[str, Any], key: str, default=None):
    v = logs.get(key, default)
    if v is None:
        return np.array([])
    return np.asarray(v)


def build_reward_df(env) -> pd.DataFrame:
    """
    Build a robust DataFrame of per-step logs and reward components from env.logs.
    Expects env.logs to contain keys similar to your GridEnv:
      - requested_action_norm, soc, pv_kwh, load_kwh, battery_applied_raw,
      - financial_scaled,
      - pv_spike_charge_bonus, load_discharge_bonus,
      - charge_from_grid_penalty, discharge_to_grid_penalty, limit_action_penalty
      - grid_import_kwh, grid_export_kwh
    Missing keys are filled with zeros.
    """
    logs = getattr(env, "logs", {})
    # candidate keys and defaults
    keys_defaults = {
        "requested_action_norm": 0.0,
        "soc": 0.0,
        "pv_kwh": 0.0,
        "load_kwh": 0.0,
        "price": 0.0,
        "battery_applied_raw": 0.0,
        "financial_scaled": 0.0,
        "self_consumption": 0.0,
        "limit_action_penalty": 0.0,
        # "pv_spike_charge_bonus": 0.0,
        "load_discharge_bonus": 0.0,
        "charge_from_grid_penalty": 0.0,
        "discharge_to_grid_penalty": 0.0,
        "fine_this_step": 0.0,
        "grid_import_kwh": 0.0,
        "grid_export_kwh": 0.0,
        "import_penalty": 0.0,
        "export_reward": 0.0,
        "price_penalty": 0.0,
        "export_penalty_pv": 0.0,
    }

    # load arrays, pad to same length
    arrays = {}
    max_len = 0
    for k, default in keys_defaults.items():
        arr = _safe_get(logs, k, [])
        if arr is None:
            arr = np.array([])
        arr = np.asarray(arr, dtype=float)
        arrays[k] = arr
        max_len = max(max_len, arr.shape[0])

    # pad all arrays to max_len
    for k in list(arrays.keys()):
        arr = arrays[k]
        if arr.shape[0] < max_len:
            arr = np.concatenate([arr, np.zeros(max_len - arr.shape[0], dtype=float)])
            arrays[k] = arr

    df = pd.DataFrame({k: arrays[k] for k in arrays.keys()})
    # reconstruct signed reward components: penalties should be negative
    df["reward_financial"] = df["financial_scaled"].fillna(0.0)
    # df["reward_pv_bonus"] = df.get("pv_spike_charge_bonus", 0.0).fillna(0.0)
    df["reward_load_bonus"] = df.get("load_discharge_bonus", 0.0).fillna(0.0)
    df["self_consumption"] = df["self_consumption"].fillna(0.0)
    df["limit_action_penalty"] = df["limit_action_penalty"].fillna(0.0)

    # penalties may have been logged positive in env; convert to negative contributions
    df["charge_from_grid_penalty"] = -df.get("charge_from_grid_penalty", 0.0).fillna(0.0)
    df["discharge_to_grid_penalty"] = -df.get("discharge_to_grid_penalty", 0.0).fillna(0.0)
    df["fine_this_step"] = -df.get("fine_this_step", 0.0).fillna(0.0)
    df["limit_action_penalty"] = -df.get("limit_action_penalty", 0.0).fillna(0.0)
    df["import_penalty"] = -df.get("import_penalty", 0.0).fillna(0.0)
    df["export_reward"] = df.get("export_reward", 0.0).fillna(0.0)
    df["price_penalty"] = -df.get("price_penalty", 0.0).fillna(0.0)
    df["export_penalty_pv"] = -df.get("export_penalty_pv", 0.0).fillna(0.0)

    # total reward (reconstructed)
    comp_cols = [
        # "reward_financial",
        "import_penalty",
        "export_reward",
        # "reward_pv_bonus",
        # "self_consumption",
        "charge_from_grid_penalty",
        "discharge_to_grid_penalty",
        "limit_action_penalty",
        "export_penalty_pv",
        # "price_penalty",
        # "fine_this_step",
    ]
    df["reward_total_recon"] = df[comp_cols].sum(axis=1)

    # handy derived columns
    df["net_grid_kwh"] = df["grid_import_kwh"] - df["grid_export_kwh"]
    df["pv_surplus_kwh"] = np.maximum(0.0, df["pv_kwh"] - df["load_kwh"])
    df["load_surplus_kwh"] = np.maximum(0.0, df["load_kwh"] - df["pv_kwh"])

    return df


def plot_reward_decomposition(
    env,
    start_date,
    end_date,
    save_dir: Optional[str] = None,
    show: bool = True,
):
    """
    Publication-style reward decomposition (single panel).

    Shows individual reward/penalty components over time.
    """

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    df = build_reward_df(env).copy()
    n = df.shape[0]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    time_index = pd.date_range(start=start_date, end=end_date, periods=n)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Paper-style settings
    # -----------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
    })

    # -----------------------------
    # Components to plot
    # -----------------------------
    component_cols = [
        "import_penalty",
        "export_reward",
        "limit_action_penalty",
        "charge_from_grid_penalty",
        "discharge_to_grid_penalty",
        "export_penalty_pv",
    ]
    component_cols = [c for c in component_cols if c in df.columns]

    # -----------------------------
    # Color + style mapping
    # -----------------------------
    color_map = {
        "export_reward": "#2ca02c",  # green

        # penalties (same family, different tones)
        "import_penalty": "#d62728",
        "limit_action_penalty": "#ff7f0e",
        "charge_from_grid_penalty": "#8c564b",
        "discharge_to_grid_penalty": "#e377c2",
        "export_penalty_pv": "#9467bd",
    }

    linestyle_map = {
        "limit_action_penalty": "--",
        "charge_from_grid_penalty": ":",
        "discharge_to_grid_penalty": "-.",
    }

    # -----------------------------
    # Single plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 4.5))

    for col in component_cols:
        is_penalty = "penalty" in col

        ax.plot(
            time_index,
            df[col],
            label=col.replace("_", " "),
            linewidth=1.6,
            color=color_map.get(col, "tab:blue"),
            linestyle=linestyle_map.get(col, "-"),
            alpha=0.9 if not is_penalty else 0.8,
        )

    # Baseline
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.8)

    # Labels
    ax.set_ylabel("Reward contribution")
    ax.set_xlabel("Time")

    # Grid and styling
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    ax.legend(frameon=False, ncol=2, loc="upper right")

    # Panel label (useful for papers)
    ax.text(0.01, 0.95, "(a)", transform=ax.transAxes, fontsize=12, va="top")

    # Date formatting
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    fig.tight_layout()

    # -----------------------------
    # Save
    # -----------------------------
    if save_dir:
        png_path = os.path.join(save_dir, "reward_decomposition_timeseries.png")
        pdf_path = os.path.join(save_dir, "reward_decomposition_timeseries.pdf")
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return df

def plot_energy_soc_action(
    env,
    start_date,
    end_date,
    rl: bool = True,
    save_dir: Optional[str] = None,
    show: bool = True,
):
    """
    Publication-style time-series figure for energy flow, grid exchange,
    SOC/action, and price.

    Layout:
        (a) PV / Load / Battery
        (b) Net grid exchange
        (c) SOC and action
        (d) Electricity price

    Returns
    -------
    df : pandas.DataFrame
        Data used for plotting.
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    df = build_reward_df(env).copy()
    n = df.shape[0]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    time_index = pd.date_range(start=start_date, end=end_date, periods=n)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # -----------------------------
    # Paper-style matplotlib config
    # -----------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(
        4, 1,
        figsize=(12, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0, 1.1, 0.85], "hspace": 0.12},
    )
    ax0, ax1, ax2, ax3 = axes

    # -------------------------------------------------
    # (a) PV / Load / Battery applied energy
    # -------------------------------------------------
    handles = []
    labels = []

    if "pv_kwh" in df.columns:
        h = ax0.plot(
            time_index, df["pv_kwh"],
            color="tab:blue", linewidth=1.8, label="PV"
        )[0]
        handles.append(h)
        labels.append("PV")

    if "load_kwh" in df.columns:
        h = ax0.plot(
            time_index, df["load_kwh"],
            color="tab:orange", linewidth=1.8, label="Load"
        )[0]
        handles.append(h)
        labels.append("Load")

    if "battery_applied_raw" in df.columns:
        h = ax0.plot(
            time_index, df["battery_applied_raw"],
            color="tab:green", linewidth=1.5, linestyle="--",
            label="Battery"
        )[0]
        handles.append(h)
        labels.append("Battery")

    ax0.axhline(0, color="black", linewidth=0.8, alpha=0.7)
    ax0.set_ylabel("Energy (kWh)")
    ax0.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    if handles:
        ax0.legend(handles, labels, frameon=False, ncol=3, loc="upper right")
    ax0.text(0.01, 0.92, "(a)", transform=ax0.transAxes, fontsize=12, va="top")

    # -------------------------------------------------
    # (b) Net grid exchange
    # Positive/negative shading makes import/export easy to see
    # -------------------------------------------------
    if "net_grid_kwh" in df.columns:
        net = df["net_grid_kwh"].astype(float).to_numpy()

        ax1.axhline(0, color="black", linewidth=0.8)
        ax1.plot(time_index, net, color="tab:purple", linewidth=1.7, label="Net grid")
        ax1.fill_between(
            time_index, 0, net,
            where=net >= 0,
            interpolate=True,
            color="tab:green",
            alpha=0.18,
            label="Export"
        )
        ax1.fill_between(
            time_index, 0, net,
            where=net < 0,
            interpolate=True,
            color="tab:red",
            alpha=0.18,
            label="Import"
        )
        ax1.set_ylabel("Energy (kWh)")
        ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.legend(frameon=False, ncol=3, loc="upper right")
    else:
        ax1.axis("off")
    ax1.text(0.01, 0.92, "(b)", transform=ax1.transAxes, fontsize=12, va="top")

    # -------------------------------------------------
    # (c) SOC and action
    # Keep SOC on the left axis and action on the right axis
    # -------------------------------------------------
    has_soc = "soc" in df.columns
    has_action = rl and "requested_action_norm" in df.columns

    if has_soc or has_action:
        if has_soc:
            soc_line = ax2.plot(
                time_index, df["soc"],
                color="black", linewidth=1.8, label="SOC"
            )[0]
            ax2.set_ylabel("SOC")
            ax2.set_ylim(0, 1) if df["soc"].min() >= 0 and df["soc"].max() <= 1.05 else None
        else:
            soc_line = None

        ax2.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        ax2r = ax2.twinx()
        if has_action:
            action_line = ax2r.plot(
                time_index, df["requested_action_norm"],
                color="tab:blue", linewidth=1.5, linestyle="--",
                label="Action"
            )[0]
            ax2r.set_ylabel("Action (norm.)")
        else:
            action_line = None

        ax2r.spines["top"].set_visible(False)
        ax2r.grid(False)

        # Combine legends from both axes
        lines = [line for line in [soc_line, action_line] if line is not None]
        if lines:
            ax2.legend(lines, [line.get_label() for line in lines],
                       frameon=False, loc="upper right")

        # Optional reference lines for normalized SOC
        if has_soc and df["soc"].min() >= 0 and df["soc"].max() <= 1.05:
            ax2.axhline(0.2, color="gray", linewidth=0.7, linestyle=":", alpha=0.7)
            ax2.axhline(0.8, color="gray", linewidth=0.7, linestyle=":", alpha=0.7)
    else:
        ax2.axis("off")
    ax2.text(0.01, 0.92, "(c)", transform=ax2.transAxes, fontsize=12, va="top")

    # -------------------------------------------------
    # (d) Price
    # -------------------------------------------------
    if "price" in df.columns:
        ax3.plot(
            time_index, df["price"],
            color="tab:red", linewidth=1.7, label="Price"
        )
        ax3.fill_between(
            time_index, df["price"].to_numpy(), alpha=0.12, color="tab:red"
        )
        ax3.set_ylabel("Price")
        ax3.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.legend(frameon=False, loc="upper right")
    else:
        ax3.axis("off")
    ax3.text(0.01, 0.92, "(d)", transform=ax3.transAxes, fontsize=12, va="top")

    # -------------------------------------------------
    # Shared x-axis formatting
    # -------------------------------------------------
    ax3.set_xlabel("Time")
    ax3.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    # Slightly reduce label clutter
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    # No in-plot title; better for paper captions
    fig.tight_layout()

    if save_dir:
        suffix = "rl" if rl else "rulebased"
        png_path = os.path.join(save_dir, f"energy_soc_action_timeseries_{suffix}.png")
        pdf_path = os.path.join(save_dir, f"energy_soc_action_timeseries_{suffix}.pdf")
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return df

def plot_pv_soc_regimes(
    env,
    start_date,
    end_date,
    save_dir: Optional[str] = None,
    show: bool = True,
    pv_low: float = 0.29,
    pv_high: float = 1.36,
):
    """
    Publication-style figure showing:
      (a) PV over time with regime thresholds
      (b) Battery SOC over time

    LIME-inspired regime interpretation:
      PV < pv_low        -> idle
      pv_low <= PV <= pv_high -> discharging
      PV > pv_high       -> charging
    """

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    df = build_reward_df(env).copy()
    n = df.shape[0]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    time_index = pd.date_range(start=start_date, end=end_date, periods=n)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Paper-style settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
    })

    pv_col = "pv_kwh"
    soc_col = "soc"

    has_pv = pv_col in df.columns
    has_soc = soc_col in df.columns

    if not has_pv and not has_soc:
        raise ValueError("Neither 'pv_kwh' nor 'soc' found in dataframe.")

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 0.95], "hspace": 0.12},
    )
    ax0, ax1 = axes

    # -----------------------------
    # (a) PV with regime thresholds
    # -----------------------------
    if has_pv:
        pv = df[pv_col].astype(float).to_numpy()
        pv_max = max(float(pd.Series(pv).max()), pv_high) * 1.08

        # Background regime bands
        ax0.axhspan(0, pv_low, facecolor="0.92", alpha=0.9, zorder=0)
        ax0.axhspan(pv_low, pv_high, facecolor="tab:orange", alpha=0.08, zorder=0)
        ax0.axhspan(pv_high, pv_max, facecolor="tab:green", alpha=0.08, zorder=0)

        # PV line
        ax0.plot(
            time_index,
            pv,
            color="tab:blue",
            linewidth=1.8,
            label="PV",
            zorder=3,
        )

        # Threshold lines
        ax0.axhline(pv_low, color="tab:gray", linestyle="--", linewidth=1.2, zorder=2)
        ax0.axhline(pv_high, color="tab:gray", linestyle="--", linewidth=1.2, zorder=2)

        # Threshold labels
        ax0.text(
            0.99, pv_low, f"  {pv_low:.2f}",
            transform=ax0.get_yaxis_transform(),
            ha="left", va="center", fontsize=9, color="tab:gray"
        )
        ax0.text(
            0.99, pv_high, f"  {pv_high:.2f}",
            transform=ax0.get_yaxis_transform(),
            ha="left", va="center", fontsize=9, color="tab:gray"
        )

        # Regime labels
        ax0.text(
            0.02, 0.12, "Idle",
            transform=ax0.transAxes,
            fontsize=10,
            color="0.35",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
        )
        ax0.text(
            0.02, 0.44, "Discharging",
            transform=ax0.transAxes,
            fontsize=10,
            color="tab:orange",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
        )
        ax0.text(
            0.02, 0.78, "Charging",
            transform=ax0.transAxes,
            fontsize=10,
            color="tab:green",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
        )

        ax0.set_ylabel("PV")
        ax0.set_ylim(0, pv_max)
        ax0.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        ax0.spines["top"].set_visible(False)
        ax0.spines["right"].set_visible(False)
        ax0.text(0.01, 0.94, "(a)", transform=ax0.transAxes, fontsize=12, va="top")

        # Compact legend for the regime interpretation
        legend_handles = [
            Patch(facecolor="0.92", edgecolor="none", label=f"PV < {pv_low:.2f} (idle)"),
            Patch(facecolor="tab:orange", edgecolor="none", alpha=0.15, label=f"{pv_low:.2f}–{pv_high:.2f} (discharge)"),
            Patch(facecolor="tab:green", edgecolor="none", alpha=0.15, label=f"PV > {pv_high:.2f} (charge)"),
        ]
        ax0.legend(handles=legend_handles, frameon=False, loc="upper right", ncol=1)
    else:
        ax0.axis("off")

    # -----------------------------
    # (b) SOC
    # -----------------------------
    if has_soc:
        soc = df[soc_col].astype(float).to_numpy()

        ax1.plot(
            time_index,
            soc,
            color="black",
            linewidth=1.9,
            label="SOC",
            zorder=3,
        )
        ax1.fill_between(
            time_index,
            soc,
            alpha=0.10,
            color="black",
            zorder=1,
        )

        ax1.set_ylabel("SOC")
        if soc.min() >= 0 and soc.max() <= 1.05:
            ax1.set_ylim(0, 1.02)
            ax1.axhline(0.2, color="0.6", linestyle=":", linewidth=0.9, alpha=0.8)
            ax1.axhline(0.8, color="0.6", linestyle=":", linewidth=0.9, alpha=0.8)

        ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.text(0.01, 0.94, "(b)", transform=ax1.transAxes, fontsize=12, va="top")
        ax1.legend(frameon=False, loc="upper right")
    else:
        ax1.axis("off")

    # -----------------------------
    # X-axis formatting
    # -----------------------------
    ax1.set_xlabel("Time")
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    # No in-figure title; better for the paper caption
    fig.tight_layout()

    if save_dir:
        png_path = os.path.join(save_dir, "pv_soc_regimes.png")
        pdf_path = os.path.join(save_dir, "pv_soc_regimes.pdf")
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return df

def plot_load_soc_regimes(
    env,
    start_date,
    end_date,
    save_dir: Optional[str] = None,
    show: bool = True,
    load_idle: float = 0.27,
    load_charge: float = 1.09,
    load_discharge: float = 2.51,
):
    """
    Publication-style figure showing:
      (a) Load over time with regime thresholds
      (b) Battery SOC over time

    LIME-inspired regime interpretation:
      load < load_idle                -> idle
      load_idle < load < load_charge  -> charge
      load_charge < load < load_discharge -> discharge
    """

    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    df = build_reward_df(env).copy()
    n = df.shape[0]

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    time_index = pd.date_range(start=start_date, end=end_date, periods=n)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
    })

    load_col = "load_kwh"
    soc_col = "soc"

    has_load = load_col in df.columns
    has_soc = soc_col in df.columns

    if not has_load and not has_soc:
        raise ValueError("Neither 'load_kwh' nor 'soc' found in dataframe.")

    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 6.8),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 0.95], "hspace": 0.12},
    )
    ax0, ax1 = axes

    # -----------------------------
    # (a) Load with regime thresholds
    # -----------------------------
    if has_load:
        load = df[load_col].astype(float).to_numpy()
        ymax = max(float(pd.Series(load).max()), load_discharge) * 1.08

        # Regime bands
        ax0.axhspan(0, load_idle, facecolor="0.92", alpha=0.9, zorder=0)
        ax0.axhspan(load_idle, load_charge, facecolor="tab:green", alpha=0.08, zorder=0)
        ax0.axhspan(load_charge, load_discharge, facecolor="tab:orange", alpha=0.08, zorder=0)

        if ymax > load_discharge:
            ax0.axhspan(load_discharge, ymax, facecolor="tab:orange", alpha=0.03, zorder=0)

        # Load trajectory
        ax0.plot(
            time_index,
            load,
            color="tab:blue",
            linewidth=1.8,
            label="Load",
            zorder=3,
        )

        # Threshold lines
        for thr in [load_idle, load_charge, load_discharge]:
            ax0.axhline(thr, color="tab:gray", linestyle="--", linewidth=1.1, zorder=2)

        # Threshold labels on the right
        ax0.text(
            0.99, load_idle, f"  {load_idle:.2f}",
            transform=ax0.get_yaxis_transform(),
            ha="left", va="center", fontsize=9, color="tab:gray"
        )
        ax0.text(
            0.99, load_charge, f"  {load_charge:.2f}",
            transform=ax0.get_yaxis_transform(),
            ha="left", va="center", fontsize=9, color="tab:gray"
        )
        ax0.text(
            0.99, load_discharge, f"  {load_discharge:.2f}",
            transform=ax0.get_yaxis_transform(),
            ha="left", va="center", fontsize=9, color="tab:gray"
        )

        # Regime labels
        ax0.text(
            0.02, 0.12, "Idle",
            transform=ax0.transAxes,
            fontsize=10,
            color="0.35",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
        )
        ax0.text(
            0.02, 0.42, "Charge",
            transform=ax0.transAxes,
            fontsize=10,
            color="tab:green",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
        )
        ax0.text(
            0.02, 0.76, "Discharge",
            transform=ax0.transAxes,
            fontsize=10,
            color="tab:orange",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75),
        )

        ax0.set_ylabel("Load")
        ax0.set_ylim(0, ymax)
        ax0.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        ax0.spines["top"].set_visible(False)
        ax0.spines["right"].set_visible(False)
        ax0.text(0.01, 0.94, "(a)", transform=ax0.transAxes, fontsize=12, va="top")

        legend_handles = [
            Patch(facecolor="0.92", edgecolor="none", label=f"Load < {load_idle:.2f} (idle)"),
            Patch(facecolor="tab:green", edgecolor="none", alpha=0.15, label=f"{load_idle:.2f}–{load_charge:.2f} (charge)"),
            Patch(facecolor="tab:orange", edgecolor="none", alpha=0.15, label=f"{load_charge:.2f}–{load_discharge:.2f} (discharge)"),
        ]
        ax0.legend(handles=legend_handles, frameon=False, loc="upper right")
    else:
        ax0.axis("off")

    # -----------------------------
    # (b) SOC
    # -----------------------------
    if has_soc:
        soc = df[soc_col].astype(float).to_numpy()

        ax1.plot(
            time_index,
            soc,
            color="black",
            linewidth=1.9,
            label="SOC",
            zorder=3,
        )
        ax1.fill_between(
            time_index,
            soc,
            alpha=0.10,
            color="black",
            zorder=1,
        )

        ax1.set_ylabel("SOC")
        if soc.min() >= 0 and soc.max() <= 1.05:
            ax1.set_ylim(0, 1.02)
            ax1.axhline(0.2, color="0.6", linestyle=":", linewidth=0.9, alpha=0.8)
            ax1.axhline(0.8, color="0.6", linestyle=":", linewidth=0.9, alpha=0.8)

        ax1.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.text(0.01, 0.94, "(b)", transform=ax1.transAxes, fontsize=12, va="top")
        ax1.legend(frameon=False, loc="upper right")
    else:
        ax1.axis("off")

    # -----------------------------
    # X-axis formatting
    # -----------------------------
    ax1.set_xlabel("Time")
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))

    fig.tight_layout()

    if save_dir:
        png_path = os.path.join(save_dir, "load_soc_regimes.png")
        pdf_path = os.path.join(save_dir, "load_soc_regimes.pdf")
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return df


import pandas as pd

def compute_lime_timeseries(
    df_eval, policy, feature_names, num_samples=200,
    battery_params: Optional[Dict[str, float]] = None, soc_feature_idx: int = 0,
):
    from lime.lime_tabular import LimeTabularExplainer
    import numpy as np

    X = np.stack(df_eval["obs"].apply(lambda x: np.array(x, dtype=np.float32)))

    explainer = LimeTabularExplainer(
        training_data=X,
        feature_names=feature_names,
        class_names=["discharge", "idle", "charge"],
        mode="classification"
    )

    predict_fn = make_lime_predict_fn(
        policy, battery_params=battery_params, soc_feature_idx=soc_feature_idx
    )

    lime_results = []

    step_indices = np.linspace(0, len(X)-1, num_samples, dtype=int)

    for i in step_indices:
        x = X[i]

        exp = explainer.explain_instance(
            x,
            predict_fn,
            num_features=len(feature_names)
        )

        row = {f: 0.0 for f in feature_names}

        for k, v in exp.as_list():
            for f in feature_names:
                if f in k:  # match substring
                    row[f] += v
        lime_results.append(row)

    # expand to full length: place each sample at its true timestep position
    # before reindexing, so interpolation fills gaps between the *correct*
    # positions rather than collapsing all samples into the first num_samples
    # rows (which left every row beyond num_samples a flat forward-fill).
    lime_df = pd.DataFrame(lime_results, index=step_indices)
    lime_df = lime_df.reindex(range(len(df_eval))).interpolate(limit_direction="both").fillna(0)

    return lime_df


def plot_pv_explainability(df, lime_results, save_path=None):
    """
    df: your dataframe with logs
    lime_results: list of dicts with feature importances per timestep
    """

    # convert lime results to DataFrame
    lime_df = lime_results

    # PV-dominant mask
    mask = df["pv_kwh"] > df["load_kwh"]

    t = np.arange(len(df))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # ------------------------
    # 1. System behavior
    # ------------------------
    axes[0].plot(t, df["pv_kwh"], label="PV", color="orange")
    axes[0].plot(t, df["load_kwh"], label="Load", color="blue")
    axes[0].plot(t, df["soc"]*10, label="SOC", color="green")

    axes[0].fill_between(t, 0, max(df["pv_kwh"]), where=mask, alpha=0.1, color="orange", label="PV dominant")

    axes[0].set_ylabel("Energy")
    axes[0].legend()
    axes[0].set_title("System behavior (PV-dominant periods highlighted)")

    # ------------------------
    # 2. LIME feature importance over time
    # ------------------------
    for col in lime_df.columns:
        axes[1].plot(t, lime_df[col], label=col)

    # axes[1].fill_between(t, lime_df.max().max(), lime_df.min().min(), where=mask, alpha=0.1, color="orange")
    axes[1].fill_between(
        t,
        np.full_like(t, lime_df.min().min()),
        np.full_like(t, lime_df.max().max()),
        # where=mask,
        alpha=0.1,
        color="orange",
    )

    axes[1].set_ylabel("LIME weight")
    axes[1].legend()
    axes[1].set_title("Feature importance (LIME)")

    # ------------------------
    # 3. Aggregated importance
    # ------------------------
    avg_importance = lime_df[mask].mean()

    axes[2].bar(avg_importance.index, avg_importance.values, color="purple")
    axes[2].set_ylabel("Avg importance")
    axes[2].set_title("Average feature contribution (PV-dominant regime)")

    plt.xlabel("Time step")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_action_vs_components_scatter(
    env,
    save_dir: Optional[str] = None,
    show: bool = True,
):
    df = build_reward_df(env)

    if "requested_action_norm" not in df.columns:
        print("Skipping scatter plots: 'requested_action_norm' not found.")
        return df

    components = [
        "import_penalty",
        "export_reward",
        "limit_action_penalty",
        "reward_charge_from_grid_penalty",
        "reward_discharge_to_grid_penalty",
        "export_penalty_pv",
    ]

    present_components = [c for c in components if c in df.columns]
    ncomp = len(present_components)

    if ncomp == 0:
        print("Skipping scatter plots: no reward components found.")
        return df

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    cols = 2
    rows = int(np.ceil(ncomp / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, comp in enumerate(present_components):
        axc = axes[i]
        axc.scatter(df["requested_action_norm"], df[comp], s=8, alpha=0.7)
        axc.axvline(0.0, color="k", linewidth=0.5)
        axc.set_xlabel("action_norm")
        axc.set_ylabel(comp)
        axc.grid(True)

    for j in range(ncomp, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Action vs reward-component scatter plots")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir:
        fig.savefig(os.path.join(save_dir, "action_vs_components_scatter.png"))

    if show:
        plt.show()

    return df


def print_policy_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute policy evaluation metrics.

    Metrics:
        - financial_total: export revenue - import cost
        - import_total: total imported energy (kWh)
        - export_total: total exported energy (kWh)
        - self_consumption_mean: average self-consumption ratio
    """

    # --- Financial metric ---
    financial = (
        df["grid_export_kwh"] * 0.75 * df["price"]
        - df["grid_import_kwh"] * df["price"]
    )

    # --- Energy totals ---
    import_total = float(df["grid_import_kwh"].sum())
    export_total = float(df["grid_export_kwh"].sum())

    # --- Self-consumption ---
    # Fraction of PV generation consumed locally rather than exported. Export is
    # clipped to pv_kwh before subtracting, since export beyond that step's PV must
    # have come from the battery, not unconsumed PV -- otherwise the ratio can go
    # negative, which isn't a meaningful self-consumption value.
    pv = df["pv_kwh"]
    exported = df["grid_export_kwh"]

    self_consumption = np.where(
        pv > 0,
        (pv - np.minimum(exported, pv)) / pv,
        0.0
    )

    # --- Aggregates ---
    financial_total = float(financial.sum())
    self_consumption_mean = float(self_consumption.mean())

    metrics = {
        "financial_total": financial_total,
        "import_total_kwh": import_total,
        "export_total_kwh": export_total,
        "self_consumption_mean": self_consumption_mean,
    }

    # --- Print nicely ---
    print("\nPolicy metrics")
    print("-------------------------")
    print(f"Financial profit (€): {financial_total:.4f}")
    print(f"Grid import (kWh):    {import_total:.4f}")
    print(f"Grid export (kWh):    {export_total:.4f}")
    print(f"Self-consumption:     {self_consumption_mean:.4f}")

    return metrics



def plot_financial_timeseries(
    df_rl: pd.DataFrame,
    df_rule: pd.DataFrame,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    """
    Plot per-timestep financial comparison between RL and Rule-based policies
    in a publication-style figure.

    Returns
    -------
    financial_rl : pd.Series
    financial_rule : pd.Series
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # Per-step financial metric
    financial_rl = (
        df_rl["grid_export_kwh"] * 0.75 * df_rl["price"]
        - df_rl["grid_import_kwh"] * df_rl["price"]
    )

    financial_rule = (
        df_rule["grid_export_kwh"] * 0.75 * df_rule["price"]
        - df_rule["grid_import_kwh"] * df_rule["price"]
    )

    # Align lengths safely
    n = min(len(financial_rl), len(financial_rule))
    financial_rl = financial_rl.iloc[:n].reset_index(drop=True)
    financial_rule = financial_rule.iloc[:n].reset_index(drop=True)

    timesteps = np.arange(n)

    # Cumulative values are easier to compare in papers
    cum_rl = financial_rl.cumsum()
    cum_rule = financial_rule.cumsum()
    diff = financial_rl - financial_rule
    cum_diff = diff.cumsum()

    total_rl = cum_rl.iloc[-1]
    total_rule = cum_rule.iloc[-1]
    improvement = total_rl - total_rule
    pct_improvement = 100 * improvement / (abs(total_rule) + 1e-12)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Publication-style defaults
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })

    fig, axes = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]}
    )

    # Top: cumulative financial value
    axes[0].plot(timesteps, cum_rl, label="RL", linewidth=2.2)
    axes[0].plot(timesteps, cum_rule, label="Rule-based", linewidth=2.2, linestyle="--")
    axes[0].set_ylabel("Cumulative financial value")
    axes[0].set_title("Cumulative financial performance comparison")
    axes[0].grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    axes[0].legend(frameon=False)

    # Annotate final totals
    axes[0].annotate(
        f"RL final = {total_rl:.2f}",
        xy=(timesteps[-1], cum_rl.iloc[-1]),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        fontsize=10,
    )
    axes[0].annotate(
        f"RB final = {total_rule:.2f}",
        xy=(timesteps[-1], cum_rule.iloc[-1]),
        xytext=(-8, -18),
        textcoords="offset points",
        ha="right",
        fontsize=10,
    )

    # Bottom: difference (RL - RB)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].plot(timesteps, cum_diff, linewidth=2.0)
    axes[1].fill_between(timesteps, cum_diff, 0, alpha=0.2)
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Cumulative\n(RL - RB)")
    axes[1].grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    fig.text(
        0.5, 0.01,
        f"Final improvement = {improvement:.2f} ({pct_improvement:.1f}%)",
        ha="center",
        fontsize=11
    )

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_dir:
        fig.savefig(os.path.join(save_dir, "financial_timeseries_comparison.pdf"),
                    bbox_inches="tight")
        fig.savefig(os.path.join(save_dir, "financial_timeseries_comparison.png"),
                    bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return financial_rl, financial_rule

import ast
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer


MODE_NAMES = ["discharge", "idle", "charge"]


def _parse_obs(obs):
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32)
    if isinstance(obs, (list, tuple)):
        return np.asarray(obs, dtype=np.float32)
    if isinstance(obs, str):
        return np.asarray(ast.literal_eval(obs), dtype=np.float32)
    return np.asarray(obs, dtype=np.float32)


def action_to_mode(action: float, eps: float = 0.05) -> int:
    """
    0 = discharge
    1 = idle
    2 = charge
    """
    a = float(action)
    if a < -eps:
        return 0
    if a > eps:
        return 2
    return 1


def raw_to_executed_action(
    a_raw: float,
    soc: float,
    battery_capacity: float,
    high_capacity: float,
    low_capacity: float,
    max_output: float,
) -> float:
    """
    Approximate the physically *executed* action (same normalized [-1, 1] units as
    a_raw) after enforcing the per-step power limit and battery headroom, mirroring
    Microgrid.run_timestep's clipping. The paper's explainability analysis is
    defined over this executed action, not the raw (potentially infeasible) policy
    output, so LIME must be evaluated against it too -- this is what lets a
    perturbed sample's SOC feature actually constrain the classified mode.
    """
    if max_output <= 0:
        return 0.0
    requested_kwh = float(np.clip(a_raw, -1.0, 1.0)) * max_output
    current_energy = float(soc) * battery_capacity
    max_charge_available = high_capacity - current_energy
    max_discharge_available = current_energy - low_capacity
    min_allowed = -min(max_output, max(0.0, max_discharge_available))
    max_allowed = min(max_output, max(0.0, max_charge_available))
    executed_kwh = float(np.clip(requested_kwh, min_allowed, max_allowed))
    return executed_kwh / max_output


def action_to_mode_proba(action: float, eps: float = 0.05, temp: float = 0.15) -> np.ndarray:
    """
    Smooth probabilities for LIME from a continuous action.
    """
    a = float(np.clip(action, -1.0, 1.0))
    logits = np.array(
        [
            (-(a + eps)) / temp,        # discharge
            (eps - abs(a)) / temp,      # idle
            ((a - eps)) / temp,         # charge
        ],
        dtype=np.float32,
    )
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)
    return probs


def make_lime_predict_fn(
    policy,
    eps: float = 0.05,
    temp: float = 0.15,
    battery_params: Optional[Dict[str, float]] = None,
    soc_feature_idx: int = 0,
):
    """
    Wrap SB3 policy so LIME gets probabilities for 3 classes.

    If `battery_params` is given (capacity/high_capacity/low_capacity/max_output),
    the raw policy output for each perturbed sample is passed through
    `raw_to_executed_action` using that sample's own SOC feature before being
    classified into a mode -- so the surrogate model LIME fits explains the
    *executed* action (as the paper claims), not the raw, potentially infeasible
    network output. Leaving it None reproduces the previous raw-action behavior.
    """
    def predict_proba(X):
        X = np.asarray(X, dtype=np.float32)
        out = []
        for x in X:
            a, _ = policy.predict(x.reshape(1, -1), deterministic=True)
            a = float(np.asarray(a).ravel()[0])
            if battery_params is not None:
                a = raw_to_executed_action(a, soc=x[soc_feature_idx], **battery_params)
            out.append(action_to_mode_proba(a, eps=eps, temp=temp))
        return np.asarray(out, dtype=np.float32)

    return predict_proba


def explain_lime_modes(
    df_eval,
    policy,
    feature_names: List[str],
    out_dir: str = "figs/lime",
    eps: float = 0.05,
    temp: float = 0.15,
    background_size: int = 500,
    num_features: int = 7,
    random_state: int = 0,
    panel_order: Optional[List[int]] = None,
    battery_params: Optional[Dict[str, float]] = None,
    soc_feature_idx: int = 0,
):
    """
    Publication-style LIME explanation figure for discrete action modes.

    Produces a compact multi-panel figure with:
    - consistent color mapping (green = positive, red = negative)
    - shared x-axis scale
    - sorted feature contributions
    - panel labels (a), (b), (c)
    - PDF/PNG export suitable for a paper
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from lime.lime_tabular import LimeTabularExplainer

    os.makedirs(out_dir, exist_ok=True)

    # Paper-style matplotlib settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
    })

    X = np.stack(df_eval["obs"].apply(_parse_obs).to_numpy()).astype(np.float32)
    actions = df_eval["action"].astype(float).to_numpy()

    if battery_params is not None:
        # Executed action: the raw policy output after enforcing SOC headroom and
        # the per-step power limit, matching what the paper's Sec. 2.8 says the
        # explainability analysis is performed on. This can differ from the raw
        # action whenever the request is infeasible (battery near a SOC bound).
        executed_actions = np.array([
            raw_to_executed_action(a, soc=x[soc_feature_idx], **battery_params)
            for a, x in zip(actions, X)
        ])
        raw_modes = np.array([action_to_mode(a, eps=eps) for a in actions], dtype=int)
        modes = np.array([action_to_mode(a, eps=eps) for a in executed_actions], dtype=int)
        clipped_frac = float(np.mean(np.abs(executed_actions - actions) > eps))
        mode_changed_frac = float(np.mean(raw_modes != modes))
        print(
            f"[LIME] executed vs. raw action: {clipped_frac:.1%} of steps were "
            f"clipped by more than eps={eps}, changing the discrete mode on "
            f"{mode_changed_frac:.1%} of steps."
        )
    else:
        modes = np.array([action_to_mode(a, eps=eps) for a in actions], dtype=int)

    if len(X) > background_size:
        rng = np.random.default_rng(random_state)
        idx_bg = rng.choice(len(X), size=background_size, replace=False)
        X_bg = X[idx_bg]
    else:
        X_bg = X

    explainer = LimeTabularExplainer(
        training_data=X_bg,
        feature_names=feature_names,
        class_names=MODE_NAMES,
        mode="classification",
        discretize_continuous=True,
        random_state=random_state,
    )

    predict_fn = make_lime_predict_fn(
        policy, eps=eps, temp=temp, battery_params=battery_params, soc_feature_idx=soc_feature_idx
    )

    if panel_order is None:
        panel_order = list(range(len(MODE_NAMES)))

    # First pass: compute explanations and collect weights for consistent x-limits
    summary = {}
    panel_data = []
    all_abs_vals = []

    for mode_idx in panel_order:
        mode_name = MODE_NAMES[mode_idx]
        mode_rows = np.where(modes == mode_idx)[0]

        if len(mode_rows) == 0:
            summary[mode_name] = None
            panel_data.append((mode_idx, mode_name, None, None, None, None))
            continue

        # Representative sample: middle sample from this mode
        sample_idx = int(mode_rows[len(mode_rows) // 2])
        x0 = X[sample_idx]

        exp = explainer.explain_instance(
            x0,
            predict_fn,
            labels=[mode_idx],
            num_features=min(num_features, X.shape[1]),
        )

        weights = exp.as_list(label=mode_idx)

        # Sort by absolute contribution, then plot most important first
        weights = sorted(weights, key=lambda kv: abs(kv[1]))
        feats = [w[0] for w in weights]
        vals = [w[1] for w in weights]

        all_abs_vals.extend([abs(v) for v in vals])

        summary[mode_name] = {
            "sample_idx": sample_idx,
            "action": float(actions[sample_idx]),
            "mode": int(mode_idx),
            "weights": weights,
        }

        panel_data.append((mode_idx, mode_name, sample_idx, actions[sample_idx], feats, vals))

    # Shared x-limits for comparability across panels
    if len(all_abs_vals) > 0:
        max_abs = max(all_abs_vals)
        xlim = (-1.10 * max_abs, 1.10 * max_abs)
    else:
        xlim = (-0.1, 0.1)

    n_panels = len(panel_order)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(7.2 * ncols, 3.6 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for i, (mode_idx, mode_name, sample_idx, action_val, feats, vals) in enumerate(panel_data):
        ax = axes[i]

        if feats is None:
            ax.set_title(f"{mode_name} (no samples)")
            ax.axis("off")
            continue

        colors = ["tab:green" if v >= 0 else "tab:red" for v in vals]

        ax.barh(feats, vals, color=colors, edgecolor="none", height=0.75)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_xlim(xlim)

        # Put strongest features at the top
        ax.invert_yaxis()

        ax.set_title(f"{mode_name}  |  action = {action_val:.3f}", pad=6)
        ax.set_xlabel("LIME weight")
        ax.set_ylabel("")

        # Panel label
        ax.text(
            0.5, -0.18, panel_labels[i],
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=13
        )

        # Optional: remove top/right spines for a cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        print(f"\n=== LIME for {mode_name.upper()} ===")
        print(f"Sample index: {sample_idx}")
        print(f"Action: {action_val:.3f}")
        for f, v in sorted(summary[mode_name]["weights"], key=lambda kv: abs(kv[1]), reverse=True):
            print(f"  {f}: {v:.4f}")

    # Hide unused axes if the grid is larger than needed
    for j in range(len(panel_data), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "LIME-based explanation of the actor policy in idle mode",
        y=1.02,
        fontsize=12,
    )

    fig_path_png = os.path.join(out_dir, "lime_modes_summary_paper.png")
    fig_path_pdf = os.path.join(out_dir, "lime_modes_summary_paper.pdf")
    fig.savefig(fig_path_png, dpi=300, bbox_inches="tight")
    fig.savefig(fig_path_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved summary plot: {fig_path_png}")
    print(f"Saved summary plot: {fig_path_pdf}")

    return explainer, summary