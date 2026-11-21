"""
Pure-RL search: can plain PPO (no oracle, no imitation, no hand-coded guards) beat the
rule-based controller if it is trained with sound RL practice?

Protocol (no test leakage):
  * Configurations are *selected* on a 14-day validation slice taken from the END of the
    training period (train' = 2023-04-15..2023-08-04, validation = 2023-08-04..2023-08-18).
  * The chosen configuration is then retrained on the full training period and reported
    once on the four held-out test weeks (see run_pure_rl_final.py).

Every config keeps the observation/reward semantics of the paper's environment; the knobs
are standard RL-engineering levers (input standardization, reward scale, time-of-day
features, random-start episodes, parallel envs, PPO hyperparameters, reward weights).
"""
import contextlib
import io
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from multiprocessing import Pool

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from components.MicrogridEnv import GridEnv
from components.RL_helper import eval_one_episode_RL, eval_one_episode_rule_based
from components.XAI import build_reward_df, print_policy_metrics
from experiments.common import build_train_eval_configs, eval_windows, _slice_to_config, _load_full_df

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_DAYS = 14
STEPS_PER_DAY = 96


def splits():
    _, _, train_df, eval_df = build_train_eval_configs()
    nv = VAL_DAYS * STEPS_PER_DAY
    return train_df, train_df.iloc[: len(train_df) - nv], train_df.iloc[len(train_df) - nv:], eval_df


def obs_norm_from(series_df):
    """Mean/std of the 5 base observation features over a (training) series."""
    cfg = _slice_to_config(series_df)
    env = GridEnv(cfg)
    rows = []
    for t in range(0, int(env.microgrid.MAX_TIMESTEP) - 2):
        env.microgrid.t = t
        rows.append(env._get_obs())
    arr = np.asarray(rows, dtype=np.float64)
    mean, std = arr.mean(0), arr.std(0) + 1e-6
    mean[0], std[0] = 0.5, 0.3  # SOC is constant in this scan; use its natural scale
    return mean.astype(np.float32), std.astype(np.float32)


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def eval_windows_profit(model, series_cfg, windows, env_kwargs_eval, rule_based=False):
    out = []
    for ws, we in windows:
        env = GridEnv(series_cfg, **env_kwargs_eval)
        if rule_based:
            _quiet(eval_one_episode_rule_based, env, start_date=ws, end_date=we)
        else:
            _quiet(eval_one_episode_RL, env, policy=model, start_date=ws, end_date=we)
        df = build_reward_df(env)
        m = _quiet(print_policy_metrics, df)
        out.append(float(m["financial_total"]))
    return out


def train_model(cfg, series_cfg, seed, norm):
    env_kwargs = dict(cfg.get("env", {}))
    if cfg.get("norm_obs"):
        env_kwargs["obs_norm"] = norm
    n_envs = int(cfg.get("n_envs", 1))
    set_random_seed(seed)

    def make():
        e = GridEnv(series_cfg, **env_kwargs)
        e.reset(seed=seed)
        return e

    venv = DummyVecEnv([make for _ in range(n_envs)])
    model = PPO("MlpPolicy", venv, verbose=0, seed=seed, device="cpu", **cfg.get("ppo", {}))
    model.learn(total_timesteps=int(cfg["steps"]))
    venv.close()
    return model


def eval_env_kwargs(cfg, norm):
    k = {kk: v for kk, v in cfg.get("env", {}).items()
         if kk in ("time_features", "rel_price", "price_lookahead_steps")}
    if cfg.get("norm_obs"):
        k["obs_norm"] = norm
    return k


def run_job(job):
    """job = (name, cfg, seed). Trains on train' and scores on the 2 validation weeks."""
    name, cfg, seed = job
    t0 = time.time()
    train_df, inner_train, val_df, _ = splits()
    if cfg.get("train_from"):  # optionally use earlier history (still strictly before the validation weeks)
        full_df = _load_full_df()
        inner_train = full_df.loc[cfg["train_from"]: val_df.index[0] - pd.Timedelta(minutes=15)]
    norm = obs_norm_from(inner_train)
    inner_cfg = _slice_to_config(inner_train)
    full_cfg = _slice_to_config(train_df)  # contains the validation weeks
    model = train_model(cfg, inner_cfg, seed, norm)
    v0 = val_df.index[0]
    windows = [(v0, v0 + pd.Timedelta(days=7)), (v0 + pd.Timedelta(days=7), v0 + pd.Timedelta(days=14))]
    profits = eval_windows_profit(model, full_cfg, windows, eval_env_kwargs(cfg, norm))
    return {"name": name, "seed": seed, "val_w1": profits[0], "val_w2": profits[1],
            "val_mean": float(np.mean(profits)), "secs": time.time() - t0}


def rule_based_val():
    train_df, _, val_df, _ = splits()
    full_cfg = _slice_to_config(train_df)
    v0 = val_df.index[0]
    windows = [(v0, v0 + pd.Timedelta(days=7)), (v0 + pd.Timedelta(days=7), v0 + pd.Timedelta(days=14))]
    p = eval_windows_profit(None, full_cfg, windows, {}, rule_based=True)
    return float(np.mean(p)), p


def run_search(configs, seeds, workers=12, out_csv=None):
    jobs = [(n, c, s) for n, c in configs.items() for s in seeds]
    t0 = time.time()
    with Pool(workers) as pool:
        rows = []
        for r in pool.imap_unordered(run_job, jobs):
            rows.append(r)
            print(f"[{time.time()-t0:6.0f}s] {r['name']:<28} seed={r['seed']} val_mean={r['val_mean']:8.2f} "
                  f"(w1 {r['val_w1']:.2f}, w2 {r['val_w2']:.2f})  {r['secs']:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df


def summarize(df, rb_mean):
    g = df.groupby("name")["val_mean"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False)
    g["vs_rule_based"] = g["mean"] - rb_mean
    return g


# ---------------------------------------------------------------------------
# Rolling-origin validation with a price-level stress test (stage 2)
# ---------------------------------------------------------------------------
FOLD_CUTS = {"F2": "2023-07-21 04:00", "F3": "2023-08-04 04:00"}
PRICE_MULTS = (1.0, 1.5)


def _scaled_cfg(df, mult):
    d = df.copy()
    d["price"] = d["price"] * mult
    return _slice_to_config(d)


def _fold_windows(cut):
    c = pd.Timestamp(cut)
    return [(c, c + pd.Timedelta(days=7)), (c + pd.Timedelta(days=7), c + pd.Timedelta(days=14))]


def rule_based_folds():
    full = _load_full_df()
    out = {}
    for fold, cut in FOLD_CUTS.items():
        c = pd.Timestamp(cut)
        series = full.loc["2023-04-15 04:00": c + pd.Timedelta(days=14)]
        for m in PRICE_MULTS:
            p = eval_windows_profit(None, _scaled_cfg(series, m), _fold_windows(cut), {}, rule_based=True)
            out[(fold, m)] = float(np.mean(p))
    return out


def run_job2(job):
    """job = (name, cfg, seed, fold). Train strictly before the fold cut; score on the two weeks after it,
    at price multipliers 1.0 and 1.5 (stress test for a rising price level, as seen from spring to late summer)."""
    name, cfg, seed, fold = job
    t0 = time.time()
    full = _load_full_df()
    cut = pd.Timestamp(FOLD_CUTS[fold])
    train_from = cfg.get("train_from", "2023-04-15 04:00")
    train_series = full.loc[train_from: cut - pd.Timedelta(minutes=15)]
    norm = obs_norm_from(train_series)
    model = train_model(cfg, _slice_to_config(train_series), seed, norm)
    eval_series = full.loc["2023-04-15 04:00": cut + pd.Timedelta(days=14)]
    res = {"name": name, "seed": seed, "fold": fold}
    for m in PRICE_MULTS:
        p = eval_windows_profit(model, _scaled_cfg(eval_series, m), _fold_windows(FOLD_CUTS[fold]),
                                eval_env_kwargs(cfg, norm))
        res[f"p_x{m}"] = float(np.mean(p))
    res["secs"] = time.time() - t0
    return res


def run_search2(configs, seeds, folds=("F2", "F3"), workers=12, out_csv=None):
    jobs = [(n, c, s, f) for n, c in configs.items() for s in seeds for f in folds]
    rb = rule_based_folds()
    print("rule-based:", {f"{k[0]}x{k[1]}": round(v, 2) for k, v in rb.items()}, flush=True)
    t0 = time.time()
    rows = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(run_job2, jobs):
            rows.append(r)
            print(f"[{time.time()-t0:6.0f}s] {r['name']:<26} {r['fold']} seed={r['seed']} "
                  f"x1.0 {r['p_x1.0']:7.2f} (rb {rb[(r['fold'], 1.0)]:.2f}) | x1.5 {r['p_x1.5']:7.2f} "
                  f"(rb {rb[(r['fold'], 1.5)]:.2f})  {r['secs']:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    for m in PRICE_MULTS:
        df[f"gain_x{m}"] = df.apply(lambda r: r[f"p_x{m}"] - rb[(r["fold"], m)], axis=1)
    df["gain"] = df[[f"gain_x{m}" for m in PRICE_MULTS]].mean(axis=1)
    if out_csv:
        df.to_csv(out_csv, index=False)
    return df, rb


def summarize2(df):
    g = df.groupby("name")[["gain", "gain_x1.0", "gain_x1.5"]].mean()
    g["gain_std_over_runs"] = df.groupby("name")["gain"].std()
    return g.sort_values("gain", ascending=False).round(2)
