# components/RL_helper.py
import os
from datetime import timedelta, datetime
from typing import Optional

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from components.MicrogridEnv import GridEnv
from config.Config import mg_config
from config.types import MicrogridConfig

MODEL_PATH = "ppo_microgrid"

def make_env(config=mg_config, seed: Optional[int] = None) -> GridEnv:
    env = GridEnv(config)
    if seed is not None:
        # Call reset with seed if env supports it
        try:
            _ = env.reset(seed=seed)
        except TypeError:
            _ = env.reset()
    return env


def train(env_fn, timesteps: int, model_path: str, seed: int = 0):
    set_random_seed(seed)
    vec_env = DummyVecEnv([env_fn])
    model = PPO("MlpPolicy", vec_env, verbose=1, seed=seed)
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    vec_env.close()
    return model


def _unpack_reset(reset_return):
    if isinstance(reset_return, tuple):
        return reset_return[0]
    else:
        return reset_return


def _unpack_step(step_return):
    # Accept either gym classic or gymnasium
    if len(step_return) == 4:
        obs, reward, done, info = step_return
    elif len(step_return) == 5:
        obs, reward, terminated, truncated, info = step_return
        done = bool(terminated or truncated)
    else:
        raise ValueError("Unexpected step() return shape")
    return obs, reward, done, info


def eval_one_episode_RL(env: GridEnv, policy=None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
    # Reset env and then set start timestep (reset clears t -> 0)
    reset_ret = env.reset()
    obs = _unpack_reset(reset_ret)

    current_date = pd.to_datetime(env.microgrid.current_date.get_date(env.microgrid.t))

    step = timedelta(minutes=15)

    start = (start_date - current_date) / step
    end = (end_date - current_date) / step

    print("Start", start)
    print("End", end)
    print("Current date", current_date)

    if start is not None:
        env.microgrid.t = int(start) - 1
        for k in env.logs:
            env.logs[k].clear()

    # print("Current Date:", pd.to_datetime(env.microgrid.current_date.get_date(env.microgrid.t)))

    rows = []
    done = False
    while not done:
        # Capture the observation the policy actually conditioned on *before*
        # stepping -- pairing "action" with the post-step obs (as before) would
        # log the wrong state for explainability (LIME needs the state that
        # caused the decision, not the state the decision produced).
        obs_before = obs
        if policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=True)
            # policy.predict for continuous returns array-like; reduce to scalar
            try:
                action = float(np.asarray(action).ravel()[0])
            except Exception:
                action = float(action)

        step_ret = env.step(np.array([action], dtype=np.float32))
        obs, reward, done, info = _unpack_step(step_ret)

        rows.append(
            {
                "t": int(env.microgrid.t),
                "current_date": pd.to_datetime(env.microgrid.current_date.get_date(env.microgrid.t)),
                "obs": getattr(obs_before, "tolist", lambda: obs_before)(),
                "action": float(action),
                "reward": float(reward),
                "info": info,
            }
        )

        if end is not None and env.microgrid.t >= end:
            break

    return pd.DataFrame(rows)


def eval_one_episode_rule_based(env: GridEnv, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None):
    # Reset env and then set start timestep (reset clears t -> 0)
    reset_ret = env.reset()
    obs = _unpack_reset(reset_ret)

    current_date = pd.to_datetime(env.microgrid.current_date.get_date(env.microgrid.t))

    step = timedelta(minutes=15)

    start = (start_date - current_date) / step
    end = (end_date - current_date) / step


    if start is not None:
        env.microgrid.t = int(start) - 1
        for k in env.logs:
            env.logs[k].clear()

    print("Current Date:", pd.to_datetime(env.microgrid.current_date.get_date(env.microgrid.t)))

    rows = []
    done = False
    while not done:


        obs, terminated, truncated, info = env.step_rule_based()
        done = bool(terminated or truncated)


        rows.append(
            {
                "t": int(env.microgrid.t),
                "current_date": pd.to_datetime(env.microgrid.current_date.get_date(env.microgrid.t)),
                "obs": getattr(obs, "tolist", lambda: obs)(),
                # "action": float(action),
                # "reward": float(reward),
                "info": info,
            }
        )

        if end is not None and env.microgrid.t >= end:
            break

    return pd.DataFrame(rows)