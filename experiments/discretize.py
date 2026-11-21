"""Discrete-action wrapper so SB3's DQN (which requires gymnasium.spaces.Discrete)
can be run on the same environment as the continuous-action baselines, for the
DQN comparison requested by reviewers."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DiscretizeActionWrapper(gym.ActionWrapper):
    def __init__(self, env, n_bins: int = 7):
        super().__init__(env)
        self.n_bins = n_bins
        self.levels = np.linspace(-1.0, 1.0, n_bins, dtype=np.float32)
        self.action_space = spaces.Discrete(n_bins)

    def action(self, act):
        idx = int(act)
        return np.array([self.levels[idx]], dtype=np.float32)

    def __getattr__(self, name):
        # gymnasium's Wrapper.__getattr__ normally proxies to self.env, but the
        # eval helpers reach for env.microgrid/env.logs directly (not through the
        # gym API), so make that explicit rather than relying on the default proxy.
        return getattr(self.env, name)
