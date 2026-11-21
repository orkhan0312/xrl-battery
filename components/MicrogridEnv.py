# components/MicrogridEnv.py
from typing import Tuple, Dict, Optional
import datetime
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from components.Microgrid import Microgrid
from config.types import MicrogridConfig


class GridEnv(gym.Env):
    """
    Env: reward = financial + pv_spike_charge_bonus + load_surplus_discharge_bonus
          - small penalties for charge-from-grid / discharge-to-grid / limit-action
    Observation: [soc, load_kW, pv_kW, import_price, export_price, weekday_norm, month_norm]
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config: MicrogridConfig,
        *,
        # penalties
        financial_coeff: float = 0.5, #2
        import_coeff: float = 4, #8
        export_coeff: float = 2, #4
        charge_from_grid_penalty_coeff: float = 2, #4
        discharge_to_grid_penalty_coeff: float = 0.5, #0.5
        limit_action_penalty_coeff: float = 0.5,
        # PV-spike bonus (tunable — default is large)
        # pv_spike_charge_coeff: float = 1.0,
        # pv_spike_threshold: float = 1.0,  # kWh surplus to call it a "spike"
        # Load-surplus discharge bonus (tunable)
        price_coeff: float = 0.002,
        export_pv_coeff: float = 1,

        load_discharge_coeff: float = 2.0,
        load_surplus_threshold: float = 0.5,  # kWh surplus to call it a "load spike"
        # Experimental: only fire the charge-from-grid / discharge-to-grid penalties
        # when the grid interaction was genuinely unnecessary (charging from the grid
        # despite PV already covering load; discharging to export despite an unmet
        # local deficit). When PV < load (a real deficit) or PV > load (a real
        # surplus) respectively, grid-timed charge/discharge is left unpenalized,
        # allowing price-arbitrage behavior similar to the rule-based controller's
        # structural self-consumption-first logic, without imitation pretraining.
        conditional_grid_penalties: bool = False,
        # Opt-in pure-RL training aids (all off by default, so published results are unchanged):
        time_features: bool = False,   # append sin/cos of time-of-day of the step being decided
        obs_norm=None,                 # (mean, std) arrays for the 5 base features, from *training* data
        reward_scale: float = 1.0,     # constant multiplier on the reward
        random_start: bool = False,    # start each episode at a random point of the series
        episode_len: Optional[int] = None,  # truncate episodes after this many steps
        rel_price: bool = False,       # replace absolute prices by price relative to its trailing 24h/7d mean
        price_lookahead_steps: tuple = (),  # observe the (day-ahead-known) price this many 15-min steps ahead
        # SOC thresholds
        # limit_penalty_low_thresh: float = 0.03,
        # limit_penalty_high_thresh: float = 0.97,
        self_consumption_coeff: float = 0.5,
        eps: float = 1e-8,
        debug: bool = False,

        daily_import_limit_kwh: float = 50.0,
        daily_export_limit_kwh: float = 20.0,
        # fine model (choose either per_kwh_fine > 0 or fixed_daily_fine > 0)
        per_kwh_fine_eur: float = 0.50,
        fixed_daily_fine_eur: float = 0.0,
        # when to consider day boundary detection (use "UTC" or "local" if microgrid.current_date is timezone-aware)
        day_boundary_tz: str = "local",
    ) -> None:
        super().__init__()

        # accept Microgrid instance or config object/dict
        if isinstance(config, Microgrid):
            self.microgrid = config
        else:
            self.microgrid = Microgrid(MicrogridConfig.parse_obj(config))

        # action/obs spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        obs_low = np.asarray(
            [
                0.0,
                float(self.microgrid.load.load_ts.min()),
                float(self.microgrid.pv.pv_production_ts.min()),
                float(self.microgrid.grid.import_prices.min()),
                float(self.microgrid.grid.export_prices.min()),
                # 0.0,
                # 0.0,
            ],
            dtype=np.float32,
        )
        obs_high = np.asarray(
            [
                1.0,
                float(self.microgrid.load.load_ts.max()),
                float(self.microgrid.pv.pv_production_ts.max()),
                float(self.microgrid.grid.import_prices.max()),
                float(self.microgrid.grid.export_prices.max()),
                # 1.0,
                # 1.0,
            ],
            dtype=np.float32,
        )
        if len(price_lookahead_steps):
            obs_low = np.concatenate([obs_low, np.full(len(price_lookahead_steps), -10.0)]).astype(np.float32)
            obs_high = np.concatenate([obs_high, np.full(len(price_lookahead_steps), 10.0)]).astype(np.float32)
        if time_features:
            obs_low = np.concatenate([obs_low, [-1.0, -1.0]]).astype(np.float32)
            obs_high = np.concatenate([obs_high, [1.0, 1.0]]).astype(np.float32)
        if obs_norm is not None:
            obs_low = np.full_like(obs_low, -np.inf)
            obs_high = np.full_like(obs_high, np.inf)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # params
        self.charge_from_grid_penalty_coeff = float(charge_from_grid_penalty_coeff)
        self.discharge_to_grid_penalty_coeff = float(discharge_to_grid_penalty_coeff)
        self.financial_coeff = float(financial_coeff)
        self.import_coeff = float(import_coeff)
        self.export_coeff = float(export_coeff)
        self.limit_action_penalty_coeff = float(limit_action_penalty_coeff)
        self.price_coeff = float(price_coeff)
        self.export_pv_coeff = float(export_pv_coeff)
        # self.limit_penalty_low_thresh = float(limit_penalty_low_thresh)
        # self.limit_penalty_high_thresh = float(limit_penalty_high_thresh)
        #
        # self.pv_spike_charge_coeff = float(pv_spike_charge_coeff)
        # self.pv_spike_threshold = float(pv_spike_threshold)

        self.load_discharge_coeff = float(load_discharge_coeff)
        self.load_surplus_threshold = float(load_surplus_threshold)
        self.conditional_grid_penalties = bool(conditional_grid_penalties)
        self.time_features = bool(time_features)
        self.obs_norm = None if obs_norm is None else (np.asarray(obs_norm[0], dtype=np.float32),
                                                       np.asarray(obs_norm[1], dtype=np.float32))
        self.reward_scale = float(reward_scale)
        self.random_start = bool(random_start)
        self.episode_len = None if episode_len is None else int(episode_len)
        self._ep_steps = 0
        self.rel_price = bool(rel_price)
        self.price_lookahead_steps = tuple(int(k) for k in price_lookahead_steps)
        if self.rel_price or self.price_lookahead_steps:
            import pandas as _pd0
            _p0 = _pd0.Series(np.asarray(self.microgrid.grid.import_prices, dtype=float))
            self._price_arr = _p0.to_numpy()
            self._ma_day = _p0.rolling(96, min_periods=1).mean().to_numpy()
            self._ma_week = _p0.rolling(672, min_periods=1).mean().to_numpy()
        self.self_consumption_coeff = float(self_consumption_coeff)

        self.eps = float(eps)
        self.debug = bool(debug)

        self.daily_import_limit_kwh = float(daily_import_limit_kwh)
        self.daily_export_limit_kwh = float(daily_export_limit_kwh)
        self.per_kwh_fine_eur = float(per_kwh_fine_eur)
        self.fixed_daily_fine_eur = float(fixed_daily_fine_eur)
        self.day_boundary_tz = str(day_boundary_tz)

        # logs expected by other helpers
        self.logs = {
            "soc": [],
            "grid_import_kwh": [],
            "grid_export_kwh": [],
            "load_kwh": [],
            "pv_kwh": [],
            "import_price": [],
            "export_price": [],
            "requested_action_norm": [],
            "battery_applied_raw": [],
            "charge_from_grid_penalty": [],
            "discharge_to_grid_penalty": [],
            "limit_action_penalty": [],
            "price_penalty": [],
            "pv_spike_charge_bonus": [],
            "load_discharge_bonus": [],
            "self_consumption": [],
            "financial_scaled": [],
            "daily_import_accum_kwh": [],
            "daily_export_accum_kwh": [],
            "daily_fine_eur": [],
        }



        self._daily_import_accum = 0.0
        self._daily_export_accum = 0.0
        # store current day for reset detection
        # try to derive a 'date' object from microgrid.current_date
        cd = getattr(self.microgrid, "current_date", None)
        try:
            if isinstance(cd, datetime.datetime):
                self._daily_date = cd.date()
            elif isinstance(cd, datetime.date):
                self._daily_date = cd
            else:
                self._daily_date = datetime.datetime.fromisoformat(str(cd)).date()
        except Exception:
            self._daily_date = datetime.datetime.utcnow().date()



    # ---------- observation ----------
    def _get_obs(self) -> np.ndarray:
        soc = float(self.microgrid.battery.state_of_charge)
        t_next = int(self.microgrid.t + 1)
        t_next = min(max(0, t_next), self.microgrid.MAX_TIMESTEP - 1)

        load_kw = float(self.microgrid.load.get_load(t_next))
        pv_kw = float(self.microgrid.pv.get_power(t_next))
        import_price = float(self.microgrid.grid.import_prices[t_next])
        export_price = float(self.microgrid.grid.export_prices[t_next])

        # robust weekday/month extraction
        current_date = getattr(self.microgrid, "current_date", None)
        dt = None
        if isinstance(current_date, datetime.datetime):
            dt = current_date
        elif isinstance(current_date, datetime.date):
            dt = datetime.datetime.combine(current_date, datetime.time())
        else:
            try:
                y = int(getattr(current_date, "year"))
                m = int(getattr(current_date, "month"))
                d = int(getattr(current_date, "day"))
                dt = datetime.datetime(y, m, d)
            except Exception:
                try:
                    dt = datetime.datetime.fromisoformat(str(current_date))
                except Exception:
                    try:
                        dt = current_date.to_pydatetime()  # type: ignore
                    except Exception:
                        dt = datetime.datetime.utcnow()


        if dt.weekday() < 5:
            weekday_norm = 1.0
        else:
            weekday_norm = 0.0
        if dt.month in (7, 9):
            month_norm = 1.0
        else:
            month_norm = 0.0


        if self.rel_price:
            eps_p = 1e-3
            r_day = import_price / (self._ma_day[t_next] + eps_p) - 1.0
            r_week = import_price / (self._ma_week[t_next] + eps_p) - 1.0
            base = np.asarray([soc, load_kw, pv_kw, np.clip(r_day, -1.0, 4.0), np.clip(r_week, -1.0, 4.0)],
                              dtype=np.float32)
        else:
            base = np.asarray([soc, load_kw, pv_kw, import_price, export_price], dtype=np.float32)
        if self.obs_norm is not None:
            base = (base - self.obs_norm[0]) / self.obs_norm[1]
        if self.price_lookahead_steps:
            n = len(self._price_arr)
            eps_p = 1e-3
            la = []
            for k in self.price_lookahead_steps:
                pk = float(self._price_arr[min(t_next + k, n - 1)])
                v = (pk / (self._ma_day[t_next] + eps_p) - 1.0) if self.rel_price else pk
                if self.obs_norm is not None:
                    v = (v - self.obs_norm[0][3]) / self.obs_norm[1][3]
                la.append(v)
            base = np.concatenate([base, np.asarray(la, dtype=np.float32)])
        if self.time_features:
            import pandas as _pd
            ts = _pd.Timestamp(self.microgrid.current_date.get_date(t_next))  # stored as epoch-ns ints
            hour = float(ts.hour) + float(ts.minute) / 60.0
            ang = 2.0 * np.pi * hour / 24.0
            base = np.concatenate([base, np.asarray([np.sin(ang), np.cos(ang)], dtype=np.float32)])
        return base.astype(np.float32)

    # ---------- step ----------
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # clamp and record requested action
        a = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
        self.logs.setdefault("requested_action_norm", []).append(float(a))

        # Compute PV surplus now (use current timestep power * delta_t if available)
        # t_now = int(self.microgrid.t)
        # try:
        #     pv_power_now_kw = float(self.microgrid.pv.get_power(t_now))
        #     load_power_now_kw = float(self.microgrid.load.get_load(t_now))
        #     price_now = float(self.microgrid.grid.get_price(t_now))
        #     delta_t = float(getattr(self.microgrid, "delta_t", 1.0))
        #     pv_now_kwh = pv_power_now_kw * delta_t
        #     print("pv_power_now_kw", pv_power_now_kw)
        #     load_now_kwh = load_power_now_kw * delta_t
        # except Exception:
        #     pv_now_kwh = 0.0
        #     load_now_kwh = 0.0
        #     price_now = 0.0
        #
        # pv_surplus_now = max(0.0, pv_now_kwh - load_now_kwh)
        # load_surplus_now = max(0.0, load_now_kwh - pv_now_kwh)

        # pass action directly to microgrid; microgrid will apply/clamp as needed
        obs_micro, done_flag = self.microgrid.run_timestep(a, logging=True)

        # retrieve energies
        energies = getattr(self.microgrid, "energies", {})
        try:
            applied_battery_kwh_raw = float(energies["battery"][-1])
            grid_kwh = float(energies["grid"][-1])
            pv_kwh = float(energies["pv"][-1])
            load_kwh = float(energies["load"][-1])
            price = float(energies["price"][-1])
        except Exception:
            t_now2 = int(self.microgrid.t)
            load_kwh = float(self.microgrid.load.get_load(t_now2) * getattr(self.microgrid, "delta_t", 1.0))
            pv_kwh = float(self.microgrid.pv.get_power(t_now2) * getattr(self.microgrid, "delta_t", 1.0))
            price = float(self.microgrid.grid.get_price(t_now2))
            applied_battery_kwh_raw = 0.0
            grid_kwh = load_kwh - pv_kwh + applied_battery_kwh_raw

        imported_kwh = max(grid_kwh, 0.0)
        exported_kwh = max(-grid_kwh, 0.0)

        load_surplus_now = max(0.0, load_kwh - pv_kwh)

        # prices & financial
        try:
            import_price = float(self.microgrid.grid.import_prices[int(self.microgrid.t)])
            export_price = float(self.microgrid.grid.export_prices[int(self.microgrid.t)])
        except Exception:
            import_price = export_price = 0.0

        # ------------------------


        # ------------------------

        #
        # financial = self.microgrid.grid.get_cost(int(self.microgrid.t), -grid_kwh)
        # financial_scaled = financial * self.financial_coeff
        import_cost = imported_kwh * float(import_price)
        # Reward for exporting this timestep (positive number)
        export_gain = exported_kwh * float(export_price)

        # scale by financial_coeff (keeps the previous scaling behavior)
        import_penalty = float(self.import_coeff) * float(import_cost)  # to be subtracted from reward
        export_reward = float(self.export_coeff) * float(export_gain)  # to be added to reward

        # interpret applied battery sign (+ charge, - discharge)
        applied = float(applied_battery_kwh_raw)
        charge_kwh = max(0.0, applied)
        discharge_kwh = max(0.0, -applied)

        # penalties
        charge_from_grid_penalty = 0
        discharge_to_grid_penalty = 0

        charge_from_grid_needed = pv_kwh < load_kwh    # a real deficit exists this step
        discharge_to_grid_needed = pv_kwh >= load_kwh  # a real surplus exists this step

        if imported_kwh > 0 and applied_battery_kwh_raw > self.eps:
            if not self.conditional_grid_penalties or not charge_from_grid_needed:
                charge_from_grid_penalty = float(self.charge_from_grid_penalty_coeff) * float(imported_kwh)
        if exported_kwh > 0 and applied_battery_kwh_raw < -self.eps:
            if not self.conditional_grid_penalties or not discharge_to_grid_needed:
                discharge_to_grid_penalty = float(self.discharge_to_grid_penalty_coeff) * float(exported_kwh)

        # limit-action penalty (based on agent choice a): penalize requesting to
        # discharge/charge further when the battery's *current* stored energy is
        # already at the low/high capacity boundary (a boundary check must use the
        # battery's absolute energy level, not the applied delta).
        soc = float(self.microgrid.battery.state_of_charge)
        battery_energy_now = float(self.microgrid.battery.energy)
        limit_action_penalty = 0.0
        if a < -self.eps and battery_energy_now - self.microgrid.battery.low_capacity <= abs(self.eps):
            limit_action_penalty = float(self.limit_action_penalty_coeff) * float(abs(a))
        elif a > self.eps and self.microgrid.battery.high_capacity - battery_energy_now <= abs(self.eps):
            limit_action_penalty = float(self.limit_action_penalty_coeff) * float(abs(a))
        # print("Limit action penalty: %.2f" % limit_action_penalty)
        # PV-spike charge bonus:
        # Award only when:
        #  - pv_surplus_now >= threshold (spike),
        #  - agent chose to charge (a > 0),
        #  - and some charged energy could have come from PV (charge_from_pv_kwh > 0)
        # charge_from_pv_kwh = min(max(0.0, pv_kwh - load_kwh), applied_battery_kwh_raw)
        # pv_spike_charge_bonus = 0.0
        # if (pv_surplus_now >= float(self.pv_spike_threshold) and a > 0.0
        #         and applied_battery_kwh_raw - self.microgrid.battery.high_capacity >= abs(self.eps)):\
        #
        #     pv_spike_charge_bonus = float(self.pv_spike_charge_coeff) * float(pv_surplus_now)
        #
        # print("Battery", applied_battery_kwh_raw)
        #
        # print("Pv spike bonus:", pv_spike_charge_bonus)
        # LOAD-surplus discharge bonus:
        # Award when:
        #  - load_surplus_now >= threshold (demand spike),
        #  - agent chose to discharge (a < 0),
        #  - and some discharged energy actually served the load (discharge_to_load_kwh > 0)
        # discharge_to_load_kwh = min(max(0.0, load_kwh - pv_kwh), discharge_kwh)
        load_discharge_bonus = 0.0
        if (load_surplus_now >= float(self.load_surplus_threshold) and a < 0.0
            and battery_energy_now - self.microgrid.battery.low_capacity >= abs(self.eps)): \
            load_discharge_bonus = float(self.load_discharge_coeff) * float(load_surplus_now)

        # limit_grid_penalty = 0.0
        # if abs(grid_kwh) > 3.0:
        #     limit_grid_penalty = 100 * grid_kwh

        # print("Limit grid penalty:", limit_grid_penalty)

        # Self-consumption ratio: fraction of PV generation used locally rather than
        # exported. Export beyond pv_kwh must have come from the battery, not this
        # step's PV, so we attribute at most pv_kwh of the export to "unconsumed PV"
        # to keep the ratio in [0, 1] (the raw pv_kwh - exported_kwh formula could
        # otherwise go negative whenever the battery discharges into export).
        self_consumption = (
            self.self_consumption_coeff * (pv_kwh - min(exported_kwh, pv_kwh)) / pv_kwh
            if pv_kwh > 0
            else 0.0
        )

        price_penalty = 0.0
        if price < 0.2 and exported_kwh > 0.0 and applied_battery_kwh_raw < 8.0:
            price_penalty = exported_kwh / float(export_price) * self.price_coeff
        if price > 0.2 and imported_kwh > 0.0 and applied_battery_kwh_raw > 2.0:
            price_penalty = imported_kwh / float(export_price) * self.price_coeff

        export_penalty_pv = 0.0
        soc = float(self.microgrid.battery.state_of_charge)

        pv_surplus_now = max(0.0, pv_kwh - load_kwh)
        if pv_surplus_now > 0.0 and soc < 0.85:
            export_penalty_pv = self.export_pv_coeff * exported_kwh

        # final reward (simple)
        reward = float(
            export_reward
            - import_penalty
            # financial_scaled
            # grid_kwh
            - charge_from_grid_penalty
            - discharge_to_grid_penalty
            # - price_penalty
            - limit_action_penalty
            - export_penalty_pv
            # + pv_spike_charge_bonus
            # + load_discharge_bonus
            # - limit_grid_penalty
            # - fine_this_step
            # + self_consumption
        )

        # logging
        self.logs.setdefault("soc", []).append(float(soc))
        self.logs.setdefault("grid_import_kwh", []).append(float(imported_kwh))
        self.logs.setdefault("grid_export_kwh", []).append(float(exported_kwh))
        self.logs.setdefault("pv_kwh", []).append(float(pv_kwh))
        self.logs.setdefault("load_kwh", []).append(float(load_kwh))
        self.logs.setdefault("price", []).append(float(price))
        self.logs.setdefault("battery_applied_raw", []).append(float(applied_battery_kwh_raw))
        self.logs.setdefault("charge_from_grid_penalty", []).append(float(charge_from_grid_penalty))
        self.logs.setdefault("discharge_to_grid_penalty", []).append(float(discharge_to_grid_penalty))
        # self.logs.setdefault("limit_action_penalty", []).append(float(limit_action_penalty))
        # self.logs.setdefault("pv_spike_charge_bonus", []).append(float(pv_spike_charge_bonus))
        self.logs.setdefault("self_consumption", []).append(float(self_consumption))
        self.logs.setdefault("load_discharge_bonus", []).append(float(load_discharge_bonus))
        self.logs.setdefault("import_penalty", []).append(float(import_penalty))
        self.logs.setdefault("export_reward", []).append(float(export_reward))
        self.logs.setdefault("price_penalty", []).append(float(price_penalty))
        self.logs.setdefault("export_penalty_pv", []).append(float(export_penalty_pv))
        # self.logs.setdefault("financial_scaled", []).append(float(financial_scaled))
        # self.logs.setdefault("fine_this_step", []).append(float(fine_this_step))

        self.logs.setdefault("daily_import_accum_kwh", []).append(float(self._daily_import_accum))
        self.logs.setdefault("daily_export_accum_kwh", []).append(float(self._daily_export_accum))




        if self.debug:
            print(
                f"t={int(self.microgrid.t)} soc={soc:.3f} a={a:.3f} "
                f"load_surplus_now={load_surplus_now:.4f} "
                f"load_bonus={load_discharge_bonus:.3f}"
            )

        info = {
            "t": int(self.microgrid.t),
            "energies_kwh": {
                "battery_applied_raw": float(applied_battery_kwh_raw),
                "charge_kwh": float(charge_kwh),
                "discharge_kwh": float(discharge_kwh),
                "grid": float(grid_kwh),
                "pv": float(pv_kwh),
                "load": float(load_kwh),
                "imported_kwh": float(imported_kwh),
                "exported_kwh": float(exported_kwh),
                # "charge_from_pv_kwh": float(charge_from_pv_kwh),
                # "discharge_to_load_kwh": float(discharge_to_load_kwh),
            },
            "prices": {"import_price": import_price, "export_price": export_price},
            "reward_components": {
                # "financial_scaled": float(financial_scaled),
                "import_penalty": float(import_penalty),
                "export_reward": float(export_reward),
                "charge_from_grid_penalty": float(charge_from_grid_penalty),
                "discharge_to_grid_penalty": float(discharge_to_grid_penalty),
                # "limit_action_penalty": float(limit_action_penalty),
                "load_discharge_bonus": float(load_discharge_bonus),
                "self_consumption": float(self_consumption),
                # "fine_this_step": float(fine_this_step),
                "export_penalty_pv": float(export_penalty_pv),
            },
            "action_sent": {"battery_norm": float(a)},
        }

        # info.setdefault("daily_fines", 0.0)
        # info["daily_fines"] = float(fine_this_step)
        # info.setdefault("daily_import_accum_kwh", 0.0)
        # info["daily_import_accum_kwh"] = float(self._daily_import_accum)
        # info.setdefault("daily_export_accum_kwh", 0.0)
        # info["daily_export_accum_kwh"] = float(self._daily_export_accum)

        obs_out = self._get_obs()
        terminated = False
        self._ep_steps += 1
        truncated = bool(done_flag) or (self.episode_len is not None and self._ep_steps >= self.episode_len)
        return obs_out, reward * self.reward_scale, terminated, truncated, info

    # ---------- reset ----------
    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            import random

            random.seed(int(seed))
            np.random.seed(int(seed))
            try:
                self.action_space.seed(int(seed))
            except Exception:
                pass

        try:
            _ = self.microgrid.reset(reset_logs=kwargs.get("reset_logs", False))
        except TypeError:
            _ = self.microgrid.reset()

        # clear logs
        for k in self.logs:
            try:
                self.logs[k].clear()
            except Exception:
                self.logs[k] = []

        self._ep_steps = 0
        if self.random_start:
            horizon = int(self.episode_len or 672)
            hi = max(int(self.microgrid.MAX_TIMESTEP) - horizon - 2, 1)
            self.microgrid.t = int(np.random.randint(0, hi))
        obs_out = self._get_obs()
        return obs_out, {}

    def render(self, mode: str = "human"):
        soc = float(self.microgrid.battery.state_of_charge)
        t = int(self.microgrid.t)
        print(f"t={t:4d}  soc={soc:.3f}")

    def step_rule_based(self) -> Tuple[np.ndarray, bool, bool, Dict]:
        obs_micro, done_flag = self.microgrid.run_timestep_rule_based(logging=True)

        energies = getattr(self.microgrid, "energies", {})
        try:
            applied_battery_kwh_raw = float(energies["battery"][-1])
            grid_kwh = float(energies["grid"][-1])
            pv_kwh = float(energies["pv"][-1])
            load_kwh = float(energies["load"][-1])
            price = float(energies["price"][-1])
        except Exception:
            t_now2 = int(self.microgrid.t)
            load_kwh = float(self.microgrid.load.get_load(t_now2) * getattr(self.microgrid, "delta_t", 1.0))
            pv_kwh = float(self.microgrid.pv.get_power(t_now2) * getattr(self.microgrid, "delta_t", 1.0))
            price = float(self.microgrid.grid.get_price(t_now2))
            applied_battery_kwh_raw = 0.0
            grid_kwh = load_kwh - pv_kwh + applied_battery_kwh_raw

        imported_kwh = max(grid_kwh, 0.0)
        exported_kwh = max(-grid_kwh, 0.0)

        load_surplus_now = max(0.0, load_kwh - pv_kwh)

        # prices & financial
        try:
            import_price = float(self.microgrid.grid.import_prices[int(self.microgrid.t)])
            export_price = float(self.microgrid.grid.export_prices[int(self.microgrid.t)])
        except Exception:
            import_price = export_price = 0.0

        soc = float(self.microgrid.battery.state_of_charge)

        applied = float(applied_battery_kwh_raw)
        charge_kwh = max(0.0, applied)
        discharge_kwh = max(0.0, -applied)

        self.logs.setdefault("soc", []).append(float(soc))
        self.logs.setdefault("grid_import_kwh", []).append(float(imported_kwh))
        self.logs.setdefault("grid_export_kwh", []).append(float(exported_kwh))
        self.logs.setdefault("pv_kwh", []).append(float(pv_kwh))
        self.logs.setdefault("load_kwh", []).append(float(load_kwh))
        self.logs.setdefault("price", []).append(float(price))
        self.logs.setdefault("battery_applied_raw", []).append(float(applied_battery_kwh_raw))

        info = {
            "t": int(self.microgrid.t),
            "energies_kwh": {
                "battery_applied_raw": float(applied_battery_kwh_raw),
                "charge_kwh": float(charge_kwh),
                "discharge_kwh": float(discharge_kwh),
                "grid": float(grid_kwh),
                "pv": float(pv_kwh),
                "load": float(load_kwh),
                "imported_kwh": float(imported_kwh),
                "exported_kwh": float(exported_kwh),
                # "charge_from_pv_kwh": float(charge_from_pv_kwh),
                # "discharge_to_load_kwh": float(discharge_to_load_kwh),
            },
            "prices": {"import_price": import_price, "export_price": export_price},
            "action_sent": {
                "battery_norm": (
                    applied_battery_kwh_raw / self.microgrid.max_actions
                    if self.microgrid.max_actions
                    else 0.0
                )
            },
        }

        obs_out = self._get_obs()
        terminated = False
        truncated = bool(done_flag)
        return obs_out, terminated, truncated, info

