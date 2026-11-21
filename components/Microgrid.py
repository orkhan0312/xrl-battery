from datetime import datetime
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from components.Date import Date
from components.Grid import Grid
from components.Load import Load
from components.PV import PV
from components.Storage import Battery
from config.types import MicrogridConfig


class Microgrid:
    def __init__(self, config: MicrogridConfig) -> None:
        self.battery = Battery(config.battery)
        self.grid = Grid(config.grid)
        self.pv = PV(config.pv)
        self.load = Load(config.load)
        self.current_date = Date(config.current_date)
        self.weekday = Date(config.weekday)
        # self.month = month

        self.overproduction_penalty = config.overprod_penalty
        self.underproduction_penalty = config.underprod_penalty
        self.MAX_TIMESTEP = config.max_timestep

        self.t = 0
        self.delta_t = 1

        self._init_logs_()

        if self.grid.__len__ != self.MAX_TIMESTEP:
            raise ValueError(
                f"Prices timeseries lengths are different ({self.grid.__len__ }) with \
                    the maximum number of timesteps ({self.MAX_TIMESTEP})"
            )
        if self.pv.__len__ != self.MAX_TIMESTEP:
            raise ValueError(
                f"PV production timeseries length is different ({self.pv.__len__ }) \
                    with the maximum number of timesteps ({self.MAX_TIMESTEP})"
            )

        if self.load.__len__ != self.MAX_TIMESTEP:
            raise ValueError(
                f"PV production timeseries length is different ({self.pv.__len__ }) \
                    with the maximum number of timesteps ({self.MAX_TIMESTEP})"
            )

    @property
    def config(self) -> MicrogridConfig:
        config = MicrogridConfig.parse_obj(
            {
                "battery": self.battery.config,
                "grid": self.grid.config,
                "load": self.load.config,
                "pv": self.pv.config,
                "current_date": self.current_date.config,
                "weekday": self.weekday.config,
                "overprod_penalty": self.overproduction_penalty,
                "underprod_penalty": self.underproduction_penalty,
                "max_timestep": self.MAX_TIMESTEP,
            }
        )
        return config

    @property
    def __len__(self):
        return min(self.load.__len__, self.pv.__len__, self.grid.__len__)

    def scale_action(self, action: float, max_val: float, min_val: float) -> float:
        """
        Scale the action from [-1, 1] to the actual [min_val, max_val]

        Args:
            action (float): The value of the action to scale in [-1, 1]
            max_val (float): The max val of the corresponding action
            min_val (float): The min val of the corresponding action

        Returns:
            float: The action scaled back to [min_val, max_val]
        """
        # map [-1,1] -> [min_val, max_val]
        return ((float(action) + 1.0) * 0.5 * (max_val - min_val)) + min_val


    def run_timestep(
        self, action: Union[float, np.ndarray], logging: bool = True
    ) -> Tuple[np.ndarray, bool]:
        """
        Executes one scalar action on the microgrid and computes the following state.

        Now there is a single continuous action in [-1, 1] which controls only the
        battery (charge + / discharge -). The grid automatically buys/sells whatever
        remains to satisfy the load (i.e. grid balances residual after PV and battery).

        Args:
            action (float or np.ndarray): Scalar action in [-1, 1] controlling battery.

        Returns:
            Tuple[np.ndarray, bool, Tuple[float, float, float]]:
                (next_observation, done_flag, (overcharge_cost, grid_cost, error_cost))
        """
        # convert possible 1-element np.array to scalar
        try:
            action_scalar = float(np.asarray(action).item())
        except Exception:
            action_scalar = float(action)

        self.t += 1

        # print("Action scalar", action_scalar)
        # print(self.battery.energy)


        energy_pv = self.pv.get_power(self.t) * self.delta_t
        energy_load = self.load.get_load(self.t) * self.delta_t
        price = self.grid.get_price(self.t) * self.delta_t

        # --- determine requested energy (kWh) from normalized action ---
        requested_energy_batt = self.scale_action(action_scalar, self.max_actions, self.min_actions)
        # Enforce per-step physical availability:
        # max discharge available (kWh) = energy above low_capacity
        max_discharge_available = float(self.battery.energy - self.battery.low_capacity)
        # max charge available (kWh) = headroom to high_capacity
        max_charge_available = float(self.battery.high_capacity - self.battery.energy)

        # also respect per-timestep power limit (self.max_actions is already kWh per timestep)
        per_step_limit = float(self.max_actions)
        # For discharge requested (negative), allowed lower bound is -min(per_step_limit, max_discharge_available)
        min_allowed = -min(per_step_limit, max(0.0, max_discharge_available))
        # For charge requested (positive), allowed upper bound is min(per_step_limit, max_charge_available)
        max_allowed = min(per_step_limit, max(0.0, max_charge_available))

        # Clip request to feasible bounds (so agent can't request more than physically possible)
        energy_battery = float(max(min(requested_energy_batt, max_allowed), min_allowed))

        # store battery energy before applying to compute actual applied delta
        energy_before = float(self.battery._energy)

        # Apply battery change (clamps internally to capacity bounds and applies
        # charge/discharge efficiency; returns the realized AC-side energy, which is
        # what the grid balance below must use -- it can differ from the internal
        # stored-energy delta whenever eta_c/eta_d < 1).
        applied_energy_batt = self.battery.charge_discharge(energy_battery)

        # Now compute grid residual based on *applied* battery energy (NOT the original requested value)
        energy_grid = energy_load + applied_energy_batt - energy_pv

        # energy balance (should be zero using applied)
        energy_balance = energy_pv + energy_grid - energy_load - applied_energy_batt

        # Logging: log applied_energy_batt (the energy actually moved) not the requested one
        if logging:
            self.log_energies(
                pd.to_datetime(self.current_date.get_date(self.t)), applied_energy_batt, energy_grid, energy_pv, energy_load, price
            )

        error_cost = self.get_error_cost(energy_balance)

        return self.obs, self.done

    def get_error_cost(self, energy_balance: float) -> float:
        """
        Compute the cost for not providing the right amount of energy.
        Either too much or not enough.

        Args:
            energy_balance (float): the energy balance (overprod:+, underprod:-)

        Returns:
            float: The cost ($) for not meeting the requirement
        """
        if energy_balance >= 0:
            return energy_balance * self.overproduction_penalty
        else:
            return energy_balance * self.underproduction_penalty

    def _init_logs_(self):
        """
        Initialize lists for logging energies and costs.
        """
        self.energies = {"current_date": [], "battery": [], "grid": [], "pv": [], "load": [], "price": []}
        self.costs = {"total": [], "overcharge": [], "grid": [], "error": []}

    def log_energies(
        self, current_date:datetime, soc:float, grid:float, pv: float, load: float, price:float
    ):
        """
        Log the different energies collected at the current timestep
        """
        # pylint: disable=too-many-arguments
        # if balance is None:
        #     balance = pv + grid - load - battery
        self.energies["current_date"].append(current_date)
        self.energies["battery"].append(soc)
        # self.energies["battery"].append(battery)
        self.energies["grid"].append(grid)
        self.energies["pv"].append(pv)
        self.energies["load"].append(load)
        self.energies["price"].append(price)
        # self.energies["balance"].append(balance)

    def log_costs(self, overcharge: float, grid: float, error: float):
        """
        Log the different costs collected at the current timestep
        """
        self.costs["overcharge"].append(overcharge)
        self.costs["grid"].append(grid)
        self.costs["error"].append(error)
        self.costs["total"].append(overcharge + grid + error)

    def get_logs(self) -> dict:
        """
        Return the logs in a dict format for energies and costs
        """
        # return {"costs": self.costs, "energies": self.energies}
        return {"energies": self.energies}

    def show_logs(self, show=True):
        """
        Plot the available logs using datetime as x-axis.
        """
        figures = []

        for data_name, data in self.get_logs().items():

            # first item is datetime
            datetime_key = list(data.keys())[0]
            x = data[datetime_key]

            # other variables to plot
            keys = list(data.keys())[1:]

            fig, ax = plt.subplots(len(keys), 1, figsize=(8, 6), sharex=True)

            if len(keys) == 1:
                ax = [ax]

            for i, name in enumerate(keys):
                points = data[name]
                ax[i].plot(x, points)
                ax[i].set_title(name.capitalize())

            plt.suptitle(data_name.capitalize())
            plt.tight_layout()

            if show:
                plt.show()
            else:
                figures.append(fig)

        if not show:
            return figures

    @property
    def obs(self) -> np.ndarray:
        """
        Returns the current observation of the microgrid (unchanged structure).
        """
        return np.array(
            [
                self.battery.state_of_charge,
                self.load.get_load(self.t),
                self.pv.get_power(self.t),
                self.grid.import_prices[self.t],
                self.grid.export_prices[self.t],
            ],
            dtype=np.float32,
        )

    @property
    def max_values(self) -> np.ndarray:
        """
        Get the max possible values for all observations
        """
        return np.array(
            [
                1.0,
                self.grid.import_prices.max(),
                self.grid.export_prices.max(),
                self.load.load_ts.max(),
                self.pv.pv_production_ts.max(),
            ],
            dtype=np.float32,
        )

    @property
    def min_values(self) -> np.ndarray:
        """
        Get the min possible values for all observations
        """
        return np.array(
            [
                0.0,
                self.grid.import_prices.min(),
                self.grid.export_prices.min(),
                self.load.load_ts.min(),
                self.pv.pv_production_ts.min(),
            ],
            dtype=np.float32,
        )

    @property
    def max_actions(self) -> float:
        """
        Max allowed energy for the single action (battery) in one timestep.

        Use battery.max_output (kW) times delta_t (hours) to produce kWh per timestep,
        which matches what Battery.charge_discharge expects.
        """
        # Prefer using per-timestep energy based on battery max_output (kW). If that attribute
        # is missing or zero, fall back to battery.capacity to avoid division by zero issues.
        try:
            max_out_kw = float(getattr(self.battery, "max_output", 0.0))
            if max_out_kw > 0:
                return max_out_kw * float(self.delta_t)  # kW * h = kWh
        except Exception:
            pass
        # Fallback: use capacity (kWh) — not ideal but keeps backward compatibility
        return float(self.battery.capacity)

    @property
    def min_actions(self) -> float:
        """
        Min allowed energy for the single action (battery) in one timestep (negative).
        """
        return -float(self.max_actions)

    @property
    def done(self) -> bool:
        """
        Returns whether the microgrid is in a final state
        """
        return self.t >= self.MAX_TIMESTEP - 2

    def print_info(self):
        """
        TBD
        """
        raise NotImplementedError

    def reset(self, reset_logs=False) -> np.ndarray:
        """
        Reset the microgrid in its original state.
        """
        self.t = 0
        self.battery.reset()
        if reset_logs:
            self._init_logs_()
        return self.obs

    def set_battery_from_duration(self, nb_of_hours: float) -> None:
        """
        Set the battery config to handle a given number of hours under mean load
        """
        new_capacity = self.load.__mean__ * nb_of_hours
        self.battery.initial_energy = (
            new_capacity / self.battery.capacity
        ) * self.battery.initial_energy
        self.battery.capacity = new_capacity
        self.battery.reset()
        self.battery.max_output = max(self.battery.max_output, self.load.__mean__)

    def save_config(self, file_name="./config.json"):
        """
        Save the current config to a json file.
        """
        with open(file_name, "w", encoding="utf-8") as json_file:
            json_file.write(self.config.json(indent=4, sort_keys=True))

    def run_timestep_rule_based(
            self, logging: bool = True
    ) -> Tuple[np.ndarray, bool]:
        """
        Executes one scalar action on the microgrid and computes the following state.

        Now the environment ignores the RL agent action and uses a simple rule-based
        controller for the battery:

          - If PV generation (kWh) >= load (kWh): charge the battery with the excess
            (up to per-step power limit and remaining battery headroom). Any remaining
            excess is exported to the grid.
          - If PV generation < load: discharge the battery to cover the deficit
            (up to per-step power limit and available battery energy). Any remaining
            deficit is imported from the grid.

        The rest of the environment (grid balancing, cost calculation, logging) is
        preserved from the previous implementation.

        Args:
            action (float or np.ndarray): Kept in the signature for compatibility but
                not used by the rule-based controller.

        Returns:
            Tuple[np.ndarray, bool, Tuple[float, float, float]]:
                (next_observation, done_flag, (overcharge_cost, grid_cost, error_cost))
        """
        # convert possible 1-element np.array to scalar (kept for API compatibility)
        # try:
        #     action_scalar = float(np.asarray(action).item())
        # except Exception:
        #     action_scalar = float(action)

        self.t += 1

        energy_pv = self.pv.get_power(self.t) * self.delta_t
        energy_load = self.load.get_load(self.t) * self.delta_t
        price = self.grid.get_price(self.t) * self.delta_t

        # --- RULE-BASED DECISION (replaces RL action) ---
        # Positive values => charge battery (kWh). Negative => discharge (kWh).
        # Compute physical availabilities and per-step limits (kWh).
        max_discharge_available = float(self.battery.energy - self.battery.low_capacity)
        max_charge_available = float(self.battery.high_capacity - self.battery.energy)
        per_step_limit = float(self.max_actions)

        # Default requested energy into battery (kWh)
        requested_energy_batt = 0.0

        if energy_pv >= energy_load:
            # excess PV available to charge battery (kWh)
            excess = energy_pv - energy_load
            # can only charge up to battery headroom and per-step power limit
            charge_possible = min(excess, per_step_limit, max(0.0, max_charge_available))
            requested_energy_batt = float(charge_possible)  # positive => charge
        else:
            # deficit: try to discharge battery to meet remaining load (kWh)
            deficit = energy_load - energy_pv
            # can only discharge up to available energy and per-step power limit
            discharge_possible = min(deficit, per_step_limit, max(0.0, max_discharge_available))
            requested_energy_batt = -float(discharge_possible)  # negative => discharge

        # Ensure requested is within physical bounds (defensive clipping)
        min_allowed = -min(per_step_limit, max(0.0, max_discharge_available))
        max_allowed = min(per_step_limit, max(0.0, max_charge_available))
        energy_battery = float(max(min(requested_energy_batt, max_allowed), min_allowed))

        # store battery energy before applying to compute actual applied delta
        energy_before = float(self.battery._energy)

        # Apply battery change (clamps internally and applies charge/discharge
        # efficiency; returns the realized AC-side energy delta -- this, not the
        # pre-step absolute energy level, is what "battery" should log, since
        # downstream code (reward terms, plots) treats it as the applied
        # charge(+)/discharge(-) amount for this step, not a SOC-like level.
        applied_energy_batt = self.battery.charge_discharge(energy_battery)

        # Now compute grid residual based on *applied* battery energy (NOT the original requested value)
        energy_grid = energy_load + applied_energy_batt - energy_pv

        if logging:
            self.log_energies(
                pd.to_datetime(self.current_date.get_date(self.t)), applied_energy_batt, energy_grid, energy_pv, energy_load,
                price
            )

        return self.obs, self.done