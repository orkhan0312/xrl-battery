# import math
#
# import numpy as np
# import pybamm
#
#
# class BatteryPybamm:
#     """
#     Interface to PyBaMM for stepwise simulation of battery dynamics and degradation.
#
#     You must select / supply a PyBaMM model and parameter set appropriate for your chemistry.
#     This wrapper keeps a "fast" state (SOC, capacity) and runs PyBaMM in small time steps.
#     NOTE: running PyBaMM per environment step can be computationally heavy; for RL you might
#     want to run a simplified surrogate or cache results.
#     """
#
#     def __init__(self,
#                  nominal_capacity_kwh: float,
#                  max_charge_kw: float,
#                  max_discharge_kw: float,
#                  efficiency: float = 0.95,
#                  dt_hours: float = 1/4,
#                  pybamm_model: str = "SPM",
#                  pybamm_parameter_values: dict = None):
#         """
#         nominal_capacity_kwh: initial energy capacity (kWh)
#         max_charge_kw / max_discharge_kw: power limits
#         efficiency: round-trip efficiency (simplified)
#         dt_hours: env timestep in hours
#         pybamm_model: which pybamm model to use; default "SPM" (single particle)
#         pybamm_parameter_values: dict or pybamm.ParameterValues-like object with parameter values
#         """
#         self.nominal_capacity_kwh = nominal_capacity_kwh
#         self.capacity_kwh = nominal_capacity_kwh  # degrade over time
#         self.max_charge_kw = max_charge_kw
#         self.max_discharge_kw = max_discharge_kw
#         self.efficiency = efficiency
#         self.dt_h = dt_hours
#         self.soc = 0.5  # start at 50% SOC by default
#         self.cycle_energy_throughput = 0.0  # for simple degradation accounting
#
#         # PyBaMM setup
#         self.pybamm_available = (pybamm is not None)
#         if self.pybamm_available:
#             # choose a model
#             if pybamm_model.upper() == "SPM":
#                 self.model = pybamm.lithium_ion.SPM()
#             elif pybamm_model.upper() == "DFN":
#                 self.model = pybamm.lithium_ion.DFN()
#             else:
#                 self.model = pybamm.lithium_ion.SPM()
#
#             # Parameter values: if provided as dict, convert
#             if pybamm_parameter_values is None:
#                 # TODO: choose a default parameter set here if you know chemistry; leaving as placeholder
#                 # Example: pvbamm.ParameterValues("Marquis2019") -- but parameter name depends on pybamm version
#                 # For now we'll use the default parameter values for the chosen model
#                 try:
#                     self.param = pybamm.ParameterValues(chemistry=None)
#                 except Exception:
#                     self.param = None
#             else:
#                 # if user passes a dict-like:
#                 try:
#                     self.param = pybamm.ParameterValues(pybamm_parameter_values)
#                 except Exception:
#                     self.param = pybamm.ParameterValues(pybamm_parameter_values)
#
#             # Create simulation object; we will re-instantiate solve each step if needed
#             try:
#                 self.sim = pybamm.Simulation(self.model, parameter_values=self.param)
#             except Exception:
#                 self.sim = None
#
#         else:
#             self.model = None
#             self.param = None
#             self.sim = None
#
#     def get_soc(self) -> float:
#         return float(self.soc)
#
#     def get_capacity_kwh(self) -> float:
#         return float(self.capacity_kwh)
#
#     def step(self, requested_power_kw: float) -> dict:
#         """
#         Apply requested battery power for one timestep (kW). Positive -> charge, Negative -> discharge.
#         Returns dictionary with:
#          - actual_power_kw (kW into battery, >0 charge)
#          - soc
#          - capacity_kwh (remaining capacity)
#          - energy_charged_kwh (>=0)
#          - energy_discharged_kwh (>=0)
#          - degradation (delta capacity in kWh)
#          - voltage (if available)
#         """
#         # Clip power to battery limits
#         if requested_power_kw >= 0:
#             actual_power_kw = min(requested_power_kw, self.max_charge_kw)
#             # convert to energy considering efficiency (charging loses energy)
#             energy_charged_kwh = actual_power_kw * self.dt_h * math.sqrt(self.efficiency)  # split efficiency evenly
#             energy_discharged_kwh = 0.0
#         else:
#             actual_power_kw = max(requested_power_kw, -self.max_discharge_kw)
#             energy_discharged_kwh = -actual_power_kw * self.dt_h * math.sqrt(self.efficiency)
#             energy_charged_kwh = 0.0
#
#         # Update SOC (simple energy balance)
#         # SOC defined relative to current capacity_kwh
#         energy_net = energy_charged_kwh - energy_discharged_kwh  # kWh added to battery
#         soc_change = 0.0
#         if self.capacity_kwh > 0:
#             soc_change = energy_net / self.capacity_kwh
#         self.soc = float(np.clip(self.soc + soc_change, 0.0, 1.0))
#
#         # Degradation modelling:
#         # Two parts:
#         #   1) physics-based via pybamm if available (more accurate)
#         #   2) simple throughput-based empirical model as fallback
#         delta_capacity_kwh = 0.0
#         voltage = None
#
#         if self.pybamm_available and self.sim is not None:
#             # Running a full pybamm simulation per env step can be slow.
#             # Here we show a template: set up experiment to apply the current for dt and simulate.
#             # WARNING: you should tune solver options and parameter set for performance.
#             try:
#                 # PyBaMM expects current in A; convert kW to A assuming nominal voltage or capacity
#                 # This is a sketch: exact mapping requires parameterization with battery nominal voltage and capacity.
#                 # We'll set current as a parameter "Current function" if model allows; else skip.
#                 # For simplicity, we skip running the heavy sim here but show where you'd call it.
#                 pass
#             except Exception as e:
#                 # fallback:
#                 pass
#
#         # Fallback degradation: simple throughput model (cycle life approximation).
#         # Example: assume that total lifetime throughput (kWh) until 1% capacity loss is "throughput_for_1pct"
#         # You will tune these numbers based on chemistry or pybamm output.
#         throughput_for_1pct_kwh = max(1e-6, 100000.0)  # placeholder: 100 MWh of throughput -> 1% loss (tune)
#         throughput = energy_charged_kwh + energy_discharged_kwh
#         self.cycle_energy_throughput += throughput
#         delta_capacity_kwh = - (throughput / throughput_for_1pct_kwh) * (self.nominal_capacity_kwh * 0.01)
#         # don't reduce capacity below small epsilon:
#         self.capacity_kwh = max(0.1 * self.nominal_capacity_kwh, self.capacity_kwh + delta_capacity_kwh)
#
#         return {
#             "actual_power_kw": actual_power_kw,
#             "soc": self.soc,
#             "capacity_kwh": self.capacity_kwh,
#             "energy_charged_kwh": energy_charged_kwh,
#             "energy_discharged_kwh": energy_discharged_kwh,
#             "delta_capacity_kwh": delta_capacity_kwh,
#             "voltage": voltage
#         }





#
# from typing import Optional
# from config.types import BatteryConfig
#
# from typing import Optional
# from config.types import BatteryConfig
#
#
# class Battery:
#     def __init__(self, battery_config: BatteryConfig) -> None:
#         self.capacity = float(battery_config.capacity)
#         # high_capacity and low_capacity are the allowed energy bounds (kWh)
#         self.high_capacity = float(battery_config.high_capacity)
#         self.low_capacity = float(battery_config.low_capacity)
#         # max/min output are kept for external logic (kW or kWh per timestep depending on usage)
#         self.max_output = float(battery_config.max_output)
#         self.min_output = float(battery_config.min_output)
#         # energy stored in kWh
#         self.initial_energy = float(battery_config.initial_energy)
#         # ensure _energy is clamped to bounds at start
#         self._energy = max(self.low_capacity, min(self.initial_energy, self.high_capacity))
#         # cost/penalty applied to any overflow/shortage returned by charge_discharge
#         self.overcharge_penalty = float(battery_config.overcharge_penalty)
#
#     def reset(self):
#         """Reset battery energy to the configured initial energy (clamped to bounds)."""
#         self._energy = max(self.low_capacity, min(self.initial_energy, self.high_capacity))
#
#     @property
#     def config(self) -> BatteryConfig:
#         return BatteryConfig.parse_obj(
#             {
#                 "capacity": self.capacity,
#                 "high_capacity": self.high_capacity,
#                 "low_capacity": self.low_capacity,
#                 "max_output": self.max_output,
#                 "min_output": self.min_output,
#                 "overcharge_penalty": self.overcharge_penalty,
#                 "initial_energy": self.initial_energy,
#             }
#         )
#
#     @property
#     def state_of_charge(self) -> float:
#         """
#         Return the state of charge in [0, 1].
#
#         Note: capacity is expected > 0. We clip to [0,1] defensively and assert to catch logic errors.
#         """
#         if self.capacity <= 0:
#             raise ValueError(f"Battery capacity must be positive (got {self.capacity})")
#         soc = float(self._energy) / float(self.capacity)
#         # numerical safety: clip small numerical noise
#         soc = max(0.0, min(soc, 1.0))
#         assert 0.0 <= soc <= 1.0, f"Soc not between 0 and 1 : {soc} ({self._energy}/{self.capacity})"
#         return soc
#
#     @property
#     def energy(self) -> Optional[float]:
#         """
#         Return the current energy stored (kWh). Should always be within [low_capacity, high_capacity].
#         """
#         if self._energy < 0:
#             # This should not happen because we clamp in charge_discharge/reset, but keep a guard.
#             raise ValueError(f"Energy is negative ({self._energy}), there is a problem")
#         return float(self._energy)
#
#     def charge_discharge(self, energy: float) -> float:
#         new_energy = self._energy + energy
#
#         clipped_energy = min(max(new_energy, self.low_capacity), self.high_capacity)
#
#         overflow = abs(new_energy - clipped_energy)
#
#         self._energy = clipped_energy
#
#         return overflow
#
#     def get_overcharge_cost(self, overcharge: float) -> float:
#         """
#         Compute cost associated with the overflow/shortage returned by charge_discharge.
#
#         The function assumes `overcharge` is non-negative (kWh). If negative is passed,
#         we still multiply it (preserving sign), but typical use should pass the returned
#         value from `charge_discharge`.
#         """
#         return float(overcharge) * float(self.overcharge_penalty)


# components/Battery.py
from typing import Optional, Dict, Any
import math

try:
    import pybamm
    _PYBAMM_AVAILABLE = True
except Exception:
    pybamm = None  # type: ignore
    _PYBAMM_AVAILABLE = False

from config.types import BatteryConfig


class Battery:
    """
    Backwards-compatible Battery class with optional PyBaMM integration.

    Public API preserved from your original implementation:
      - __init__(battery_config: BatteryConfig)
      - reset()
      - property config -> BatteryConfig
      - property state_of_charge -> float in [0,1]
      - property energy -> float (kWh)
      - charge_discharge(energy: float) -> float (overflow kWh)
      - get_overcharge_cost(overcharge: float) -> float

    Optional PyBaMM:
      - If battery_config.use_pybamm=True and pybamm is installed, the class
        will build a light SPM model on construction and (optionally) run
        a short solve inside charge_discharge for diagnostic outputs
        (terminal voltage, particle states). The internal energy bookkeeping
        (kWh) remains identical unless you explicitly change use_pybamm_mode.
    """

    def __init__(self, battery_config: BatteryConfig) -> None:
        # basic capacity bounds and power limits (kept as in your original class)
        self.capacity = float(battery_config.capacity)
        self.high_capacity = float(battery_config.high_capacity)
        self.low_capacity = float(battery_config.low_capacity)
        self.max_output = float(battery_config.max_output)
        self.min_output = float(battery_config.min_output)

        # initial energy (kWh), clamp in constructor
        self.initial_energy = float(battery_config.initial_energy)
        self._energy = max(self.low_capacity, min(self.initial_energy, self.high_capacity))

        # cost/penalty for overflow/shortage (kWh * penalty)
        self.overcharge_penalty = float(battery_config.overcharge_penalty)

        # Round-trip charge/discharge efficiency (Eq. 5 of the paper: SOC update uses
        # eta_c on charge and 1/eta_d on discharge). Defaults to 1.0 (lossless) if the
        # config doesn't specify it, preserving old behavior unless set explicitly.
        self.eta_c = float(getattr(battery_config, "eta_c", 1.0))
        self.eta_d = float(getattr(battery_config, "eta_d", 1.0))

        # Optional PyBaMM integration parameters (defaults provided)
        # These fields may not be present in your existing BatteryConfig; if not, defaults are used.
        # If you want PyBaMM usage, set `use_pybamm=True` in the config.
        self.use_pybamm = bool(getattr(battery_config, "use_pybamm", False))
        self.pybamm_mode = str(getattr(battery_config, "pybamm_mode", "diagnostic"))
        # nominal voltage (V) used to convert kWh <-> Ah for pybamm currents
        self.nominal_voltage = float(getattr(battery_config, "nominal_voltage", 3.7))
        # timestep length in hours for converting energy->current
        self.timestep_hours = float(getattr(battery_config, "timestep_hours", 1.0))

        # Build a minimal pybamm model if requested and available
        self._pybamm_model = None
        self._pybamm_param = None
        self._pybamm_sim = None
        if self.use_pybamm and _PYBAMM_AVAILABLE:
            try:
                # simple single-particle model (fast) for diagnostics
                model = pybamm.lithium_ion.SPM()
                param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.C3_0)  # example parameter set
                # If user provided custom parameter set name in config, use it
                param_name = getattr(battery_config, "pybamm_parameter_set", None)
                if param_name is not None:
                    try:
                        param = pybamm.ParameterValues(chemistry=param_name)
                    except Exception:
                        # Fall back silently to default param
                        param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.C3_0)

                # create simulation object (do not auto-run)
                sim = pybamm.Simulation(model, parameter_values=param)

                self._pybamm_model = model
                self._pybamm_param = param
                self._pybamm_sim = sim
            except Exception:
                # If anything fails, disable pybamm for safety
                self.use_pybamm = False
                self._pybamm_model = None
                self._pybamm_param = None
                self._pybamm_sim = None

    def reset(self) -> None:
        """Reset battery energy to the configured initial energy (clamped to bounds)."""
        self._energy = max(self.low_capacity, min(self.initial_energy, self.high_capacity))

        # optional: if pybamm sim exists, reset its internal state by re-instantiating (cheap)
        if self.use_pybamm and _PYBAMM_AVAILABLE and self._pybamm_sim is not None:
            try:
                # re-create sim to clear internal state between episodes
                self._pybamm_sim = pybamm.Simulation(self._pybamm_model, parameter_values=self._pybamm_param)
            except Exception:
                # ignore errors in re-instantiation — core functionality still works
                pass

    @property
    def config(self) -> BatteryConfig:
        """Return a BatteryConfig dataclass-like object (parsed) representing current battery params."""
        # include pybamm flags in the returned config so it can be re-used
        cfg_dict = {
            "capacity": self.capacity,
            "high_capacity": self.high_capacity,
            "low_capacity": self.low_capacity,
            "max_output": self.max_output,
            "min_output": self.min_output,
            "overcharge_penalty": self.overcharge_penalty,
            "initial_energy": self.initial_energy,
            "use_pybamm": self.use_pybamm,
            "pybamm_mode": self.pybamm_mode,
            "nominal_voltage": self.nominal_voltage,
            "timestep_hours": self.timestep_hours,
        }
        return BatteryConfig.parse_obj(cfg_dict)

    @property
    def state_of_charge(self) -> float:
        """Return SOC in [0,1]."""
        if self.capacity <= 0:
            raise ValueError(f"Battery capacity must be positive (got {self.capacity})")
        soc = float(self._energy) / float(self.capacity)
        soc = max(0.0, min(soc, 1.0))
        return soc

    @property
    def energy(self) -> Optional[float]:
        """Return stored energy in kWh (always non-negative)."""
        if self._energy < 0:
            # guard — should not happen because we clamp in charge_discharge
            raise ValueError(f"Energy is negative ({self._energy}), check battery logic")
        return float(self._energy)

    def charge_discharge(self, energy_ac: float) -> float:
        """
        Apply an AC-side (grid/PV-facing) energy request to the battery: positive ->
        charge, negative -> discharge (kWh).

        Charging efficiency eta_c and discharging efficiency eta_d (Eq. 5 of the
        paper) are applied when converting the AC-side request into the internal
        (DC/stored) energy delta: charging stores only eta_c of what's requested,
        while discharging must draw 1/eta_d from storage for every unit that reaches
        the AC side. Both default to 1.0 (lossless) if not set in the config.

        Returns:
            realized_ac (kWh): the AC-side energy actually transferred after
            clipping the internal energy to [low_capacity, high_capacity] -- this is
            what the rest of the simulation (grid power balance) should use, since it
            can be smaller in magnitude than the request both due to efficiency loss
            and due to hitting the battery's capacity bounds.
        Notes:
            - If PyBaMM integration is enabled, we optionally run a short diagnostic solve to produce
              terminal voltage & internal states, but the kWh bookkeeping below remains authoritative.
        """
        energy_ac = float(energy_ac)
        if energy_ac >= 0:
            dc_delta = energy_ac * self.eta_c
        else:
            dc_delta = energy_ac / self.eta_d if self.eta_d > 0 else 0.0

        # compute the new (internal, DC-side) energy and clamp
        new_energy = self._energy + dc_delta
        clipped_energy = min(max(new_energy, self.low_capacity), self.high_capacity)
        realized_dc_delta = clipped_energy - self._energy
        self._energy = clipped_energy

        if energy_ac >= 0:
            realized_ac = realized_dc_delta / self.eta_c if self.eta_c > 0 else 0.0
        else:
            realized_ac = realized_dc_delta * self.eta_d

        # kept for the (unused/disabled) PyBaMM diagnostic block below, which
        # reasons in terms of the internal DC-side energy change
        energy = dc_delta
        overflow = abs(new_energy - clipped_energy)

        # Optional: if PyBaMM enabled and simulation object exists, run a short solve to compute diagnostics.
        # This does *not* alter the energy bookkeeping (we keep simpler kWh update).
        if self.use_pybamm and _PYBAMM_AVAILABLE and self._pybamm_sim is not None:
            try:
                # Convert energy (kWh) applied over the configured timestep into current (A):
                # energy (kWh) -> Wh -> Ah = Wh / Vnom -> current A = Ah / timestep_hours
                # Note: we use the energy that *actually flowed into/out of* the battery this step:
                # applied_energy_kwh = clipped_energy - (previous energy before clamp)
                previous_energy = clipped_energy - (energy - (new_energy - clipped_energy))
                # but simpler and robust is to use the energy argument clipped by overflow:
                applied_energy_kwh = float(energy) - (new_energy - clipped_energy)
                # convert to coulomb-based current via Ah
                applied_ah = (applied_energy_kwh * 1000.0) / max(1e-6, self.nominal_voltage)  # Ah
                current_a = applied_ah / max(1e-9, self.timestep_hours)  # A

                # pybamm expects current convention: negative for discharge in many examples.
                # We'll follow an explicit mapping: treat positive energy -> charging current (positive).
                # Adjust sign if your pybamm model/params assume opposite sign.
                # Build a times vector for the solve: single step from 0 to dt
                t_end = float(self.timestep_hours * 3600.0)  # seconds
                # Many PyBaMM parameter sets use SI units (seconds). Build time vector small.
                # We'll run a short simulation from t=0 to t=t_end (seconds) with a constant current.
                # Use the pybamm simulation object's `.solve` interface.
                # Provide the "Current function [A]" as input via parameter values if supported.
                # Simpler approach: set inputs via the "Current function [A]" variable if it exists.
                # Note: exact variable names vary by model; we try common ones and otherwise skip.
                sim = self._pybamm_sim

                # Prepare times for pybamm (seconds)
                times = [0.0, t_end]

                # try to set up input current via simulation inputs
                # Many models have a parameter "Current function [A]" or similar; we attempt common keys.
                # Fallback: run sim.solve with an applied current via inputs dict where possible.
                # We'll build an inputs dict: variable name -> value (scalar or function).
                inputs: Dict[str, Any] = {}
                # Common pybamm variable keys to try:
                candidate_current_names = [
                    "Current function [A]",
                    "Current [A]",
                    "Current input [A]",
                    "Current",  # last-ditch
                ]
                for name in candidate_current_names:
                    try:
                        # test if the variable exists in the model
                        if name in sim.model.variables:
                            inputs[name] = current_a
                            break
                    except Exception:
                        # ignore and continue
                        pass

                # Run a short simulate/solve if we have candidate input; otherwise do a short solve without inputs.
                # This is done for diagnostics only; failures must not break main app.
                if inputs:
                    try:
                        sim.solve(t_eval=times, inputs=inputs)
                    except Exception:
                        # fallback: try solve without inputs (may still produce states)
                        try:
                            sim.solve(t_eval=times)
                        except Exception:
                            pass
                else:
                    try:
                        sim.solve(t_eval=times)
                    except Exception:
                        pass

                # after solve, store last terminal voltage if available (for debugging)
                try:
                    y = sim.solution
                    # terminal voltage variable names vary; try frequent names
                    for varname in ["Terminal voltage [V]", "Electrolyte potential [V]", "Voltage [V]"]:
                        if varname in y.keys():
                            self._last_terminal_voltage = float(y[varname].data[-1])
                            break
                    else:
                        self._last_terminal_voltage = None
                except Exception:
                    self._last_terminal_voltage = None

            except Exception:
                # swallow any pybamm-related errors to keep behavior robust
                pass

        return realized_ac

    def get_overcharge_cost(self, overcharge: float) -> float:
        """Compute cost associated with overflow/shortage (kWh * penalty)."""
        return float(overcharge) * float(self.overcharge_penalty)

    # Optional diagnostics accessor for advanced usage:
    def last_terminal_voltage(self) -> Optional[float]:
        """Return last terminal voltage from the optional PyBaMM solve (or None)."""
        return getattr(self, "_last_terminal_voltage", None)