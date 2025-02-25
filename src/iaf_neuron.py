from typing import Dict, Any, List
from itertools import repeat
import numpy as np
from numba import njit
from .synapse_group import SynapseGroup


@njit
def update_membrane_voltage(vm, rest, dt, tau, exc_conductance, inh_conductance, ex_rev, resistance):
    """Update membrane voltage using pre-computed terms."""
    leak_term = (rest - vm) * dt / tau
    exc_df = ex_rev - vm
    inh_df = rest - vm
    exc_dv = exc_conductance * exc_df * resistance * dt / tau
    inh_dv = inh_conductance * inh_df * resistance * dt / tau
    return vm + leak_term + exc_dv + inh_dv


class IaF:
    def __init__(
        self,
        time_constant: float = 20e-3,
        resistance: float = 100e6,
        reset_voltage: float = -70e-3,
        spike_threshold: float = -50e-3,
        dt: float = 0.001,
        use_homeostasis: bool = True,
        homeostasis_tau: float = 20,
        homeostasis_set_point: float | None = None,
    ):
        # Set membrane properties
        self.time_constant = time_constant
        self.resistance = resistance
        self.reset_voltage = reset_voltage
        self.spike_threshold = spike_threshold
        self.dt = dt
        self.vm = reset_voltage  # initial membrane voltage

        # Set homeostasis properties
        self.use_homeostasis = use_homeostasis
        if self.use_homeostasis:
            self.homeostasis_tau = homeostasis_tau
            if homeostasis_set_point is None:
                raise ValueError("homeostasis_set_point must be provided if use_homeostasis is True")
            self.homeostasis_set_point = homeostasis_set_point

        self.synapse_groups: list[SynapseGroup] = []

    def add_synapse_group(
        self,
        num_synapses: int,
        max_weight: float,
        reversal: float,
        tau: float,
        **plasticity_params: Dict[str, Any],
    ):
        synapse_group = SynapseGroup(num_synapses, max_weight, reversal, tau, **plasticity_params)
        self.synapse_groups.append(synapse_group)

    def step(
        self,
        input_rates: np.ndarray | List[np.ndarray],
        same_input_rates: bool = True,
    ):
        """Implement a step of the IaF neuron.

        Args:
            input_rates: The input rates to the neuron.
            same_input_rates: Whether the input rates are the same for all synapse groups.
                If True, then input_rates should be a single numpy array.
                If False, then input_rates should be a list of numpy arrays, one for each synapse group.
        """
        if same_input_rates:
            if not isinstance(input_rates, np.ndarray):
                raise ValueError("input_rates must be a numpy array if same_input_rates is True")
            input_rates = repeat(input_rates)
        else:
            if not isinstance(input_rates, list) and len(input_rates) != len(self.synapse_groups):
                raise ValueError(
                    "input_rates must be a list of numpy arrays (one for each synapse group) if same_input_rates is False"
                )

        for synapse_group, input_rate in zip(self.synapse_groups, input_rates):
            synapse_group.step(input_rate)
