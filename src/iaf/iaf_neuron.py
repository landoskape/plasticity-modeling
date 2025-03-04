from typing import Dict, Any, List
from itertools import repeat
import numpy as np
from numba import njit
from .synapse_group import SynapseGroup


@njit
def update_membrane_voltage(vm, vrest, tau, resistance, current, dt):
    """Update membrane voltage.

    Parameters
    ----------
    vm: float
        The membrane voltage.
    vrest: float
        The resting membrane voltage.
    tau: float
        The time constant of the membrane.
    resistance: float
        The resistance of the membrane.
    current: float
        Total membrane current not from the leak current (e.g. synaptic input)
    dt: float
        The time step.
    """
    leak_term = (vrest - vm) * dt / tau
    synapse_term = current * resistance * dt / tau
    return vm + leak_term + synapse_term


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
            self._min_allowed_rate_estimate = self.homeostasis_set_point / 100
            self._dt_homeostasis_tau = self.dt / self.homeostasis_tau

        self.synapse_groups: dict[str, SynapseGroup] = {}

    def initialize(self, include_synapses: bool = True, reset_weights: bool = False):
        """Method for returning the neuron and synapses to resting state."""
        self.vm = self.reset_voltage
        if self.use_homeostasis:
            self.homeostasis_rate_estimate = self.homeostasis_set_point
        if include_synapses:
            for synapse_group in self.synapse_groups.values():
                synapse_group.initialize(reset_weights=reset_weights)

    def add_synapse_group(self, synapse_group: SynapseGroup, name: str | None = None):
        """Add a synapse group to the neuron.

        Args:
            synapse_group: The synapse group to add.
            name: The name of the synapse group.
        """
        if name is None:
            name = f"synapse_group_{len(self.synapse_groups)}"
        self.synapse_groups[name] = synapse_group

    def step(self, input_rates: List[np.ndarray]):
        """Implement a step of the IaF neuron.

        Args:
            input_rates: The input rates to each synapse group.
        """
        if self.vm > self.spike_threshold:
            self.spike = True
            self.vm = self.reset_voltage
            for synapse_group in self.synapse_groups.values():
                synapse_group.postsynaptic_spike()

        else:
            self.spike = False
            input_current = [synapse_group.get_current(self.vm) for synapse_group in self.synapse_groups.values()]
            input_current = np.sum(input_current)
            self.vm = update_membrane_voltage(
                self.vm,
                self.reset_voltage,
                self.time_constant,
                self.resistance,
                input_current,
                self.dt,
            )

        # Check that the input rates are a list of numpy arrays
        if not isinstance(input_rates, list) and len(input_rates) != len(self.synapse_groups):
            raise ValueError(
                "input_rates must be a list of numpy arrays (one for each synapse group) if same_input_rates is False"
            )

        if self.use_homeostasis:
            # We acquire an estimate of the homeostatic firing rate by the difference method
            # Instantaneous firing rate in this timestep
            spike_term = 1 * self.spike / self.dt
            # Update the estimate with decay term and instantaneous rate
            self.homeostasis_rate_estimate += self._dt_homeostasis_tau * (spike_term - self.homeostasis_rate_estimate)
            # Then compute the drive as the log ratio between the set point and the estimate
            constrained_rate_estimate = max(self._min_allowed_rate_estimate, self.homeostasis_rate_estimate)
            homeostatic_drive = np.log(self.homeostasis_set_point / constrained_rate_estimate)
        else:
            homeostatic_drive = None

        for synapse_group, input_rate in zip(self.synapse_groups.values(), input_rates):
            synapse_group.step(input_rate, homeostasis=homeostatic_drive)
