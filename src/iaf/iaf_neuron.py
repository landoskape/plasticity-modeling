from __future__ import annotations
from pathlib import Path
from typing import List, Union, Optional, Tuple
import yaml
import numpy as np
from numba import njit
from .synapse_group import SynapseGroup, SourcedSynapseGroup, DirectSynapseGroup
from .config import NeuronConfig

SynapseGroupTypes = Union[SourcedSynapseGroup, DirectSynapseGroup]


class IaF:
    """Integrate-and-Fire neuron model.

    This class implements a simple leaky integrate-and-fire neuron model with
    synaptic inputs and homeostatic regulation. The neuron integrates synaptic
    currents, fires when its membrane potential exceeds a threshold, and can
    adjust its firing rate through homeostasis.

    Attributes
    ----------
    time_constant : float
        The membrane time constant in seconds.
    resistance : float
        The membrane resistance in Ohms.
    reset_voltage : float
        The voltage to reset to after a spike, in Volts.
    spike_threshold : float
        The threshold voltage for spike generation, in Volts.
    dt : float
        The time step of the simulation in seconds.
    vm : float
        The current membrane voltage, in Volts.
    use_homeostasis : bool
        Whether to use homeostatic mechanisms.
    homeostasis_tau : float
        The time constant for homeostasis in seconds.
    homeostasis_set_point : float
        The target firing rate in Hz.
    synapse_groups : dict[str, SynapseGroupTypes]
        Dictionary mapping names to synapse groups connected to this neuron.
    spike : bool
        Whether the neuron spiked in the most recent time step.
    homeostasis_rate_estimate : float
        Estimate of the current firing rate (used for homeostasis).
    """

    def __init__(
        self,
        time_constant: float = 20e-3,
        resistance: float = 100e6,
        reset_voltage: float = -70e-3,
        spike_threshold: float = -50e-3,
        dt: float = 0.001,
        use_homeostasis: bool = True,
        homeostasis_tau: float = 20,
        homeostasis_set_point: Optional[float] = None,
    ):
        """Initialize an integrate-and-fire neuron.

        Parameters
        ----------
        time_constant : float, optional
            The membrane time constant in seconds, default is 20e-3.
        resistance : float, optional
            The membrane resistance in Ohms, default is 100e6.
        reset_voltage : float, optional
            The voltage to reset to after a spike, in Volts, default is -70e-3.
        spike_threshold : float, optional
            The threshold voltage for spike generation, in Volts, default is -50e-3.
        dt : float, optional
            The time step of the simulation in seconds, default is 0.001.
        use_homeostasis : bool, optional
            Whether to use homeostatic mechanisms, default is True.
        homeostasis_tau : float, optional
            The time constant for homeostasis in seconds, default is 20.
        homeostasis_set_point : float, optional
            The target firing rate in Hz, required if use_homeostasis is True.

        Raises
        ------
        ValueError
            If use_homeostasis is True but homeostasis_set_point is not provided.
        """
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

        self.synapse_groups: dict[str, SynapseGroupTypes] = {}

    def __repr__(self) -> str:
        attrs_to_show = [
            "time_constant",
            "resistance",
            "reset_voltage",
            "spike_threshold",
            "dt",
            "use_homeostasis",
            "homeostasis_tau",
            "homeostasis_set_point",
            "synapse_groups",
        ]
        attrs = "\n    ".join([f"{attr}={getattr(self, attr)}" for attr in attrs_to_show])
        return f"IaF(\n    {attrs}\n)"

    @classmethod
    def from_yaml(cls, fpath: Path) -> "IaF":
        """Create an IaF neuron from a YAML configuration file.

        Parameters
        ----------
        fpath : Path
            The path to the YAML configuration file.

        Returns
        -------
        IaF
            A new IaF neuron instance.
        """
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(NeuronConfig.model_validate(config))

    @classmethod
    def from_config(cls, config: NeuronConfig) -> "IaF":
        """Create an IaF neuron from a configuration object.

        Parameters
        ----------
        config : NeuronConfig
            The configuration for the neuron.

        Returns
        -------
        IaF
            A new IaF neuron instance.
        """
        return cls(
            time_constant=config.time_constant,
            resistance=config.resistance,
            reset_voltage=config.reset_voltage,
            spike_threshold=config.spike_threshold,
            dt=config.dt,
            use_homeostasis=config.use_homeostasis,
            homeostasis_tau=config.homeostasis_tau,
            homeostasis_set_point=config.homeostasis_set_point,
        )

    def initialize(self, include_synapses: bool = True, reset_weights: bool = True) -> None:
        """Initialize the neuron and optionally its synapses to their resting state.

        This method resets the membrane voltage to the reset voltage and optionally
        initializes the synapse groups. It also resets the homeostasis rate estimate
        if homeostasis is enabled.

        Parameters
        ----------
        include_synapses : bool, optional
            Whether to initialize the synapse groups as well, default is True.
        reset_weights : bool, optional
            Whether to reset the synaptic weights to their initial values,
            default is True. Only relevant if include_synapses is True.
        """
        self.vm = self.reset_voltage
        if self.use_homeostasis:
            self.homeostasis_rate_estimate = self.homeostasis_set_point
        if include_synapses:
            for synapse_group in self.synapse_groups.values():
                synapse_group.initialize(reset_weights=reset_weights)

    def add_synapse_group(self, synapse_group: SynapseGroup, name: Optional[str] = None) -> None:
        """Add a synapse group to the neuron.

        Parameters
        ----------
        synapse_group : SynapseGroup
            The synapse group to add.
        name : str, optional
            The name of the synapse group. If None, a name will be generated
            based on the existing number of synapse groups.
        """
        if name is None:
            name = f"synapse_group_{len(self.synapse_groups)}"
        self.synapse_groups[name] = synapse_group

    def step(self, input_rates: List[np.ndarray]) -> Tuple[float, bool]:
        """Simulate one time step of the neuron dynamics.

        This method updates the membrane voltage based on synaptic inputs,
        determines if the neuron spikes, and processes homeostasis.

        The step proceeds as follows:
        1. Check if the neuron is above threshold to determine if it spikes
        2. If spiking, reset the membrane voltage and notify synapse groups
        3. If not spiking, update the membrane voltage based on inputs
        4. Update the homeostasis rate estimate if enabled
        5. Step each synapse group with the corresponding input rates

        Parameters
        ----------
        input_rates : List[np.ndarray]
            A list of input rates arrays, one for each synapse group.
            Each array should have one rate value per input to the corresponding
            synapse group.

        Raises
        ------
        ValueError
            If the input_rates list does not match the number of synapse groups.
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

        return self.vm, self.spike


@njit
def update_membrane_voltage(vm: float, vrest: float, tau: float, resistance: float, current: float, dt: float) -> float:
    """Update membrane voltage using the leaky integrate-and-fire equation.

    This function implements the discrete-time update equation for the
    leaky integrate-and-fire neuron model:

    V(t+dt) = V(t) + dt/tau * (Vrest - V(t) + R*I(t))

    Parameters
    ----------
    vm : float
        The current membrane voltage in Volts.
    vrest : float
        The resting membrane voltage in Volts.
    tau : float
        The membrane time constant in seconds.
    resistance : float
        The membrane resistance in Ohms.
    current : float
        The total input current in Amperes (not including leak).
    dt : float
        The time step in seconds.

    Returns
    -------
    float
        The updated membrane voltage in Volts.
    """
    leak_term = (vrest - vm) * dt / tau
    synapse_term = current * resistance * dt / tau
    return vm + leak_term + synapse_term
