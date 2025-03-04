from typing import Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from numba import njit
from ..utils import create_rng, resolve_dataclass


@njit
def generate_spikes_numba(random_values, rate, dt):
    """JIT-compiled function to generate spikes."""
    return random_values < (rate * dt)


@njit
def compute_conductance(spikes, weights, threshold):
    """Compute conductance from spikes and weights with threshold."""
    return np.sum(spikes * weights * (weights > threshold))


@njit
def clip_weights(weights, min_val, max_val):
    """Clip weights to range [min_val, max_val] in-place."""
    np.clip(weights, min_val, max_val, out=weights)
    return weights


@njit
def stdp_step(
    weights,
    spikes,
    depression_eligibility,
    potentiation_eligibility,
    potentiation_increment,
    potentiation_decay_factor,
):
    weights += spikes * depression_eligibility
    potentiation_eligibility += spikes * potentiation_increment - potentiation_eligibility * potentiation_decay_factor
    return weights, potentiation_eligibility


@dataclass
class SpikeGenerator:
    num_neurons: int
    dt: float
    max_batch: int = 10

    def __post_init__(self):
        self.rng = create_rng()
        self.batch_idx = 0
        self.current_batch = None
        self.batch_size = 0

        # Warmup the numba function
        generate_spikes_numba(self.rng.random(self.num_neurons), 1.0, self.dt)

    def initialize_batch(self, steps_remaining=None):
        """Create a batch of random numbers for reuse in poisson spike generation."""
        batch_size = min(self.max_batch, steps_remaining or self.max_batch)
        self.current_batch = self.rng.random((batch_size, self.num_neurons))
        self.batch_idx = 0
        self.batch_size = batch_size

    def get_spikes(self, rate, steps_remaining=None):
        """Generate spikes for the current timestep."""
        if self.batch_idx >= self.batch_size:
            self.initialize_batch(steps_remaining)

        spikes = generate_spikes_numba(self.current_batch[self.batch_idx], rate, self.dt)
        self.batch_idx += 1
        return spikes


@dataclass
class PlasticityParams:
    use_stdp: bool = True  # Whether to use STDP
    stdp_rate: float = 0.01  # The rate of potentiation/depression
    depression_potentiation_ratio: float = 1.1  # The ratio of depression to potentiation
    potentiation_tau: float = 0.02  # The time constant of potentiation
    depression_tau: float = 0.02  # The time constant of depression
    use_homeostasis: bool = True  # Whether to use homeostasis
    homeostasis_tau: float = 20  # The time constant of homeostasis
    homeostasis_scale: float = 1.0  # The scale of homeostasis


@dataclass
class ReplacementParams:
    use_replacement: bool = True  # Whether to use replacement
    lose_synapse_ratio: float = 0.01  # The ratio of maximum weight that causes a synapse to be lost
    new_synapse_ratio: float = 0.01  # The ratio of maximum weight that new synapses are initialized to


class SynapseGroup(ABC):
    """
    Abstract base class for a group of synapses with shared properties.

    This class defines the interface and common properties for different types
    of synapse groups, whether they receive input from a source population or
    direct inputs to each synapse.
    """

    # Common properties for all synapse groups
    num_synapses: int
    max_weight: float  # in units of conductance
    reversal: float  # in V
    tau: float  # in seconds
    dt: float  # in seconds
    conductance_threshold: float  # in relative units of max_weight
    min_conductance: float  # in units of conductance

    # Plasticity related properties
    plasticity_params: PlasticityParams

    # State variables
    weights: np.ndarray  # Weight of each synapse
    conductance: float  # Current conductance

    # RNG for random processes
    rng: np.random.Generator

    def _init_base_params(
        self,
        num_synapses: int,
        max_weight: float,
        reversal: float,
        tau: float,
        dt: float,
        conductance_threshold: float | None = None,
        plasticity_params: PlasticityParams | Dict[str, Any] | None = None,
    ):
        """
        Initialize the synapse group with common parameters.

        Parameters
        ----------
        num_synapses : int
            Number of synapses in the group
        max_weight : float
            Maximum weight for each synapse (in units of conductance)
        reversal : float
            Reversal potential (in V)
        tau : float
            Time constant (in seconds)
        dt : float
            Time step (in seconds)
        conductance_threshold : float, optional
            Threshold for conductance activation (in relative units of max_weight)
        plasticity_params : PlasticityParams or dict, optional
            Parameters for plasticity mechanisms
        """
        self.num_synapses = num_synapses
        self.max_weight = max_weight
        self.reversal = reversal
        self.tau = tau
        self.dt = dt
        self.conductance_threshold: float = conductance_threshold or 0.0
        self.min_conductance = self.conductance_threshold * self.max_weight

        # Initialize state variables
        self.weights: np.ndarray = self.rng.random(num_synapses) * self.max_weight
        self.conductance = 0.0

        # Create a spike generator for the group
        self._spike_generator = SpikeGenerator(num_synapses, dt)

        # Set up plasticity parameters
        self.plasticity_params = resolve_dataclass(plasticity_params, PlasticityParams)

        # Set up STDP parameters
        if self.plasticity_params.use_stdp:
            # This determines how much the weight can change for a single pre/post pairing
            self.potentiation_increment = self.plasticity_params.stdp_rate
            self.depression_increment = (
                self.plasticity_params.stdp_rate * self.plasticity_params.depression_potentiation_ratio
            )

            # Buffers to store the potentiation/depression for the current timestep
            self.potentiation_eligibility: np.ndarray = np.zeros(self.num_synapses)
            self.depression_eligibility: float = 0.0

        # Precompute dt / tau divisions for optimization
        self._dt_tau = self.dt / self.tau
        if self.plasticity_params.use_stdp:
            self._dt_potentiation_tau = self.dt / self.plasticity_params.potentiation_tau
            self._dt_depression_tau = self.dt / self.plasticity_params.depression_tau
        if self.plasticity_params.use_homeostasis:
            self._dt_homeostasis_tau = self.dt / self.plasticity_params.homeostasis_tau

        # Create RNG
        self.rng = create_rng()

    def initialize(self, reset_weights: bool = False, reset_sources: bool = False):
        """Initialize the synapse group."""
        self.conductance = 0.0
        if self.plasticity_params.use_stdp:
            self.potentiation_eligibility = np.zeros(self.num_synapses)
            self.depression_eligibility = 0.0
        if reset_weights:
            self.weights = self.rng.random(self.num_synapses) * self.max_weight
        if reset_sources and isinstance(self, SourcedSynapseGroup):
            self.presynaptic_source = self.rng.integers(0, self.num_presynaptic_neurons, self.num_synapses)

    def postsynaptic_spike(self):
        """Implement STDP updates in the case of a postsynaptic spike."""
        if self.plasticity_params.use_stdp:
            # Whenever a postsynaptic spike occurs, the depression eligibility trace is incremented
            # We make it negative because we want to depress the synapse (and it's a linear STDP model so it's additive)
            self.depression_eligibility -= self.depression_increment

            # Postsynaptic spikes cause potentiation in proportion to the potentiation eligibility trace
            self.weights += self.potentiation_eligibility

            # Weights are clipped to the range [0, self.max_weight]
            clip_weights(self.weights, 0, self.max_weight)

    def step(self, input_rates: np.ndarray, homeostasis: float | None = None):
        """Step the synapse group based on the presynaptic input rates."""
        input_rates = self.transform_input_rates(input_rates)

        # Get the spikes for each synapse
        spikes = self._spike_generator.get_spikes(input_rates)

        # Compute the conductance added from the synapses with spikes
        new_conductance = compute_conductance(spikes, self.weights, self.min_conductance)

        # Update the total conductance from this synapse group
        self.conductance += new_conductance - self.conductance * self._dt_tau

        # Implement STDP
        if self.plasticity_params.use_stdp:
            self.weights, self.potentiation_eligibility = stdp_step(
                self.weights,
                spikes,
                self.depression_eligibility,
                self.potentiation_eligibility,
                self.potentiation_increment,
                self._dt_potentiation_tau,
            )

            # Presynaptic spikes evoke a change in potentiation eligibility traces (it also decays over time)
            self.potentiation_eligibility += (
                spikes * self.potentiation_increment - self.potentiation_eligibility * self._dt_potentiation_tau
            )

            # The depression eligibility trace decays over time
            self.depression_eligibility -= self.depression_eligibility * self._dt_depression_tau

        # Implement homeostasis
        if self.plasticity_params.use_homeostasis and homeostasis is not None:
            # <homeostasis> is the postsynaptic neuron's estimate of the ratio between the
            # current firing rate and the set point firing rate.
            # Neurons scale their synaptic weight by this factor at a given time constant,
            # and each synapse group can vary the amount of homeostasis with self.homeostasis_scale
            homeostasis_factor = homeostasis * self.plasticity_params.homeostasis_scale * self._dt_homeostasis_tau
            self.weights += homeostasis_factor * self.weights

        # Implement replacement
        self.handle_replacement()

        # Clip the weights to the range [0, self.max_weight]
        clip_weights(self.weights, 0, self.max_weight)

    def get_current(self, vm: float):
        """
        Calculate current based on conductance and membrane potential.

        Parameters
        ----------
        vm : float
            Membrane potential

        Returns
        -------
        float
            Current produced by the synapse group
        """
        return self.conductance * (self.reversal - vm)

    def handle_replacement(self):
        """Called in step by default but only used for some subclasses which should overwrite this."""
        pass

    @abstractmethod
    def transform_input_rates(self, input_rates: np.ndarray):
        """
        Transform the input rates to the correct format for this synapse group.

        Parameters
        ----------
        input_rates : np.ndarray
            Input rates to validate

        Returns
        -------
        np.ndarray
            Transformed input rates

        Raises
        ------
        ValueError
            If the input rates are incompatible
        """
        pass


class SourcedSynapseGroup(SynapseGroup):
    """
    A synapse group that receives inputs from a source population.

    This group internally manages which synapse is connected to which source.
    """

    num_presynaptic_neurons: int
    presynaptic_source: np.ndarray
    replacement_params: ReplacementParams

    def __init__(
        self,
        num_synapses: int,
        max_weight: float,
        reversal: float,
        tau: float,
        dt: float,
        conductance_threshold: float | None = None,
        num_presynaptic_neurons: int | None = None,
        presynaptic_source: np.ndarray | None = None,
        plasticity_params: PlasticityParams | Dict[str, Any] | None = None,
        replacement_params: ReplacementParams | Dict[str, Any] | None = None,
    ):
        """
        Initialize the synapse group with common parameters.

        Parameters
        ----------
        num_synapses : int
            Number of synapses in the group
        max_weight : float
            Maximum weight for each synapse (in units of conductance)
        reversal : float
            Reversal potential (in V)
        tau : float
            Time constant (in seconds)
        dt : float
            Time step (in seconds)
        conductance_threshold : float, optional
            Threshold for conductance activation (in relative units of max_weight)
        num_presynaptic_neurons : int, optional
            Number of presynaptic neurons
        presynaptic_source : np.ndarray, optional
            Array of presynaptic neuron indices for each synapse
        plasticity_params : PlasticityParams or dict, optional
            Parameters for plasticity mechanisms
        replacement_params : ReplacementParams or dict, optional
            Parameters for synapse replacement
        """
        self._init_base_params(num_synapses, max_weight, reversal, tau, dt, conductance_threshold, plasticity_params)

        # Set up presynaptic source parameters
        if num_presynaptic_neurons is None:
            raise ValueError("num_presynaptic_neurons must be provided")
        self.num_presynaptic_neurons = num_presynaptic_neurons
        if presynaptic_source is not None:
            if np.any(presynaptic_source < 0) or np.any(presynaptic_source >= num_presynaptic_neurons):
                raise ValueError("initial_source must be an array of integers between 0 and num_presynaptic_neurons")
            self.presynaptic_source: np.ndarray = presynaptic_source
        else:
            self.presynaptic_source: np.ndarray = self.rng.integers(0, num_presynaptic_neurons, num_synapses)

        # Set up replacement parameters
        self.replacement_params = resolve_dataclass(replacement_params, ReplacementParams)

        if self.replacement_params.use_replacement:
            self.min_weight = self.replacement_params.lose_synapse_ratio * self.max_weight
            self.new_weight = self.replacement_params.new_synapse_ratio * self.max_weight

    def handle_replacement(self):
        """
        Implement synapse replacement: remove weak synapses and create new ones.
        """
        if self.replacement_params.use_replacement:
            synapses_to_replace = self.weights < self.min_weight
            n_synapses_to_replace = np.sum(synapses_to_replace)
            if n_synapses_to_replace > 0:
                self.weights[synapses_to_replace] = self.new_weight
                self.presynaptic_source[synapses_to_replace] = self.rng.integers(
                    0, self.num_presynaptic_neurons, n_synapses_to_replace
                )

    def validate_input_rates(self, input_rates: np.ndarray):
        """
        Validate that the input rates match the number of presynaptic neurons.

        Parameters
        ----------
        input_rates : np.ndarray
            Input rates to validate

        Raises
        ------
        ValueError
            If the input rates length doesn't match the number of presynaptic neurons
        """
        if len(input_rates) != self.num_presynaptic_neurons:
            raise ValueError(
                f"Number of input rates provided ({len(input_rates)}) does not match "
                f"number of presynaptic neurons ({self.num_presynaptic_neurons})."
            )
        return input_rates[self.presynaptic_source]


class DirectSynapseGroup(SynapseGroup):
    """
    A synapse group that receives inputs directly from a matched set of input rates.
    """

    def __init__(
        self,
        num_synapses: int,
        max_weight: float,
        reversal: float,
        tau: float,
        dt: float,
        conductance_threshold: float | None = None,
        plasticity_params: PlasticityParams | Dict[str, Any] | None = None,
    ):
        """
        Initialize the synapse group with common parameters.

        Parameters
        ----------
        num_synapses : int
            Number of synapses in the group
        max_weight : float
            Maximum weight for each synapse (in units of conductance)
        reversal : float
            Reversal potential (in V)
        tau : float
            Time constant (in seconds)
        dt : float
            Time step (in seconds)
        conductance_threshold : float, optional
            Threshold for conductance activation (in relative units of max_weight)
        plasticity_params : PlasticityParams or dict, optional
            Parameters for plasticity mechanisms
        """
        self._init_base_params(num_synapses, max_weight, reversal, tau, dt, conductance_threshold, plasticity_params)

    def validate_input_rates(self, input_rates: np.ndarray):
        """
        Validate that the input rates match the number of synapses.

        Parameters
        ----------
        input_rates : np.ndarray
            Input rates to validate

        Raises
        ------
        ValueError
            If the input rates length doesn't match the number of synapses
        """
        if len(input_rates) != self.num_synapses:
            raise ValueError(
                f"Number of input rates provided ({len(input_rates)}) does not match "
                f"number of synapses ({self.num_synapses})."
            )
        return input_rates
