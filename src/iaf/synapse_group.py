import numpy as np
from numba import njit
from dataclasses import dataclass
from ..utils import create_rng


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
    depression_decay_factor,
):
    weights += spikes * depression_eligibility
    potentiation_eligibility += spikes * potentiation_increment - potentiation_eligibility * potentiation_decay_factor
    depression_eligibility -= depression_eligibility * depression_decay_factor
    return weights, potentiation_eligibility, depression_eligibility


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


class SynapseGroup:
    """
    A group of synapses with shared properties.

    Each synapse group has shared properties, including the maximum weight, replacement
    properties (lose/new synapse ratio), and conductance threshold. Each synapse group
    also has shared plasticity properties, including potentiation/depression rates,
    potentiation/depression time constants, and homeostasis rate.

    Attributes
    ----------
    name: str
        The name of the synapse group.
    conductance: float
        The current conductance of the synapse group. All synapses in the group contribute to
        the conductance equally, and it is funneled into a single value to improve performance.
    max_weight: float
        The maximum weight of the synapse group.
    num_synapses: int
        The number of synapses in the group.
    conductance_threshold: float | None
        The conductance threshold for the synapse group - the synapses will only contribute to
        current if the conductance is above this threshold.
    """

    def __init__(
        self,
        name: str,
        num_synapses: int,
        max_weight: float,  # in units of conductance
        reversal: float,  # in V
        tau: float,  # in seconds
        dt: float,  # in seconds
        conductance_threshold: float | None = None,  # in relative units of max_weight
        # The following parameters are only used if use_replacement is True
        use_replacement: bool = True,
        num_presynaptic_neurons: int | None = None,  # The number of presynaptic neurons
        lose_synapse_ratio: float = 0.01,  # in relative units of max_weight
        new_synapse_ratio: float = 0.01,  # in relative units of max_weight
        # The following parameters are only used if use_stdp is True
        use_stdp: bool = True,
        stdp_rate: float = 0.01,  # in relative units of max_weight
        depression_potentiation_ratio: float = 1.1,  # in relative units of max_weight
        potentiation_tau: float = 0.02,  # in seconds
        depression_tau: float = 0.02,  # in seconds
        # The following parameters are only used if use_homeostasis is True
        use_homeostasis: bool = True,
        homeostasis_tau: float = 20,  # in seconds
        homeostasis_scale: float = 1.0,  # in relative units
    ):
        # Basic biophysical properties
        self.name: str = name
        self.conductance: float = 0.0
        self.max_weight: float = max_weight
        self.num_synapses: int = num_synapses
        self.reversal: float = reversal
        self.tau: float = tau
        self.dt: float = dt
        self.conductance_threshold: float | None = conductance_threshold
        if conductance_threshold is not None:
            self.min_conductance = self.max_weight * conductance_threshold
        else:
            self.min_conductance = 0.0
        self.rng = create_rng()  # biophysically plausible random number generator

        # Properties for each synapse
        self.weights: np.ndarray = self.rng.random(num_synapses) * self.max_weight

        # Create a spike generator for the group
        self._spike_generator = SpikeGenerator(num_synapses, dt)

        # Synapse replacement parameters
        self.use_replacement: bool = use_replacement
        if use_replacement:
            if num_presynaptic_neurons is None:
                raise ValueError("num_presynaptic_neurons must be specified if use_replacement is True")
            self.num_presynaptic_neurons: int = num_presynaptic_neurons
            self.presynaptic_source: np.ndarray = self.rng.integers(0, num_presynaptic_neurons, num_synapses)
            self.lose_synapse_ratio: float = lose_synapse_ratio
            self.new_synapse_ratio: float = new_synapse_ratio
            self.min_weight: float = self.max_weight * lose_synapse_ratio
            self.new_weight: float = self.max_weight * new_synapse_ratio

        # STDP parameters
        self.use_stdp: bool = use_stdp
        if use_stdp:
            # STDP Time Constants
            self.potentiation_tau: float = potentiation_tau
            self.depression_tau: float = depression_tau

            # Set rate of potentiation & depression
            self.stdp_rate: float = stdp_rate
            self.depression_potentiation_ratio: float = depression_potentiation_ratio

            # This determines how much the weight can change for a single pre/post pairing
            self.potentiation_increment: float = stdp_rate * max_weight
            self.depression_increment: float = stdp_rate * max_weight * depression_potentiation_ratio

            # Buffers to store the potentiation/depression for the current timestep
            self.potentiation_eligibility: np.ndarray = np.zeros(self.num_synapses)
            self.depression_eligibility: float = 0.0

        # Homeostasis parameters
        self.use_homeostasis: bool = use_homeostasis
        if use_homeostasis:
            self.homeostasis_tau: float = homeostasis_tau
            self.homeostasis_scale: float = homeostasis_scale

        # Precompute time constants
        self._dt_tau = self.dt / self.tau
        if self.use_stdp:
            self._dt_potentiation_tau = self.dt / self.potentiation_tau
            self._dt_depression_tau = self.dt / self.depression_tau
        if self.use_homeostasis:
            self._dt_homeostasis_tau = self.dt / self.homeostasis_tau

    def initialize(self, reset_weights: bool = False):
        """Initialize the synapse group."""
        self.conductance = 0.0
        if self.use_stdp:
            self.potentiation_eligibility = np.zeros(self.num_synapses)
            self.depression_eligibility = 0.0
        if reset_weights:
            self.weights = self.rng.random(self.num_synapses) * self.max_weight

    def postsynaptic_spike(self):
        """Implement STDP updates in the case of a postsynaptic spike."""
        if self.use_stdp:
            # Whenever a postsynaptic spike occurs, the depression eligibility trace is incremented
            # We make it negative because we want to depress the synapse (and it's a linear STDP model so it's additive)
            self.depression_eligibility -= self.depression_increment

            # Postsynaptic spikes cause potentiation in proportion to the potentiation eligibility trace
            self.weights += self.potentiation_eligibility

            # Weights are clipped to the range [0, self.max_weight]
            clip_weights(self.weights, 0, self.max_weight)

    def step(self, input_rates: np.ndarray, homeostasis: float | None = None):
        """Step the synapse group based on the presynaptic input rates.

        TODO: Consider adding dt to these input arguments instead of having it as an attribute.
        """
        if self.use_replacement:
            if len(input_rates) != self.num_presynaptic_neurons:
                raise ValueError("input_rates must be the same length as num_presynaptic_neurons")
            # Get the specific input rate to each synapse in this group
            input_rates = input_rates[self.presynaptic_source]

        else:
            if len(input_rates) != self.num_synapses:
                raise ValueError("input_rates must be the same length as num_synapses")

        # Get the spikes for each synapse
        spikes = self._spike_generator.get_spikes(input_rates)
        print(np.mean(spikes))

        # Compute the conductance added from the synapses with spikes
        new_conductance = compute_conductance(spikes, self.weights, self.min_conductance)

        # Compute the decay of the conductance that lingers from the previous spikes
        conductance_decay = self.conductance * self._dt_tau

        # Update the total conductance from this synapse group
        self.conductance += new_conductance - conductance_decay

        # Implement STDP
        if self.use_stdp:
            self.weights, self.potentiation_eligibility, self.depression_eligibility = stdp_step(
                self.weights,
                spikes,
                self.depression_eligibility,
                self.potentiation_eligibility,
                self.potentiation_increment,
                self._dt_potentiation_tau,
                self._dt_depression_tau,
            )

        # Implement homeostasis
        if self.use_homeostasis:
            if homeostasis is None:
                raise ValueError("homeostasis must be provided if use_homeostasis is True")
            # <homeostasis> is the postsynaptic neurons estimate of the ratio between the
            # current firing rate and the set point firing rate.
            # Neurons scale their synaptic weight by this factor at a given time constant,
            # and each synapse group can vary the amount of homeostasis with self.homeostasis_scale
            homeostasis_factor = homeostasis * self.homeostasis_scale * self._dt_homeostasis_tau
            self.weights += homeostasis_factor * self.weights

        # Implement replacement
        if self.use_replacement:
            synapses_to_replace = self.weights < self.min_weight
            n_synapses_to_replace = np.sum(synapses_to_replace)
            if n_synapses_to_replace > 0:
                self.weights[synapses_to_replace] = self.new_weight
                self.presynaptic_source[synapses_to_replace] = self.rng.integers(
                    0, self.num_presynaptic_neurons, n_synapses_to_replace
                )

        # Clip the weights to the range [0, self.max_weight]
        clip_weights(self.weights, 0, self.max_weight)

    def get_current(self, vm: float):
        """Compute the current from the synapse group.

        Uses Ohm's law to compute the driving force based on the reversal potential and the
        membrane voltage then multiplies by the total conductance to get the current.

        Args:
            vm: The membrane voltage of the postsynaptic neuron.

        Returns:
            The current from the synapse group.
        """
        driving_force = self.reversal - vm
        return self.conductance * driving_force
