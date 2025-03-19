from typing import Dict, Any, Literal
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import yaml
import numpy as np
from numba import njit
from ..utils import rng, resolve_dataclass
from .config import BaseSynapseConfig, SourcedSynapseConfig, DirectSynapseConfig, PlasticityConfig, ReplacementConfig


def create_synapse_group(config: BaseSynapseConfig) -> "SynapseGroup":
    if isinstance(config, SourcedSynapseConfig):
        return SourcedSynapseGroup.from_config(config)
    elif isinstance(config, DirectSynapseConfig):
        return DirectSynapseGroup.from_config(config)
    else:
        raise ValueError(f"Invalid synapse group config: {config}")


@dataclass
class SpikeGenerator:
    num_neurons: int
    dt: float
    max_batch: int = 10

    def __post_init__(self):
        self.batch_idx = 0
        self.current_batch = None
        self.batch_size = 0

        # Warmup the numba function
        generate_spikes_numba(rng.random(self.num_neurons), 1.0, self.dt)

    def initialize_batch(self, steps_remaining=None):
        """Create a batch of random numbers for reuse in poisson spike generation."""
        batch_size = min(self.max_batch, steps_remaining or self.max_batch)
        self.current_batch = rng.random((batch_size, self.num_neurons))
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
    use_stdp: bool = False  # Whether to use STDP
    stdp_rate: float = 0.01  # The rate of potentiation/depression
    depression_potentiation_ratio: float = 1.1  # The ratio of depression to potentiation
    potentiation_tau: float = 0.02  # The time constant of potentiation
    depression_tau: float = 0.02  # The time constant of depression
    use_homeostasis: bool = False  # Whether to use homeostasis
    homeostasis_tau: float = 20  # The time constant of homeostasis
    homeostasis_scale: float = 1.0  # The scale of homeostasis


@dataclass
class ReplacementParams:
    use_replacement: bool = False  # Whether to use replacement
    lose_synapse_ratio: float = 0.01  # The ratio of maximum weight that causes a synapse to be lost
    new_synapse_ratio: float = 0.01  # The ratio of maximum weight that new synapses are initialized to


@dataclass
class SourceParams:
    num_synapses: int
    num_presynaptic_neurons: int
    source_rule: Literal["random", "divided", "random-restricted"]
    valid_sources: np.ndarray | None = field(repr=False)
    presynaptic_source: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        if self.valid_sources is not None:
            self.valid_sources = np.array(self.valid_sources)
        self._generate_presynaptic_source()

    def _generate_presynaptic_source(self):
        if self.source_rule == "random":
            source = rng.integers(0, self.num_presynaptic_neurons, self.num_synapses)
        elif self.source_rule == "random-restricted":
            source = self.valid_sources[rng.integers(0, len(self.valid_sources), self.num_synapses)]
        elif self.source_rule == "divided":
            if self.num_synapses % self.num_presynaptic_neurons != 0:
                raise ValueError(
                    f"num_synapses must be divisible by num_presynaptic_neurons for source_rule == 'divided'"
                )
            num_source_per_input = int(self.num_synapses / self.num_presynaptic_neurons)
            source_array = np.repeat(
                np.arange(self.num_presynaptic_neurons).reshape(-1, 1), num_source_per_input, axis=1
            )
            source = source_array.flatten()
        else:
            raise ValueError(f"Invalid source rule: {self.source_rule}")
        self.presynaptic_source = source


@dataclass
class InitializationParams:
    min_weight: float = 0.1  # Minimum fraction of max_weight for initialization
    max_weight: float = 1.0  # Maximum fraction of max_weight for initialization


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
    source_population: str  # The name of the source population of the synapse group
    conductance_threshold: float  # in relative units of max_weight
    min_conductance: float  # in units of conductance
    initialization_params: InitializationParams

    # Plasticity related properties
    plastic: bool = False
    plasticity_params: PlasticityParams

    # State variables
    weights: np.ndarray  # Weight of each synapse
    conductance: float  # Current conductance

    # RNG for random processes
    rng: np.random.Generator

    def __repr__(self):
        group_type = type(self).__name__
        return f"{group_type}(num_synapses={self.num_synapses}, plastic={self.plastic})"

    def _init_base_params(
        self,
        num_synapses: int,
        max_weight: float,
        reversal: float,
        tau: float,
        dt: float,
        plastic: bool,
        source_population: str,
        conductance_threshold: float | None = None,
        plasticity_params: PlasticityParams | Dict[str, Any] | None = None,
        initialization_params: InitializationParams | Dict[str, Any] | None = None,
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
        plastic : bool
            Whether the weights are plastic
        source_population : str
            The name of the source population of the synapse group. Will be used by a
            simulation to determine which input rates to pass to the synapse group.
        conductance_threshold : float, optional
            Threshold for conductance activation (in relative units of max_weight)
        plasticity_params : PlasticityParams or dict, optional
            Parameters for plasticity mechanisms
        initialization_params : InitializationParams or dict, optional
            Parameters for weight initialization
        """
        # Create base parameters
        self.num_synapses = num_synapses
        self.max_weight = max_weight
        self.reversal = reversal
        self.tau = tau
        self.dt = dt
        self.plastic = plastic
        self.conductance_threshold: float = conductance_threshold or 0.0
        self.min_conductance = self.conductance_threshold * self.max_weight
        self.source_population = source_population

        # Set up initialization parameters
        self.initialization_params = resolve_dataclass(initialization_params, InitializationParams)

        # Initialize state variables with random weights between min_weight and max_weight
        self._generate_weights()
        self.conductance = 0.0

        # Create a spike generator for the group
        self._spike_generator = SpikeGenerator(num_neurons=self.num_synapses, dt=self.dt)

        # Set up plasticity parameters
        self.plasticity_params = resolve_dataclass(plasticity_params, PlasticityParams)

        # Set up STDP parameters if weights are plastic
        if self.plastic and self.plasticity_params.use_stdp:
            # This determines how much the weight can change for a single pre/post pairing
            self.potentiation_increment = self.plasticity_params.stdp_rate * self.max_weight
            self.depression_increment = (
                self.plasticity_params.stdp_rate
                * self.plasticity_params.depression_potentiation_ratio
                * self.max_weight
            )

            # Buffers to store the potentiation/depression for the current timestep
            self.potentiation_eligibility: np.ndarray = np.zeros(self.num_synapses)
            self.depression_eligibility: float = 0.0

        # Precompute dt / tau divisions for optimization
        self._dt_tau = self.dt / self.tau
        if self.plastic:
            if self.plasticity_params.use_stdp:
                self._dt_potentiation_tau = self.dt / self.plasticity_params.potentiation_tau
                self._dt_depression_tau = self.dt / self.plasticity_params.depression_tau
            if self.plasticity_params.use_homeostasis:
                self._dt_homeostasis_tau = self.dt / self.plasticity_params.homeostasis_tau

    def update_depression_potention_ratio(self, dp_ratio: float):
        self.plasticity_params.depression_potentiation_ratio = dp_ratio
        if self.plastic and self.plasticity_params.use_stdp:
            self.depression_increment = (
                self.plasticity_params.stdp_rate
                * self.plasticity_params.depression_potentiation_ratio
                * self.max_weight
            )

    def _generate_weights(self):
        weight_fractions = rng.uniform(
            self.initialization_params.min_weight, self.initialization_params.max_weight, self.num_synapses
        )
        self.weights = weight_fractions * self.max_weight

    def initialize(self, reset_weights: bool = False, reset_sources: bool = False):
        """Initialize the synapse group."""
        self.conductance = 0.0
        if self.plastic and self.plasticity_params.use_stdp:
            self.potentiation_eligibility = np.zeros(self.num_synapses)
            self.depression_eligibility = 0.0
        if reset_weights:
            self._generate_weights()
        if reset_sources and isinstance(self, SourcedSynapseGroup):
            self.source_params._generate_presynaptic_source()

    def postsynaptic_spike(self):
        """Implement STDP updates in the case of a postsynaptic spike."""
        if self.plastic and self.plasticity_params.use_stdp:
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
        if self.plastic and self.plasticity_params.use_stdp:
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
        if self.plastic and self.plasticity_params.use_homeostasis and homeostasis is not None:
            # <homeostasis> is the postsynaptic neuron's estimate of the ratio between the
            # current firing rate and the set point firing rate.
            # Neurons scale their synaptic weight by this factor at a given time constant,
            # and each synapse group can vary the amount of homeostasis with self.homeostasis_scale
            homeostasis_factor = homeostasis * self.plasticity_params.homeostasis_scale * self._dt_homeostasis_tau
            self.weights += homeostasis_factor * self.weights

        # Implement replacement only if weights are plastic
        if self.plastic:
            self.handle_replacement()

        # Clip the weights to the range [0, self.max_weight]
        if self.plastic:
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

    @classmethod
    @abstractmethod
    def from_config(cls, config: BaseSynapseConfig):
        """Create a synapse group from a configuration object.

        Args:
            config: The configuration for the synapse group.

        Returns:
            A new synapse group instance.
        """
        pass

    @classmethod
    @abstractmethod
    def from_yaml(cls, fpath: Path):
        """Create a synapse group from a YAML configuration file.

        Args:
            fpath: The path to the YAML configuration file.
        """


class SourcedSynapseGroup(SynapseGroup):
    """
    A synapse group that receives inputs from a source population.

    This group internally manages which synapse is connected to which source.
    """

    source_params: SourceParams
    replacement_params: ReplacementParams

    def __init__(
        self,
        num_synapses: int,
        max_weight: float,
        reversal: float,
        tau: float,
        dt: float,
        source_population: str,
        conductance_threshold: float | None = None,
        plastic: bool = True,
        plasticity_params: PlasticityParams | Dict[str, Any] | None = None,
        initialization_params: InitializationParams | Dict[str, Any] | None = None,
        source_params: SourceParams | Dict[str, Any] | None = None,
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
        source_population : str
            The name of the source population of the synapse group. Will be used by a
            simulation to determine which input rates to pass to the synapse group.
        conductance_threshold : float, optional
            Threshold for conductance activation (in relative units of max_weight)
        plastic : bool, optional
            Whether the weights are plastic
        plasticity_params : PlasticityParams or dict, optional
            Parameters for plasticity mechanisms
        initialization_params : dict, optional
            Parameters for weight initialization
        source_params : SourceParams or dict, optional
            Parameters for the source population
        replacement_params : ReplacementParams or dict, optional
            Parameters for synapse replacement
        """
        self._init_base_params(
            num_synapses=num_synapses,
            max_weight=max_weight,
            reversal=reversal,
            tau=tau,
            dt=dt,
            plastic=plastic,
            source_population=source_population,
            conductance_threshold=conductance_threshold,
            plasticity_params=plasticity_params,
            initialization_params=initialization_params,
        )

        # Set up source parameters for routing source inputs to these synapses
        self.source_params = resolve_dataclass(source_params, SourceParams)

        # Check if num_synapses match
        if self.num_synapses != self.source_params.num_synapses:
            raise ValueError(
                f"Number of synapses ({self.num_synapses}) does not match "
                f"number of synapses in source population ({self.source_params.num_synapses})."
            )

        # Set up replacement parameters
        self.replacement_params = resolve_dataclass(replacement_params, ReplacementParams)

        if self.replacement_params.use_replacement:
            self.min_weight = self.replacement_params.lose_synapse_ratio * self.max_weight
            self.new_weight = self.replacement_params.new_synapse_ratio * self.max_weight

        # Set up initialization parameters
        self.initialization_params = initialization_params

    @classmethod
    def from_yaml(cls, fpath: Path):
        """Create a synapse group from a YAML configuration file.

        Args:
            fpath: The path to the YAML configuration file.
        """
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(SourcedSynapseConfig.model_validate(config))

    @classmethod
    def from_config(cls, config: SourcedSynapseConfig):
        """Create a sourced synapse group from a configuration object."""
        source_params = None
        if config.source is not None:
            source_params = SourceParams(**config.source.model_dump())

        plasticity_params = None
        if config.plasticity is not None:
            plasticity_params = PlasticityParams(**config.plasticity.model_dump())

        replacement_params = None
        if config.replacement is not None:
            replacement_params = ReplacementParams(**config.replacement.model_dump())

        initialization_params = None
        if config.initialization is not None:
            initialization_params = InitializationParams(**config.initialization.model_dump())

        return cls(
            num_synapses=config.num_synapses,
            max_weight=config.max_weight,
            reversal=config.reversal,
            tau=config.tau,
            dt=config.dt,
            source_population=config.source_population,
            conductance_threshold=config.conductance_threshold,
            plastic=config.plastic,
            plasticity_params=plasticity_params,
            initialization_params=initialization_params,
            source_params=source_params,
            replacement_params=replacement_params,
        )

    def handle_replacement(self):
        """
        Implement synapse replacement: remove weak synapses and create new ones.
        """
        if self.plastic and self.replacement_params.use_replacement:
            synapses_to_replace = self.weights < self.min_weight
            n_synapses_to_replace = np.sum(synapses_to_replace)
            if n_synapses_to_replace > 0:
                self.weights[synapses_to_replace] = self.new_weight
                if self.source_params.source_rule == "random":
                    new_sources = rng.integers(0, self.source_params.num_presynaptic_neurons, n_synapses_to_replace)
                elif self.source_params.source_rule == "random-restricted":
                    new_sources = self.source_params.valid_sources[
                        rng.integers(0, len(self.source_params.valid_sources), n_synapses_to_replace)
                    ]
                elif self.source_params.source_rule == "divided":
                    raise ValueError("Replacement plasticity cannot be used with source rule == 'divided'")
                else:
                    raise ValueError(f"Invalid source rule: {self.source_params.source_rule}")
                self.source_params.presynaptic_source[synapses_to_replace] = new_sources

    def transform_input_rates(self, input_rates: np.ndarray):
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
        if len(input_rates) != self.source_params.num_presynaptic_neurons:
            raise ValueError(
                f"Number of input rates provided ({len(input_rates)}) does not match "
                f"number of presynaptic neurons ({self.source_params.num_presynaptic_neurons})."
            )
        return input_rates[self.source_params.presynaptic_source]


class DirectSynapseGroup(SynapseGroup):
    """
    A synapse group that receives direct inputs to each synapse.

    This group maps inputs to synapses in a 1:1 manner, so the number of inputs
    must match the number of synapses.
    """

    def __init__(
        self,
        num_synapses: int,
        max_weight: float,
        reversal: float,
        tau: float,
        dt: float,
        source_population: str,
        conductance_threshold: float | None = None,
        plastic: bool = True,
        plasticity_params: PlasticityParams | Dict[str, Any] | None = None,
        initialization_params: InitializationParams | Dict[str, Any] | None = None,
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
        source_population : str
            The name of the source population of the synapse group. Will be used by a
            simulation to determine which input rates to pass to the synapse group.
        conductance_threshold : float, optional
            Threshold for conductance activation (in relative units of max_weight)
        plastic : bool, optional
            Whether the weights are plastic
        plasticity_params : PlasticityParams or dict, optional
            Parameters for plasticity mechanisms
        initialization_params : dict, optional
            Parameters for weight initialization
        """
        self._init_base_params(
            num_synapses=num_synapses,
            max_weight=max_weight,
            reversal=reversal,
            tau=tau,
            dt=dt,
            plastic=plastic,
            source_population=source_population,
            conductance_threshold=conductance_threshold,
            plasticity_params=plasticity_params,
            initialization_params=initialization_params,
        )

    @classmethod
    def from_yaml(cls, fpath: Path):
        """Create a synapse group from a YAML configuration file.

        Args:
            fpath: The path to the YAML configuration file.
        """
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(DirectSynapseConfig.model_validate(config))

    @classmethod
    def from_config(cls, config: DirectSynapseConfig):
        """Create a direct synapse group from a configuration object."""
        plasticity_params = None
        if config.plasticity is not None:
            plasticity_params = PlasticityParams(**config.plasticity.model_dump())

        initialization_params = None
        if config.initialization is not None:
            initialization_params = InitializationParams(**config.initialization.model_dump())

        return cls(
            num_synapses=config.num_synapses,
            max_weight=config.max_weight,
            reversal=config.reversal,
            tau=config.tau,
            dt=config.dt,
            source_population=config.source_population,
            conductance_threshold=config.conductance_threshold,
            plastic=config.plastic,
            plasticity_params=plasticity_params,
            initialization_params=initialization_params,
        )

    def transform_input_rates(self, input_rates: np.ndarray):
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
