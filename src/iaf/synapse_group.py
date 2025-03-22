from typing import Dict, Any, Literal, Union, Optional, List, Type, Tuple, ClassVar
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import yaml
import numpy as np
from numba import njit
from ..utils import rng, resolve_dataclass
from .config import BaseSynapseConfig, SourcedSynapseConfig, DirectSynapseConfig, PlasticityConfig, ReplacementConfig


def create_synapse_group(config: BaseSynapseConfig) -> "SynapseGroup":
    """Create a synapse group instance based on the configuration type.

    This factory function creates the appropriate synapse group object
    based on the type of configuration provided.

    Parameters
    ----------
    config : BaseSynapseConfig
        The configuration for the synapse group. Must be one of the following types:
        - SourcedSynapseConfig: Creates a SourcedSynapseGroup
        - DirectSynapseConfig: Creates a DirectSynapseGroup

    Returns
    -------
    SynapseGroup
        The created synapse group instance.

    Raises
    ------
    ValueError
        If the configuration type is not supported.
    """
    if isinstance(config, SourcedSynapseConfig):
        return SourcedSynapseGroup.from_config(config)
    elif isinstance(config, DirectSynapseConfig):
        return DirectSynapseGroup.from_config(config)
    else:
        raise ValueError(f"Invalid synapse group config: {config}")


@dataclass
class SpikeGenerator:
    """A class for efficiently generating Poisson spikes.

    This class provides a mechanism to efficiently generate Poisson-distributed
    spikes by pre-generating random numbers and reusing them in batches. This
    improves performance for simulations that require many spike generation steps.

    Attributes
    ----------
    num_neurons : int
        The number of neurons to generate spikes for.
    dt : float
        The time step of the simulation in seconds.
    max_batch : int
        The maximum batch size of random numbers to pre-generate.
    batch_idx : int
        The current index in the batch.
    current_batch : np.ndarray or None
        The current batch of random numbers.
    batch_size : int
        The current batch size.
    """

    num_neurons: int
    dt: float
    max_batch: int = 10

    def __post_init__(self) -> None:
        self.batch_idx = 0
        self.current_batch = None
        self.batch_size = 0

        # Warmup the numba function
        generate_spikes_numba(rng.random(self.num_neurons), 1.0, self.dt)

    def initialize_batch(self, steps_remaining: Optional[int] = None) -> None:
        """Create a batch of random numbers for reuse in poisson spike generation.

        Pre-generates a batch of random numbers to be used for spike generation,
        which is more efficient than generating new random numbers at each step.

        Parameters
        ----------
        steps_remaining : int, optional
            The number of time steps remaining in the simulation. If provided,
            the batch size will be min(max_batch, steps_remaining).
        """
        batch_size = min(self.max_batch, steps_remaining or self.max_batch)
        self.current_batch = rng.random((batch_size, self.num_neurons))
        self.batch_idx = 0
        self.batch_size = batch_size

    def get_spikes(self, rate: np.ndarray, steps_remaining: Optional[int] = None) -> np.ndarray:
        """Generate spikes for the current timestep.

        Generates Poisson-distributed spikes based on the provided rates.
        If the current batch of random numbers is exhausted, a new batch
        will be initialized.

        Parameters
        ----------
        rate : np.ndarray
            Array of firing rates for each neuron.
        steps_remaining : int, optional
            The number of time steps remaining in the simulation.

        Returns
        -------
        np.ndarray
            Boolean array of shape (num_neurons,) indicating which neurons spiked.
        """
        if self.batch_idx >= self.batch_size:
            self.initialize_batch(steps_remaining)

        spikes = generate_spikes_numba(self.current_batch[self.batch_idx], rate, self.dt)
        self.batch_idx += 1
        return spikes


@dataclass
class PlasticityParams:
    """Parameters for synaptic plasticity.

    This class encapsulates the parameters for different forms of synaptic
    plasticity including spike-timing-dependent plasticity (STDP) and homeostasis.

    Attributes
    ----------
    use_stdp : bool
        Whether to use spike-timing-dependent plasticity.
    stdp_rate : float
        The learning rate for weight changes during STDP.
    depression_potentiation_ratio : float
        The ratio of depression to potentiation magnitudes in STDP.
        Values > 1 cause stronger depression than potentiation.
    potentiation_tau : float
        The time constant for the decay of the potentiation eligibility trace in seconds.
    depression_tau : float
        The time constant for the decay of the depression eligibility trace in seconds.
    use_homeostasis : bool
        Whether to use homeostatic plasticity to regulate firing rates.
    homeostasis_tau : float
        The time constant of homeostatic plasticity in seconds.
    homeostasis_scale : float
        The scaling factor applied to homeostatic plasticity.
    """

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
    """Parameters for synapse replacement.

    This class contains parameters controlling the replacement of weak synapses.
    Synapse replacement is a form of structural plasticity where synapses with
    weights below a threshold are removed and replaced with new synapses.

    Attributes
    ----------
    use_replacement : bool
        Whether to use synapse replacement.
    lose_synapse_ratio : float
        The ratio of max_weight below which a synapse is considered for replacement.
        For example, if lose_synapse_ratio=0.01 and max_weight=1, synapses with
        weights < 0.01 will be replaced.
    new_synapse_ratio : float
        The ratio of max_weight used to initialize new replacement synapses.
        For example, if new_synapse_ratio=0.01 and max_weight=1, new synapses
        will be initialized with a weight of 0.01.
    """

    use_replacement: bool = False  # Whether to use replacement
    lose_synapse_ratio: float = 0.01  # The ratio of maximum weight that causes a synapse to be lost
    new_synapse_ratio: float = 0.01  # The ratio of maximum weight that new synapses are initialized to


@dataclass
class SourceParams:
    """Parameters for mapping presynaptic sources to synapses.

    This class controls how synapses are connected to presynaptic neurons in a
    source population. It maintains a mapping from each synapse to its
    corresponding presynaptic neuron.

    Attributes
    ----------
    num_synapses : int
        The number of synapses in the group.
    num_presynaptic_neurons : int
        The number of neurons in the presynaptic source population.
    source_rule : str
        The rule for assigning presynaptic sources to synapses.
        Options:
        - "random": Randomly assign sources to synapses
        - "divided": Evenly divide synapses among sources (requires num_synapses
          to be divisible by num_presynaptic_neurons)
        - "random-restricted": Randomly assign sources from a subset of valid sources
    valid_sources : np.ndarray or None
        Array of valid source indices to sample from when source_rule is
        "random-restricted". Must be provided in that case.
    presynaptic_source : np.ndarray
        Array of shape (num_synapses,) mapping each synapse to its presynaptic source.
    """

    num_synapses: int
    num_presynaptic_neurons: int
    source_rule: Literal["random", "divided", "random-restricted"]
    valid_sources: np.ndarray | None = field(repr=False)
    presynaptic_source: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.valid_sources is not None:
            self.valid_sources = np.array(self.valid_sources)
            if any(self.valid_sources < 0) or any(self.valid_sources >= self.num_presynaptic_neurons):
                raise ValueError(
                    f"valid_sources must be an array of integers between 0 and {self.num_presynaptic_neurons - 1}"
                )
        self._generate_presynaptic_source()

    def _generate_presynaptic_source(self) -> None:
        """Generate the mapping from synapses to presynaptic sources.

        Creates an array that maps each synapse to its presynaptic source neuron
        based on the specified source_rule.

        Raises
        ------
        ValueError
            If the source_rule is invalid or if constraints are not met
            (e.g., num_synapses not divisible by num_presynaptic_neurons for 'divided').
        """
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
        elif self.source_rule == "divided-restricted":
            num_valid_sources = len(self.valid_sources)
            if self.num_synapses % num_valid_sources != 0:
                raise ValueError(
                    f"num_synapses must be divisible by len(valid_sources) for source_rule == 'divided-restricted'"
                )
            num_source_per_input = int(self.num_synapses / num_valid_sources)
            source_array = np.repeat(self.valid_sources.reshape(-1, 1), num_source_per_input, axis=1)
            source = source_array.flatten()
        else:
            raise ValueError(f"Invalid source rule: {self.source_rule}")
        self.presynaptic_source = source


@dataclass
class InitializationParams:
    """Parameters for synapse weight initialization.

    This class controls how synaptic weights are initialized when a synapse group
    is created or reset.

    Attributes
    ----------
    min_weight : float
        The minimum fraction of max_weight for initialization.
        Weights will be initialized uniformly between min_weight * max_weight
        and max_weight * max_weight.
    max_weight : float
        The maximum fraction of max_weight for initialization.
    """

    min_weight: float = 0.1  # Minimum fraction of max_weight for initialization
    max_weight: float = 1.0  # Maximum fraction of max_weight for initialization


class SynapseGroup(ABC):
    """Abstract base class for groups of synapses with shared properties.

    This class defines the interface and common properties for different types
    of synapse groups, whether they receive input from a source population or
    direct inputs to each synapse. It handles synaptic conductance dynamics,
    plasticity mechanisms, and the processing of presynaptic spikes.

    Subclasses must implement the transform_input_rates method to handle
    the specific way inputs are routed to synapses.

    Attributes
    ----------
    num_synapses : int
        The number of synapses in the group.
    max_weight : float
        The maximum synaptic weight in units of conductance.
    reversal : float
        The reversal potential of the synapses in Volts.
    tau : float
        The synaptic time constant in seconds.
    dt : float
        The time step of the simulation in seconds.
    source_population : str
        The name of the source population providing inputs to the synapses.
    conductance_threshold : float
        The threshold for converting weights to conductance, as a fraction of max_weight.
        Synapses with weights below this threshold do not contribute to conductance.
    min_conductance : float
        The minimum conductance in units of conductance (conductance_threshold * max_weight).
    initialization_params : InitializationParams
        Parameters for weight initialization.
    plastic : bool
        Whether the synapses are plastic (weights can change).
    plasticity_params : PlasticityParams
        Parameters for synaptic plasticity mechanisms.
    weights : np.ndarray
        The weights of each synapse.
    conductance : float
        The current synaptic conductance.
    _spike_generator : SpikeGenerator
        Generator for presynaptic spikes based on input rates.
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

    def __repr__(self) -> str:
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
        conductance_threshold: Optional[float] = None,
        plasticity_params: Optional[Union[PlasticityParams, Dict[str, Any]]] = None,
        initialization_params: Optional[Union[InitializationParams, Dict[str, Any]]] = None,
    ) -> None:
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

    def update_depression_potention_ratio(self, dp_ratio: float) -> None:
        self.plasticity_params.depression_potentiation_ratio = dp_ratio
        if self.plastic and self.plasticity_params.use_stdp:
            self.depression_increment = (
                self.plasticity_params.stdp_rate
                * self.plasticity_params.depression_potentiation_ratio
                * self.max_weight
            )

    def _generate_weights(self) -> None:
        """Generate initial weights for all synapses.

        Weights are initialized uniformly between min_weight and max_weight,
        as specified in initialization_params, and then scaled by max_weight
        in conductance determined by the synapse group's parameters.
        """
        weight_fractions = rng.uniform(
            self.initialization_params.min_weight, self.initialization_params.max_weight, self.num_synapses
        )
        self.weights = weight_fractions * self.max_weight

    def initialize(self, reset_weights: bool = False, reset_sources: bool = False) -> None:
        """Initialize the synapse group to its default state.

        Resets the conductance and eligibility traces. Optionally resets the
        weights and/or presynaptic source mappings.

        Parameters
        ----------
        reset_weights : bool, optional
            Whether to reset synaptic weights to new random values, default is False.
        reset_sources : bool, optional
            Whether to regenerate the mapping from synapses to presynaptic sources
            (only applies to SourcedSynapseGroup), default is False.
        """
        self.conductance = 0.0
        if self.plastic and self.plasticity_params.use_stdp:
            self.potentiation_eligibility = np.zeros(self.num_synapses)
            self.depression_eligibility = 0.0
        if reset_weights:
            self._generate_weights()
        if reset_sources and isinstance(self, SourcedSynapseGroup):
            self.source_params._generate_presynaptic_source()

    def postsynaptic_spike(self) -> None:
        """Process a postsynaptic spike for STDP.

        When the postsynaptic neuron spikes, this method:
        1. Increments the depression eligibility trace
        2. Updates weights based on the potentiation eligibility trace
        3. Clips weights to the valid range

        This is only active if plasticity is enabled with STDP.
        """
        if self.plastic and self.plasticity_params.use_stdp:
            # Whenever a postsynaptic spike occurs, the depression eligibility trace is incremented
            # We make it negative because we want to depress the synapse (and it's a linear STDP model so it's additive)
            self.depression_eligibility -= self.depression_increment

            # Postsynaptic spikes cause potentiation in proportion to the potentiation eligibility trace
            self.weights += self.potentiation_eligibility

            # Weights are clipped to the range [0, self.max_weight]
            clip_weights(self.weights, 0, self.max_weight)

    def step(self, input_rates: np.ndarray, homeostasis: Optional[float] = None) -> None:
        """Step the synapse group forward by one time step.

        This method processes input rates to generate presynaptic spikes,
        updates the synaptic conductance, and applies plasticity mechanisms.

        The process includes:
        1. Transform input rates to the correct format for this synapse group
        2. Generate presynaptic spikes based on input rates
        3. Update synaptic conductance based on spikes and weights
        4. Apply STDP plasticity if enabled
        5. Apply homeostatic plasticity if enabled
        6. Apply synapse replacement if enabled
        7. Clip weights to the valid range

        Parameters
        ----------
        input_rates : np.ndarray
            Input firing rates for each presynaptic neuron.
        homeostasis : float, optional
            Homeostatic factor from the postsynaptic neuron, which is the
            log ratio between the target rate and estimated rate.
            Only used if homeostasis is enabled.
        """
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

    def get_current(self, vm: float) -> float:
        """Calculate the synaptic current based on conductance and driving force.

        The current is calculated as I = g * (E - V), where g is the conductance,
        E is the reversal potential, and V is the membrane voltage.

        Parameters
        ----------
        vm : float
            Membrane potential of the postsynaptic neuron in Volts.

        Returns
        -------
        float
            Current produced by the synapse group in Amperes.
        """
        return self.conductance * (self.reversal - vm)

    def handle_replacement(self) -> None:
        """Handle replacement of weak synapses.

        This is a placeholder method in the base class, which is overridden
        by subclasses that implement synapse replacement mechanisms.
        """
        pass

    @abstractmethod
    def transform_input_rates(self, input_rates: np.ndarray) -> np.ndarray:
        """Transform input rates to the format needed by this synapse group.

        This abstract method must be implemented by subclasses to handle the
        specific way inputs are routed to synapses (e.g., direct mapping or
        source-based mapping).

        Parameters
        ----------
        input_rates : np.ndarray
            Input rates to transform.

        Returns
        -------
        np.ndarray
            Transformed input rates, one per synapse.

        Raises
        ------
        ValueError
            If the input rates are incompatible with this synapse group.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: BaseSynapseConfig) -> "SynapseGroup":
        """Create a synapse group from a configuration object.

        Parameters
        ----------
        config : BaseSynapseConfig
            The configuration for the synapse group.

        Returns
        -------
        SynapseGroup
            A new synapse group instance.
        """
        pass

    @classmethod
    @abstractmethod
    def from_yaml(cls, fpath: Path) -> "SynapseGroup":
        """Create a synapse group from a YAML configuration file.

        Parameters
        ----------
        fpath : Path
            The path to the YAML configuration file.

        Returns
        -------
        SynapseGroup
            A new synapse group instance.
        """


class SourcedSynapseGroup(SynapseGroup):
    """A synapse group that receives inputs from a source population.

    This class represents a group of synapses that are connected to a source
    population. Each synapse is connected to a specific presynaptic neuron
    in the source population, determined by the source_params.

    This group supports synapse replacement, where weak synapses can be
    removed and replaced with new connections to different presynaptic neurons.

    Attributes
    ----------
    source_params : SourceParams
        Parameters controlling how synapses are mapped to presynaptic sources.
    replacement_params : ReplacementParams
        Parameters controlling the replacement of weak synapses.
    min_weight : float
        Minimum weight below which a synapse is considered for replacement.
    new_weight : float
        Weight assigned to newly created replacement synapses.
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
        conductance_threshold: Optional[float] = None,
        plastic: bool = True,
        plasticity_params: Optional[Union[PlasticityParams, Dict[str, Any]]] = None,
        initialization_params: Optional[Union[InitializationParams, Dict[str, Any]]] = None,
        source_params: Optional[Union[SourceParams, Dict[str, Any]]] = None,
        replacement_params: Optional[Union[ReplacementParams, Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a sourced synapse group.

        Parameters
        ----------
        num_synapses : int
            Number of synapses in the group.
        max_weight : float
            Maximum weight for each synapse (in units of conductance).
        reversal : float
            Reversal potential (in V).
        tau : float
            Time constant (in seconds).
        dt : float
            Time step (in seconds).
        source_population : str
            The name of the source population providing inputs to the synapses.
        conductance_threshold : float, optional
            Threshold for conductance activation (in relative units of max_weight).
        plastic : bool, optional
            Whether the weights are plastic, default is True.
        plasticity_params : PlasticityParams or dict, optional
            Parameters for plasticity mechanisms.
        initialization_params : InitializationParams or dict, optional
            Parameters for weight initialization.
        source_params : SourceParams or dict, optional
            Parameters for mapping synapses to presynaptic sources.
        replacement_params : ReplacementParams or dict, optional
            Parameters for synapse replacement.

        Raises
        ------
        ValueError
            If num_synapses doesn't match source_params.num_synapses.
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
    def from_yaml(cls, fpath: Path) -> "SourcedSynapseGroup":
        """Create a sourced synapse group from a YAML configuration file.

        Parameters
        ----------
        fpath : Path
            The path to the YAML configuration file.

        Returns
        -------
        SourcedSynapseGroup
            A new sourced synapse group instance.
        """
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(SourcedSynapseConfig.model_validate(config))

    @classmethod
    def from_config(cls, config: SourcedSynapseConfig) -> "SourcedSynapseGroup":
        """Create a sourced synapse group from a configuration object.

        Parameters
        ----------
        config : SourcedSynapseConfig
            The configuration for the sourced synapse group.

        Returns
        -------
        SourcedSynapseGroup
            A new sourced synapse group instance.
        """
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

    def handle_replacement(self) -> None:
        """Replace weak synapses with new ones connected to different sources.

        This method implements a form of structural plasticity where synapses
        with weights below the minimum threshold are removed and replaced with
        new synapses connected to different presynaptic neurons. The new synapses
        are initialized with a small weight.

        Only active if plastic=True and replacement_params.use_replacement=True.

        Raises
        ------
        ValueError
            If replacement is used with an incompatible source_rule.
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
                elif self.source_params.source_rule == "divided-restricted":
                    raise ValueError("Replacement plasticity cannot be used with source rule == 'divided-restricted'")
                else:
                    raise ValueError(f"Invalid source rule: {self.source_params.source_rule}")
                self.source_params.presynaptic_source[synapses_to_replace] = new_sources

    def transform_input_rates(self, input_rates: np.ndarray) -> np.ndarray:
        """Map input rates from the source population to each synapse.

        This method takes the input rates for each presynaptic neuron in the
        source population and maps them to the appropriate synapses based on
        the presynaptic_source mapping.

        Parameters
        ----------
        input_rates : np.ndarray
            Input rates for each presynaptic neuron in the source population.

        Returns
        -------
        np.ndarray
            Input rates mapped to each synapse based on its presynaptic source.

        Raises
        ------
        ValueError
            If the number of input rates doesn't match the number of presynaptic neurons.
        """
        if len(input_rates) != self.source_params.num_presynaptic_neurons:
            raise ValueError(
                f"Number of input rates provided ({len(input_rates)}) does not match "
                f"number of presynaptic neurons ({self.source_params.num_presynaptic_neurons})."
            )
        return input_rates[self.source_params.presynaptic_source]


class DirectSynapseGroup(SynapseGroup):
    """A synapse group with direct one-to-one mapping from inputs to synapses.

    This class represents a group of synapses where each synapse receives input
    directly from a corresponding input, with a one-to-one mapping. The number
    of inputs must match the number of synapses.

    This type of synapse group is useful when the input rates already have the
    same dimensionality as the number of synapses, and no mapping from a source
    population is required.

    Unlike SourcedSynapseGroup, this class does not support synapse replacement.
    """

    def __init__(
        self,
        num_synapses: int,
        max_weight: float,
        reversal: float,
        tau: float,
        dt: float,
        source_population: str,
        conductance_threshold: Optional[float] = None,
        plastic: bool = True,
        plasticity_params: Optional[Union[PlasticityParams, Dict[str, Any]]] = None,
        initialization_params: Optional[Union[InitializationParams, Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a direct synapse group.

        Parameters
        ----------
        num_synapses : int
            Number of synapses in the group.
        max_weight : float
            Maximum weight for each synapse (in units of conductance).
        reversal : float
            Reversal potential (in V).
        tau : float
            Time constant (in seconds).
        dt : float
            Time step (in seconds).
        source_population : str
            The name of the source population providing inputs to the synapses.
        conductance_threshold : float, optional
            Threshold for conductance activation (in relative units of max_weight).
        plastic : bool, optional
            Whether the weights are plastic, default is True.
        plasticity_params : PlasticityParams or dict, optional
            Parameters for plasticity mechanisms.
        initialization_params : InitializationParams or dict, optional
            Parameters for weight initialization.
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

    def transform_input_rates(self, input_rates: np.ndarray) -> np.ndarray:
        """Pass input rates directly to synapses with validation.

        This method checks that the number of input rates matches the number
        of synapses, and returns the input rates unchanged.

        Parameters
        ----------
        input_rates : np.ndarray
            Input rates for each synapse.

        Returns
        -------
        np.ndarray
            The same input rates, unchanged.

        Raises
        ------
        ValueError
            If the number of input rates doesn't match the number of synapses.
        """
        if len(input_rates) != self.num_synapses:
            raise ValueError(
                f"Number of input rates provided ({len(input_rates)}) does not match "
                f"number of synapses ({self.num_synapses})."
            )
        return input_rates

    @classmethod
    def from_yaml(cls, fpath: Path) -> "DirectSynapseGroup":
        """Create a direct synapse group from a YAML configuration file.

        Parameters
        ----------
        fpath : Path
            The path to the YAML configuration file.

        Returns
        -------
        DirectSynapseGroup
            A new direct synapse group instance.
        """
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(DirectSynapseConfig.model_validate(config))

    @classmethod
    def from_config(cls, config: DirectSynapseConfig) -> "DirectSynapseGroup":
        """Create a direct synapse group from a configuration object.

        Parameters
        ----------
        config : DirectSynapseConfig
            The configuration for the direct synapse group.

        Returns
        -------
        DirectSynapseGroup
            A new direct synapse group instance.
        """
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


@njit
def generate_spikes_numba(random_values: np.ndarray, rate: np.ndarray, dt: float) -> np.ndarray:
    """Generate Poisson-distributed spikes using pre-generated random values.

    Parameters
    ----------
    random_values : np.ndarray
        Array of uniform random values between 0 and 1.
    rate : np.ndarray or float
        Firing rate(s) in Hz.
    dt : float
        Time step in seconds.

    Returns
    -------
    np.ndarray
        Boolean array indicating which neurons spiked.
    """
    return random_values < (rate * dt)


@njit
def compute_conductance(spikes: np.ndarray, weights: np.ndarray, threshold: float) -> float:
    """Compute the total synaptic conductance from spikes and weights.

    Only synapses that have a weight above the threshold contribute to
    the conductance when they spike.

    Parameters
    ----------
    spikes : np.ndarray
        Boolean array indicating which synapses received a spike.
    weights : np.ndarray
        Array of synaptic weights.
    threshold : float
        Minimum weight for a synapse to contribute to conductance.

    Returns
    -------
    float
        Total conductance from active synapses.
    """
    return np.sum(spikes * weights * (weights > threshold))


@njit
def clip_weights(weights: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """Clip weights to range [min_val, max_val] in-place.

    Parameters
    ----------
    weights : np.ndarray
        Array of weights to clip.
    min_val : float
        Minimum allowed weight value.
    max_val : float
        Maximum allowed weight value.

    Returns
    -------
    np.ndarray
        The clipped weights array (same object as input).
    """
    np.clip(weights, min_val, max_val, out=weights)
    return weights


@njit
def stdp_step(
    weights: np.ndarray,
    spikes: np.ndarray,
    depression_eligibility: float,
    potentiation_eligibility: np.ndarray,
    potentiation_increment: float,
    potentiation_decay_factor: float,
    depression_decay_factor: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Apply one step of STDP to update weights and eligibility traces.

    This function implements the core STDP algorithm:
    1. Update weights based on presynaptic spikes and depression eligibility
    2. Update potentiation eligibility trace based on presynaptic spikes
    3. Decay both eligibility traces

    Parameters
    ----------
    weights : np.ndarray
        Array of synaptic weights.
    spikes : np.ndarray
        Boolean array indicating which synapses received a spike.
    depression_eligibility : float
        Current value of the depression eligibility trace.
    potentiation_eligibility : np.ndarray
        Array of potentiation eligibility trace values.
    potentiation_increment : float
        Amount to increment potentiation eligibility by for each spike.
    potentiation_decay_factor : float
        Factor by which to decay potentiation eligibility (dt/tau).
    depression_decay_factor : float
        Factor by which to decay depression eligibility (dt/tau).

    Returns
    -------
    tuple
        Tuple containing:
        - Updated weights array
        - Updated potentiation eligibility array
        - Updated depression eligibility value
    """
    weights += spikes * depression_eligibility
    potentiation_eligibility += spikes * potentiation_increment - potentiation_eligibility * potentiation_decay_factor
    depression_eligibility -= depression_eligibility * depression_decay_factor
    return weights, potentiation_eligibility, depression_eligibility
