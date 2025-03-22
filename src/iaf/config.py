from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, List, Literal, Union, TypeVar, Type
from pathlib import Path
import yaml


ConfigT = TypeVar("ConfigT", bound="BaseConfig")


class BaseConfig(BaseModel):
    """Base class for all configuration classes.

    This class extends Pydantic's BaseModel and provides common functionality
    for all configuration classes, such as loading from YAML files.
    """

    @classmethod
    def from_yaml(cls: Type[ConfigT], fpath: Path) -> ConfigT:
        """Load configuration from a YAML file.

        Parameters
        ----------
        fpath : Path
            The path to the YAML configuration file.

        Returns
        -------
        ConfigT
            An instance of the configuration class with values loaded from the YAML file.
        """
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.model_validate(config)


class SourcePopulationConfig(BaseConfig):
    """Base configuration for source populations.

    This abstract base class defines common parameters for all source population types.
    Specific source populations will use subclasses with additional parameters.

    Attributes
    ----------
    type : str
        The type of source population. This determines which subclass will be used.
    tau_stim : float
        The time constant for the stimulus in seconds, determining how frequently
        the input rates change. Default is 0.01.
    dt : float
        The time step in seconds. Default is 0.001.
    """

    type: str = Field(..., description="Type of source population")
    tau_stim: float = Field(0.01, gt=0, description="Time constant for the stimulus in seconds")
    dt: float = Field(0.001, gt=0, description="Time step in seconds")


class SourceGaborConfig(SourcePopulationConfig):
    """Configuration for Gabor source population.

    This class defines parameters for a source population that generates rates
    based on Gabor patterns with controlled orientation and probability.

    Attributes
    ----------
    type : str
        The type of source population, always "gabor".
    edge_probability : float
        Probability of an edge appearing in the stimulus. Default is 1.0.
    concentration : float
        Concentration of the Gabor field, controlling how focused the patterns are.
        Default is 1.0.
    baseline_rate : float
        Baseline firing rate in Hz when no edge is present. Default is 5.0.
    driven_rate : float
        Firing rate in Hz when an edge is present. Default is 45.0.
    """

    type: Literal["gabor"] = Field("gabor", description="Type of source population (gabor)")
    edge_probability: float = Field(1.0, gt=0, le=1, description="Probability of an edge appearing in the stimulus")
    concentration: float = Field(1.0, gt=0, description="Concentration of the Gabor field")
    baseline_rate: float = Field(5.0, gt=0, description="Baseline firing rate")
    driven_rate: float = Field(45.0, gt=0, description="Driven firing rate")


class SourceICAConfig(SourcePopulationConfig):
    """Configuration for ICA source population.

    This class defines parameters for a source population that generates rates
    based on Independent Component Analysis (ICA), allowing for controlled
    statistics and correlations between inputs.

    Attributes
    ----------
    type : str
        The type of source population, always "ica".
    num_inputs : int
        Number of input neurons in the source population. Default is 100.
    num_signals : int
        Number of independent components/signals. Default is 3.
    source_method : str
        Method for generating source loading. Options are "divide", "gauss", or
        "correlated". Default is "gauss".
    source_strength : float
        Signal-to-noise ratio, controlling how strongly the signals influence
        the inputs. Default is 3.0.
    rate_std : float
        Standard deviation of input rates in Hz. Default is 10.0.
    rate_mean : float
        Mean of input rates in Hz. Default is 20.0.
    gauss_source_width : float
        Width of the Gaussian source when source_method is "gauss". Default is 2/5.
    """

    type: Literal["ica"] = Field("ica", description="Type of source population (ica)")
    num_inputs: int = Field(100, ge=1, description="Number of input neurons")
    num_signals: int = Field(3, ge=1, description="Number of independent components/signals")
    source_method: Literal["divide", "gauss", "correlated"] = Field(
        "gauss", description="Method for generating source loading"
    )
    source_strength: float = Field(3.0, gt=0, description="Signal-to-noise ratio")
    rate_std: float = Field(10.0, gt=0, description="Standard deviation of input rates")
    rate_mean: float = Field(20.0, gt=0, description="Mean of input rates")
    gauss_source_width: float = Field(2 / 5, gt=0, description="Width of the Gaussian source")


class SourceCorrelationConfig(SourcePopulationConfig):
    """Configuration for correlation source population.

    This class defines parameters for a source population that generates rates
    with explicitly controlled correlation structure between inputs.

    Attributes
    ----------
    type : str
        The type of source population, always "correlation".
    num_inputs : int
        Number of input neurons in the source population. Default is 100.
    max_correlation : float
        Maximum correlation between any two inputs. Default is 0.4.
    decay_function : str
        Function describing how correlation decays with distance between inputs.
        Currently only "linear" is supported. Default is "linear".
    rate_std : float
        Standard deviation of input rates in Hz. Default is 10.0.
    rate_mean : float
        Mean of input rates in Hz. Default is 20.0.
    """

    type: Literal["correlation"] = Field("correlation", description="Type of source population (correlation)")
    num_inputs: int = Field(100, ge=1, description="Number of input neurons")
    max_correlation: float = Field(0.4, gt=0, le=1, description="Maximum correlation")
    decay_function: Literal["linear"] = Field("linear", description="Decay function")
    rate_std: float = Field(10.0, gt=0, description="Standard deviation of input rates")
    rate_mean: float = Field(20.0, gt=0, description="Mean of input rates")


class SourcePoissonConfig(SourcePopulationConfig):
    """Configuration for Poisson source population.

    This class defines parameters for a source population that generates rates
    based on Poisson processes with specified base rates.

    Attributes
    ----------
    type : str
        The type of source population, always "poisson".
    num_inputs : int
        Number of input neurons in the source population. Default is 100.
    rates : List[float] or float
        Base firing rates for each input in Hz. Can be a single value applied to
        all inputs or a list with one value per input.
    """

    type: Literal["poisson"] = Field("poisson", description="Type of source population (poisson)")
    num_inputs: int = Field(100, ge=1, description="Number of input neurons")
    rates: List[float] | float = Field(..., description="Base firing rates for each input")


class PlasticityConfig(BaseConfig):
    """Configuration for synaptic plasticity.

    This class defines parameters for various forms of synaptic plasticity,
    including spike-timing-dependent plasticity (STDP) and homeostasis.

    Attributes
    ----------
    use_stdp : bool
        Whether to use spike-timing-dependent plasticity. Default is True.
    stdp_rate : float
        The learning rate for weight changes during STDP. Default is 0.01.
    depression_potentiation_ratio : float
        The ratio of depression to potentiation magnitudes in STDP.
        Values > 1 cause stronger depression than potentiation. Default is 1.1.
    potentiation_tau : float
        The time constant for the decay of the potentiation eligibility trace
        in seconds. Default is 0.02.
    depression_tau : float
        The time constant for the decay of the depression eligibility trace
        in seconds. Default is 0.02.
    use_homeostasis : bool
        Whether to use homeostatic plasticity to regulate firing rates. Default is True.
    homeostasis_tau : float
        The time constant of homeostatic plasticity in seconds. Default is 20.0.
    homeostasis_scale : float
        The scaling factor applied to homeostatic plasticity. Default is 1.0.
    """

    use_stdp: bool = Field(True, description="Whether to use STDP")
    stdp_rate: float = Field(0.01, gt=0, description="Rate of potentiation/depression")
    depression_potentiation_ratio: float = Field(1.1, gt=0, description="Ratio of depression to potentiation")
    potentiation_tau: float = Field(0.02, gt=0, description="Time constant of potentiation in seconds")
    depression_tau: float = Field(0.02, gt=0, description="Time constant of depression in seconds")
    use_homeostasis: bool = Field(True, description="Whether to use homeostasis")
    homeostasis_tau: float = Field(20.0, gt=0, description="Time constant of homeostasis in seconds")
    homeostasis_scale: float = Field(1.0, gt=0, description="Scale of homeostasis")


class InitializationConfig(BaseConfig):
    """Configuration for synapse weight initialization.

    This class defines parameters for initializing the weights of synapses
    within a synapse group.

    Attributes
    ----------
    min_weight : float
        Minimum fraction of max_weight used for initialization.
        Must be between 0 and 1. Default is 0.1.
    max_weight : float
        Maximum fraction of max_weight used for initialization.
        Must be between 0 and 1. Default is 1.0.
    """

    min_weight: float = Field(0.1, ge=0, le=1, description="Minimum fraction of max_weight for initialization")
    max_weight: float = Field(1.0, ge=0, le=1, description="Maximum fraction of max_weight for initialization")


class ReplacementConfig(BaseConfig):
    """Configuration for synapse replacement.

    This class defines parameters for the replacement of weak synapses
    with new ones during learning.

    Attributes
    ----------
    use_replacement : bool
        Whether to use synapse replacement. Default is True.
    lose_synapse_ratio : float
        The weight threshold (as a ratio of max_weight) below which
        a synapse is considered for replacement. Default is 0.01.
    new_synapse_ratio : float
        The initial weight (as a ratio of max_weight) for newly
        created replacement synapses. Default is 0.01.
    """

    use_replacement: bool = Field(True, description="Whether to use replacement")
    lose_synapse_ratio: float = Field(0.01, ge=0, le=1, description="Ratio of max weight that causes synapse loss")
    new_synapse_ratio: float = Field(0.01, ge=0, le=1, description="Ratio of max weight for new synapses")


class SourceConfig(BaseConfig):
    """Configuration for source populations.

    This class defines parameters for how presynaptic neurons connect to synapses,
    including the number of synapses, number of presynaptic neurons, and the rules
    for assigning connections.

    Attributes
    ----------
    num_synapses : int
        Number of synapses to create.
    num_presynaptic_neurons : int
        Number of presynaptic neurons available as inputs.
    source_rule : str
        Rule for generating presynaptic source indices. Options:
        "random": Random assignment with replacement.
        "divided": Equal division of inputs.
        "random-restricted": Random assignment from valid_sources only.
        "divided-restricted": Equal division of valid_sources.
    valid_sources : List[int], optional
        List of valid presynaptic source indices when using restricted rules.
        Default is None, in which case all sources are valid.
    """

    num_synapses: int = Field(..., ge=1, description="Number of synapses")
    num_presynaptic_neurons: int = Field(..., ge=1, description="Number of presynaptic neurons")
    source_rule: Literal["random", "divided", "random-restricted", "divided-restricted"] = Field(
        ..., description="Rule for generating presynaptic source indices"
    )
    valid_sources: Optional[List[int]] = Field(None, description="List of valid presynaptic source indices")


class BaseSynapseConfig(BaseConfig):
    """Base configuration for synapse groups.

    This abstract base class defines common parameters for all synapse group types.
    Specific synapse groups will use subclasses with additional parameters.

    Attributes
    ----------
    type : str
        The type of synapse group. This determines which subclass will be used.
    num_synapses : int
        Number of synapses in the group.
    max_weight : float
        Maximum synaptic weight in Siemens.
    reversal : float
        Reversal potential in Volts.
    tau : float
        Synaptic time constant in seconds.
    dt : float
        Time step in seconds.
    source_population : str
        Name of the source population providing inputs.
    conductance_threshold : float
        Minimum conductance (as a fraction of max_weight) required for
        a synapse to contribute current. Default is 0.1.
    plastic : bool
        Whether synapse weights are modified by plasticity rules. Default is True.
    plasticity : PlasticityConfig, optional
        Configuration for plasticity. Required if plastic is True.
    initialization : InitializationConfig, optional
        Configuration for weight initialization. Default is None.
    """

    type: str = Field(..., description="Type of synapse group")
    num_synapses: int = Field(..., ge=1, description="Number of synapses")
    max_weight: float = Field(..., gt=0, description="Maximum synaptic weight in Siemens")
    reversal: float = Field(..., description="Reversal potential in Volts")
    tau: float = Field(..., gt=0, description="Time constant in seconds")
    dt: float = Field(..., gt=0, description="Time step in seconds")
    source_population: str = Field(..., description="Name of the source population")
    conductance_threshold: float = Field(
        0.1,
        ge=0,
        le=1,
        description="Conductance threshold in relative units of max_weight",
    )
    plastic: bool = Field(True, description="Whether synapse weights are plastic")
    plasticity: Optional[PlasticityConfig] = Field(None, description="Plasticity configuration")
    initialization: Optional[InitializationConfig] = Field(None, description="Weight initialization configuration")

    @model_validator(mode="after")
    def validate_plasticity_config(cls, model: "BaseSynapseConfig"):
        """Validate that plasticity configuration is provided when needed.

        Parameters
        ----------
        model : BaseSynapseConfig
            The model instance being validated.

        Returns
        -------
        BaseSynapseConfig
            The validated model instance.

        Raises
        ------
        ValueError
            If plasticity configuration is missing when synapses are plastic.
        """
        if model.plastic and model.plasticity is None:
            raise ValueError("Plasticity configuration must be provided when synapses are plastic")
        return model


class SourcedSynapseConfig(BaseSynapseConfig):
    """Configuration for sourced synapse groups.

    This class defines parameters for synapse groups that receive input from
    source populations through configurable connection patterns.

    Attributes
    ----------
    type : str
        The type of synapse group. Always "sourced".
    source : SourceConfig, optional
        Configuration for how presynaptic neurons connect to synapses.
        Default is None.
    replacement : ReplacementConfig, optional
        Configuration for synapse replacement. Default is None.
    """

    type: Literal["sourced"] = Field("sourced", description="Type of synapse group (sourced)")
    source: Optional[SourceConfig] = Field(None, description="Source configuration")
    replacement: Optional[ReplacementConfig] = Field(None, description="Replacement configuration")


class DirectSynapseConfig(BaseSynapseConfig):
    """Configuration for direct synapse groups.

    This class defines parameters for synapse groups with a one-to-one mapping
    between input neurons and synapses. Unlike SourcedSynapseGroup, this type
    does not support synapse replacement. This is only used for routing, and
    all relevant parameters are inherited from the BaseSynapseConfig.

    Attributes
    ----------
    type : str
        The type of synapse group. Always "direct".
    """

    type: Literal["direct"] = Field("direct", description="Type of synapse group (direct)")


class NeuronConfig(BaseConfig):
    """Configuration for integrate-and-fire neuron.

    This class defines parameters for a leaky integrate-and-fire neuron model,
    including membrane properties, spike generation, and homeostasis.

    Attributes
    ----------
    time_constant : float
        Membrane time constant in seconds. Default is 20e-3 (20 ms).
    resistance : float
        Membrane resistance in Ohms. Default is 100e6 (100 MΩ).
    reset_voltage : float
        Reset voltage after spike in Volts. Default is -70e-3 (-70 mV).
    spike_threshold : float
        Voltage threshold for spike generation in Volts. Default is -50e-3 (-50 mV).
    dt : float
        Time step in seconds. Default is 0.001 (1 ms).
    use_homeostasis : bool
        Whether to use homeostatic plasticity to regulate firing rates. Default is True.
    homeostasis_tau : float
        Time constant for homeostatic plasticity in seconds. Default is 20.0.
    homeostasis_set_point : float, optional
        Target firing rate in Hz. Required if use_homeostasis is True. Default is 20.0.
    """

    time_constant: float = Field(20e-3, gt=0, description="Membrane time constant in seconds")
    resistance: float = Field(100e6, gt=0, description="Membrane resistance in Ohms")
    reset_voltage: float = Field(-70e-3, description="Reset voltage in Volts")
    spike_threshold: float = Field(-50e-3, description="Spike threshold in Volts")
    dt: float = Field(0.001, gt=0, description="Time step in seconds")
    use_homeostasis: bool = Field(True, description="Whether to use homeostasis")
    homeostasis_tau: float = Field(20.0, gt=0, description="Time constant of homeostasis in seconds")
    homeostasis_set_point: Optional[float] = Field(20.0, gt=0, description="Target firing rate in Hz")

    @field_validator("spike_threshold")
    def validate_threshold(cls, v, info):
        """Validate spike threshold is higher than reset voltage.

        Parameters
        ----------
        v : float
            The spike threshold value being validated.
        info : ValidationInfo
            Information about the validation context.

        Returns
        -------
        float
            The validated spike threshold value.

        Raises
        ------
        ValueError
            If spike_threshold is not greater than reset_voltage.
        """
        values = info.data
        if "reset_voltage" in values and v <= values["reset_voltage"]:
            raise ValueError("spike_threshold must be greater than reset_voltage")
        return v

    @field_validator("homeostasis_set_point")
    def validate_homeostasis_set_point(cls, v, info):
        """Validate homeostasis set point is provided if homeostasis is enabled.

        Parameters
        ----------
        v : float, optional
            The homeostasis set point value being validated.
        info : ValidationInfo
            Information about the validation context.

        Returns
        -------
        float, optional
            The validated homeostasis set point value.

        Raises
        ------
        ValueError
            If use_homeostasis is True but homeostasis_set_point is None.
        """
        values = info.data
        if values.get("use_homeostasis", False) and v is None:
            raise ValueError("homeostasis_set_point must be provided if use_homeostasis is True")
        return v


class SimulationConfig(BaseConfig):
    """Configuration for simulation.

    This class defines the complete configuration for a simulation, including
    the neuron, source populations, synapse groups, and global parameters.

    Attributes
    ----------
    neuron : NeuronConfig
        The configuration for the neuron.
    sources : Dict[str, Union[SourceGaborConfig, SourceICAConfig, SourceCorrelationConfig, SourcePoissonConfig]]
        A dictionary mapping names to source population configurations.
    synapses : Dict[str, Union[SourcedSynapseConfig, DirectSynapseConfig]]
        A dictionary mapping names to synapse group configurations.
    num_simulations : int
        The number of simulations to run with the given configuration. Default is 1.
    dt : float
        The time step in seconds. Default is 0.001.
    """

    neuron: NeuronConfig = Field(..., description="Neuron configuration")
    sources: Dict[str, Union[SourceGaborConfig, SourceICAConfig, SourceCorrelationConfig, SourcePoissonConfig]] = Field(
        ..., description="Source population configurations"
    )
    synapses: Dict[str, Union[SourcedSynapseConfig, DirectSynapseConfig]] = Field(
        ..., description="Synapse group configurations"
    )
    num_simulations: int = Field(1, ge=1, description="Number of simulations to run with the given configuration")
    dt: float = Field(0.001, gt=0, description="Time step in seconds")

    @model_validator(mode="after")
    def validate_dt_consistency(cls, model: "SimulationConfig"):
        """Validate dt is consistent across components.

        This validator ensures that the time step (dt) is consistent across all
        components in the simulation by propagating the simulation dt to the neuron,
        synapses, and source populations.

        Parameters
        ----------
        model : SimulationConfig
            The model instance being validated.

        Returns
        -------
        SimulationConfig
            The validated model instance with consistent dt values across components.
        """
        dt = model.dt
        if dt is not None:
            # Ensure neuron dt matches simulation dt
            model.neuron.dt = dt

            # Update dt in all synapse configs
            for synapse_config in model.synapses.values():
                synapse_config.dt = dt

            # Update dt in all source configs
            for source_config in model.sources.values():
                source_config.dt = dt

        return model
