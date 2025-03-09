from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, List, Literal, Union


class SourcePopulationConfig(BaseModel):
    """Base configuration for source populations."""

    type: str = Field(..., description="Type of source population")


class SourceICAConfig(SourcePopulationConfig):
    """Configuration for ICA source population."""

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
    tau_stim: float = Field(0.01, gt=0, description="Time constant for the stimulus in seconds")
    dt: float = Field(0.001, gt=0, description="Time step in seconds")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    class Config:
        use_enum_values = True


class SourcePoissonConfig(SourcePopulationConfig):
    """Configuration for Poisson source population."""

    type: Literal["poisson"] = Field("poisson", description="Type of source population (poisson)")
    num_inputs: int = Field(100, ge=1, description="Number of input neurons")
    rates: List[float] | float = Field(..., description="Base firing rates for each input")
    tau_stim: float = Field(0.01, gt=0, description="Time constant for the stimulus in seconds")
    dt: float = Field(0.001, gt=0, description="Time step in seconds")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class ReplacementConfig(BaseModel):
    """Configuration for synapse replacement."""

    use_replacement: bool = Field(True, description="Whether to use replacement")
    lose_synapse_ratio: float = Field(0.01, ge=0, le=1, description="Ratio of max weight that causes synapse loss")
    new_synapse_ratio: float = Field(0.01, ge=0, le=1, description="Ratio of max weight for new synapses")


class InitializationConfig(BaseModel):
    """Configuration for synapse weight initialization."""

    min_weight: float = Field(0.1, ge=0, le=1, description="Minimum fraction of max_weight for initialization")
    max_weight: float = Field(1.0, ge=0, le=1, description="Maximum fraction of max_weight for initialization")


class PlasticityConfig(BaseModel):
    """Configuration for synaptic plasticity."""

    use_stdp: bool = Field(True, description="Whether to use STDP")
    stdp_rate: float = Field(0.01, gt=0, description="Rate of potentiation/depression")
    depression_potentiation_ratio: float = Field(1.1, gt=0, description="Ratio of depression to potentiation")
    potentiation_tau: float = Field(0.02, gt=0, description="Time constant of potentiation in seconds")
    depression_tau: float = Field(0.02, gt=0, description="Time constant of depression in seconds")
    use_homeostasis: bool = Field(True, description="Whether to use homeostasis")
    homeostasis_tau: float = Field(20.0, gt=0, description="Time constant of homeostasis in seconds")
    homeostasis_scale: float = Field(1.0, gt=0, description="Scale of homeostasis")


class BaseSynapseConfig(BaseModel):
    """Base configuration for synapse groups."""

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
        if model.plastic and model.plasticity is None:
            raise ValueError("Plasticity configuration must be provided when synapses are plastic")
        return model


class SourcedSynapseConfig(BaseSynapseConfig):
    """Configuration for sourced synapse groups."""

    type: Literal["sourced"] = Field("sourced", description="Type of synapse group (sourced)")
    num_presynaptic_neurons: int = Field(..., ge=1, description="Number of presynaptic neurons")
    replacement: Optional[ReplacementConfig] = Field(None, description="Replacement configuration")
    presynaptic_source: Optional[List[int]] = Field(None, description="Presynaptic source indices")

    @field_validator("presynaptic_source")
    def validate_presynaptic_source(cls, v, info):
        """Validate presynaptic source indices are within range."""
        if v is not None:
            values = info.data
            if "num_presynaptic_neurons" not in values:
                raise ValueError("num_presynaptic_neurons must be set before presynaptic_source")

            if len(v) != values["num_synapses"]:
                raise ValueError(
                    f"presynaptic_source length ({len(v)}) must match num_synapses ({values['num_synapses']})"
                )

            if min(v) < 0 or max(v) >= values["num_presynaptic_neurons"]:
                raise ValueError(
                    f"presynaptic_source indices must be in range [0, {values['num_presynaptic_neurons']-1}]"
                )

        return v


class DirectSynapseConfig(BaseSynapseConfig):
    """Configuration for direct synapse groups."""

    type: Literal["direct"] = Field("direct", description="Type of synapse group (direct)")


class NeuronConfig(BaseModel):
    """Configuration for integrate-and-fire neuron."""

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
        """Validate spike threshold is higher than reset voltage."""
        values = info.data
        if "reset_voltage" in values and v <= values["reset_voltage"]:
            raise ValueError("spike_threshold must be greater than reset_voltage")
        return v

    @field_validator("homeostasis_set_point")
    def validate_homeostasis_set_point(cls, v, info):
        """Validate homeostasis set point is provided if homeostasis is enabled."""
        values = info.data
        if values.get("use_homeostasis", False) and v is None:
            raise ValueError("homeostasis_set_point must be provided if use_homeostasis is True")
        return v


class SpikeGeneratorConfig(BaseModel):
    """Configuration for spike generator."""

    num_neurons: int = Field(..., ge=1, description="Number of neurons")
    dt: float = Field(..., gt=0, description="Time step in seconds")
    max_batch: int = Field(10, ge=1, description="Maximum batch size")


class SimulationConfig(BaseModel):
    """Configuration for simulation."""

    neuron: NeuronConfig = Field(..., description="Neuron configuration")
    sources: Dict[str, Union[SourceICAConfig, SourcePoissonConfig]] = Field(
        ..., discriminator="type", description="Source population configurations"
    )
    synapses: Dict[str, Union[SourcedSynapseConfig, DirectSynapseConfig]] = Field(
        ..., discriminator="type", description="Synapse group configurations"
    )
    dt: float = Field(0.001, gt=0, description="Time step in seconds")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    @model_validator(mode="after")
    def validate_dt_consistency(cls, model: "SimulationConfig"):
        """Validate dt is consistent across components."""
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
