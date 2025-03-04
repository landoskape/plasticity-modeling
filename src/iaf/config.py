from pydantic import BaseModel
from typing import Optional, Dict
import numpy as np


class SourceICAConfig(BaseModel):
    num_inputs: int
    num_signals: int
    source_method: str
    source_strength: float
    rate_std: float
    rate_mean: float
    gauss_source_width: float
    seed: Optional[int] = None


class NeuronConfig(BaseModel):
    time_constant: float
    resistance: float
    reset_voltage: float
    spike_threshold: float
    use_homeostasis: bool
    homeostasis_tau: float
    homeostasis_set_point: float


class PlasticityConfig(BaseModel):
    use_stdp: bool
    stdp_rate: float
    depression_potentiation_ratio: float
    potentiation_tau: float
    depression_tau: float
    use_homeostasis: bool
    homeostasis_tau: float
    homeostasis_set_point: float
    use_replacement: bool
    lose_synapse_ratio: float
    new_synapse_ratio: float


class SynapseConfig(BaseModel):
    num_synapses: int
    max_weight: float
    reversal: float
    tau: float
    dt: float
    conductance_threshold: float
    plasticity_config: PlasticityConfig
    num_presynaptic_neurons: int
    initial_source: Optional[np.ndarray] = None


class SimulationConfig(BaseModel):
    duration: float
    dt: float
    seed: Optional[int] = None
    source: SourceICAConfig
    neuron: NeuronConfig
    synapses: Dict[str, SynapseConfig]
