import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SpikeGenerator:
    """Manages batched spike generation for efficient random number generation."""

    num_neurons: int
    dt: float
    max_batch: int = 10

    def __post_init__(self):
        self.batch_idx = 0
        self.current_batch = None
        self.batch_size = 0

    def initialize_batch(self, steps_remaining: Optional[int] = None):
        """Initialize a new batch of random numbers."""
        if steps_remaining is not None:
            batch_size = min(self.max_batch, steps_remaining)
        else:
            batch_size = self.max_batch
        self.current_batch = np.random.rand(batch_size, self.num_neurons)
        self.batch_idx = 0
        self.batch_size = batch_size

    def get_spikes(self, rate: float, steps_remaining: Optional[int] = None) -> np.ndarray:
        """Get spikes for current timestep and advance batch index."""
        if self.batch_idx >= self.batch_size:
            self.initialize_batch(steps_remaining)

        spikes = self.current_batch[self.batch_idx] < (rate * self.dt)
        self.batch_idx += 1
        return spikes


class IafNeuron:
    def __init__(self, options: Dict[str, Any]):
        # Time parameters with defaults
        self.dt = options.get("dt", 0.001)  # in seconds
        self.T = options.get("T", 1.0)  # total time
        self.tau = options.get("tau", 20e-3)  # membrane time constant (s)

        # Basic neuron parameters with defaults
        self.resistance = options.get("resistance", 100e6)  # Ohm
        self.rest = options.get("rest", -70e-3)  # V
        self.thresh = options.get("thresh", -50e-3)  # V
        self.vm = options.get("vm", -70e-3)  # initial membrane voltage
        self.spike = False

        # Reversal potentials with defaults
        self.exRev = options.get("exRev", 0)  # V
        self.excTau = options.get("excTau", 20e-3)  # s

        # Synaptic structure
        self.numBasal = options.get("numBasal")
        self.numApical = options.get("numApical")

        # Stimulus structure
        self.numInputs = options.get("numInputs")
        self.numSignals = options.get("numSignals")
        self.sourceMethod = options.get("sourceMethod")
        self.sourceStrength = options.get("sourceStrength")
        self.sourceLoading = options.get("sourceLoading")
        self.varAdjustment = options.get("varAdjustment")
        self.rateStd = options.get("rateStd")
        self.rateMean = options.get("rateMean")

        # Basal parameters
        self.maxBasalWeight = options.get("maxBasalWeight")
        if self.maxBasalWeight is not None:
            lose_synapse_ratio = options.get("loseSynapseRatio", 0.01)
            new_synapse_ratio = options.get("newSynapseRatio", 0.01)
            conductance_threshold = options.get("conductanceThreshold", 0.1)

            self.minBasalWeight = self.maxBasalWeight * lose_synapse_ratio
            self.basalStartWeight = self.maxBasalWeight * new_synapse_ratio
            self.basalCondThresh = self.maxBasalWeight * conductance_threshold
            self.basalWeight = self.maxBasalWeight * np.random.rand(self.numBasal)
            self.basalTuneIdx = np.random.randint(0, self.numInputs, size=self.numBasal)

        self.basalConductance = 0

        # Apical parameters
        self.maxApicalWeight = options.get("maxApicalWeight")
        if self.maxApicalWeight is not None:
            lose_synapse_ratio = options.get("loseSynapseRatio", 0.01)
            new_synapse_ratio = options.get("newSynapseRatio", 0.01)
            conductance_threshold = options.get("conductanceThreshold", 0.1)

            self.minApicalWeight = self.maxApicalWeight * lose_synapse_ratio
            self.apicalStartWeight = self.maxApicalWeight * new_synapse_ratio
            self.apicalCondThresh = self.maxApicalWeight * conductance_threshold
            self.apicalWeight = self.maxApicalWeight * np.random.rand(self.numApical)
            self.apicalTuneIdx = np.random.randint(0, self.numInputs, size=self.numApical)

        self.apicalConductance = 0

        # STDP parameters
        plasticity_rate = options.get("plasticityRate", 0.01)

        if self.maxBasalWeight is not None:
            self.basalPotentiation = np.zeros_like(self.basalWeight)
            self.basalPotValue = plasticity_rate * self.maxBasalWeight
            self.basalDepValue = plasticity_rate * options.get("basalDepression", 1.1) * self.maxBasalWeight

        if self.maxApicalWeight is not None:
            self.apicalPotentiation = np.zeros_like(self.apicalWeight)
            self.apicalPotValue = plasticity_rate * self.maxApicalWeight
            self.apicalDepValue = plasticity_rate * options.get("apicalDepression", 1.0) * self.maxApicalWeight

        self.basalDepression = 0
        self.apicalDepression = 0

        # Time constants for plasticity
        self.potTau = options.get("potTau", 0.02)
        self.depTau = options.get("depTau", 0.02)

        # Homeostasis parameters
        self.homTau = options.get("homeostasisTau")
        self.homRate = options.get("homeostasisRate")
        self.homRateEstimate = self.homRate if self.homRate is not None else None
        self.homScale = 0

        # Inhibitory parameters with defaults
        self.numInhibitory = options.get("numInhibitory", 200)
        self.inhRate = options.get("inhRate", 20)
        self.inhWeight = options.get("inhWeight", 100e-12)
        self.gabaTau = options.get("gabaTau", 20e-3)
        self.gabaConductance = 0

        # Initialize optimization fields
        self._init_optimization_fields()

    def _init_optimization_fields(self):
        """Initialize optimization arrays and spike generators."""
        # Initialize spike generators if we have the required parameters
        self._spike_generators = {}

        if hasattr(self, "basalWeight"):
            self._spike_generators["basal"] = SpikeGenerator(len(self.basalWeight), self.dt)

        if hasattr(self, "apicalWeight"):
            self._spike_generators["apical"] = SpikeGenerator(len(self.apicalWeight), self.dt)

        if self.numInhibitory:
            self._spike_generators["inhibitory"] = SpikeGenerator(self.numInhibitory, self.dt)

        # Pre-compute time constants
        self._dt_tau = self.dt / self.tau
        self._dt_exc_tau = self.dt / self.excTau
        self._dt_gaba_tau = self.dt / self.gabaTau
        self._dt_pot_tau = self.dt / self.potTau
        self._dt_dep_tau = self.dt / self.depTau
        self._dt_hom_tau = self.dt / self.homTau if hasattr(self, "homTau") else None

    def get_spikes(self, location: str, input_rates: np.ndarray, steps_remaining: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if location not in self._spike_generators:
            raise ValueError(f"Invalid location: {location}")
        return self._spike_generators[location].get_spikes(input_rates, steps_remaining)
