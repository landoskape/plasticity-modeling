from dataclasses import dataclass
import numpy as np

@dataclass
class IafNeuron:
    # Time parameters
    dt: float  # in seconds
    T: float   # total time
    tau: float = 20e-3  # s
    
    # Neuron parameters
    resistance: float = 100e6  # Ohm
    rest: float = -70e-3  # V
    thresh: float = -50e-3  # V
    vm: float = -70e-3  # initial membrane voltage
    spike: bool = False
    
    # Reversal potentials
    exRev: float = 0  # V
    excTau: float = 20e-3  # s
    
    # Synaptic structure
    numBasal: int = None
    numApical: int = None
    
    # Stimulus structure
    numInputs: int = None
    numSignals: int = None
    sourceMethod: str = None
    sourceStrength: float = None
    sourceLoading: np.ndarray = None
    varAdjustment: np.ndarray = None
    rateStd: float = None
    rateMean: float = None
    
    # Basal parameters
    maxBasalWeight: float = None
    minBasalWeight: float = None
    basalStartWeight: float = None
    basalCondThresh: float = None
    basalWeight: np.ndarray = None
    basalConductance: float = 0
    basalTuneIdx: np.ndarray = None
    
    # Apical parameters
    maxApicalWeight: float = None
    minApicalWeight: float = None
    apicalStartWeight: float = None
    apicalCondThresh: float = None
    apicalWeight: np.ndarray = None
    apicalConductance: float = 0
    apicalTuneIdx: np.ndarray = None
    
    # STDP parameters
    basalPotentiation: np.ndarray = None
    basalDepression: float = 0
    basalPotValue: float = None
    basalDepValue: float = None
    
    apicalPotentiation: np.ndarray = None
    apicalDepression: float = 0
    apicalPotValue: float = None
    apicalDepValue: float = None
    
    potTau: float = 0.02
    depTau: float = 0.02
    homTau: float = None
    homRate: float = None
    homRateEstimate: float = None
    homScale: float = 0
    
    # Inhibitory parameters
    numInhibitory: int = 200
    inhRate: float = 20
    inhWeight: float = 100e-12
    gabaTau: float = 20e-3
    gabaConductance: float = 0