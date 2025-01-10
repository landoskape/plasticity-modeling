import numpy as np
from .IaF import IafNeuron

def build_iaf(options: dict) -> IafNeuron:
    """Build an integrate-and-fire neuron with the given options."""
    
    # Create basic neuron with time parameters
    iaf = IafNeuron(
        dt=options['dt'],
        T=options['T']
    )
    
    # Set stimulus structure
    iaf.numInputs = options['numInputs']
    iaf.numSignals = options['numSignals']
    iaf.sourceMethod = options['sourceMethod']
    iaf.sourceStrength = options['sourceStrength']
    iaf.sourceLoading = options['sourceLoading']
    iaf.varAdjustment = options['varAdjustment']
    iaf.rateStd = options['rateStd']
    iaf.rateMean = options['rateMean']
    
    # Set synaptic structure
    iaf.numBasal = options['numBasal']
    iaf.numApical = options['numApical']
    
    # Basal weights
    iaf.maxBasalWeight = options['maxBasalWeight']
    iaf.minBasalWeight = iaf.maxBasalWeight * options['loseSynapseRatio']
    iaf.basalStartWeight = iaf.maxBasalWeight * options['newSynapseRatio']
    iaf.basalCondThresh = iaf.maxBasalWeight * options['conductanceThreshold']
    iaf.basalWeight = iaf.maxBasalWeight * np.random.rand(iaf.numBasal)
    iaf.basalTuneIdx = np.random.randint(0, iaf.numInputs, size=iaf.numBasal)
    
    # Apical weights
    iaf.maxApicalWeight = options['maxApicalWeight']
    iaf.minApicalWeight = iaf.maxApicalWeight * options['loseSynapseRatio']
    iaf.apicalStartWeight = iaf.maxApicalWeight * options['newSynapseRatio']
    iaf.apicalCondThresh = iaf.maxApicalWeight * options['conductanceThreshold']
    iaf.apicalWeight = iaf.maxApicalWeight * np.random.rand(iaf.numApical)
    iaf.apicalTuneIdx = np.random.randint(0, iaf.numInputs, size=iaf.numApical)
    
    # STDP parameters
    iaf.basalPotentiation = np.zeros_like(iaf.basalWeight)
    iaf.basalPotValue = options['plasticityRate'] * iaf.maxBasalWeight
    iaf.basalDepValue = (options['plasticityRate'] * 
                        options['basalDepression'] * 
                        iaf.maxBasalWeight)
    
    iaf.apicalPotentiation = np.zeros_like(iaf.apicalWeight)
    iaf.apicalPotValue = options['plasticityRate'] * iaf.maxApicalWeight
    iaf.apicalDepValue = (options['plasticityRate'] * 
                         options['apicalDepression'] * 
                         iaf.maxApicalWeight)
    
    # Homeostasis parameters
    iaf.homTau = options['homeostasisTau']
    iaf.homRate = options['homeostasisRate']
    iaf.homRateEstimate = iaf.homRate  # start at homRate to avoid blowups
    
    return iaf