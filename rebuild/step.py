import numpy as np
from .IaF import IafNeuron
def step_iaf(iaf: IafNeuron, input_rates: np.ndarray) -> IafNeuron:
    """Step the IAF neuron forward one timestep."""
    
    # Check for spike and handle accordingly
    if iaf.vm > iaf.thresh:
        iaf.spike = True
        iaf.vm = iaf.rest
        
        # Plasticity on Basal Synapses
        iaf.basalDepression -= iaf.basalDepValue
        iaf.basalWeight += iaf.basalPotentiation
        
        # Prevent out of range values
        iaf.basalWeight = np.clip(iaf.basalWeight, 0, iaf.maxBasalWeight)
        
        # Plasticity on Apical Synapses
        iaf.apicalDepression -= iaf.apicalDepValue
        iaf.apicalWeight += iaf.apicalPotentiation
        
        # Prevent out of range values
        iaf.apicalWeight = np.clip(iaf.apicalWeight, 0, iaf.maxApicalWeight)
        
    else:
        iaf.spike = False
        
        # Update membrane voltage
        leak_term = (iaf.rest - iaf.vm) * iaf.dt/iaf.tau
        exc_df = iaf.exRev - iaf.vm
        inh_df = iaf.rest - iaf.vm
        exc_dv = ((iaf.basalConductance + iaf.apicalConductance) * 
                 exc_df * iaf.resistance * iaf.dt/iaf.tau)
        inh_dv = iaf.gabaConductance * inh_df * iaf.resistance * iaf.dt/iaf.tau
        iaf.vm += leak_term + exc_dv + inh_dv
    
    # Generate Basal Input Conductances
    basal_rate = input_rates[iaf.basalTuneIdx]
    basal_spikes = np.random.rand(len(iaf.basalWeight)) < (basal_rate * iaf.dt)
    c_basal_conductance = np.sum(
        basal_spikes * iaf.basalWeight * (iaf.basalWeight > iaf.basalCondThresh)
    )
    d_basal_conductance = (c_basal_conductance - 
                          iaf.basalConductance * iaf.dt/iaf.excTau)
    iaf.basalConductance += d_basal_conductance
    
    # Generate Apical Input Conductances
    apical_rate = input_rates[iaf.apicalTuneIdx]
    apical_spikes = np.random.rand(len(iaf.apicalWeight)) < (apical_rate * iaf.dt)
    c_apical_conductance = np.sum(
        apical_spikes * iaf.apicalWeight * (iaf.apicalWeight > iaf.apicalCondThresh)
    )
    d_apical_conductance = (c_apical_conductance - 
                           iaf.apicalConductance * iaf.dt/iaf.excTau)
    iaf.apicalConductance += d_apical_conductance
    
    # Generate Inhibitory Conductances
    inh_pre_spikes = np.random.rand(iaf.numInhibitory) < (iaf.inhRate * iaf.dt)
    inh_conductance = iaf.inhWeight * np.sum(inh_pre_spikes)
    iaf.gabaConductance += (inh_conductance - 
                           iaf.gabaConductance * iaf.dt/iaf.gabaTau)
    
    # Do depression and replacement with basal inputs
    iaf.basalWeight += basal_spikes * iaf.basalDepression
    basal_replace = iaf.basalWeight < iaf.minBasalWeight
    iaf.basalTuneIdx[basal_replace] = np.random.randint(
        0, iaf.numInputs, size=np.sum(basal_replace)
    )
    iaf.basalWeight[basal_replace] = iaf.basalStartWeight
    
    # Do depression and replacement with apical inputs
    iaf.apicalWeight += apical_spikes * iaf.apicalDepression
    apical_replace = iaf.apicalWeight < iaf.minApicalWeight
    iaf.apicalTuneIdx[apical_replace] = np.random.randint(
        0, iaf.numInputs, size=np.sum(apical_replace)
    )
    iaf.apicalWeight[apical_replace] = iaf.apicalStartWeight
    
    # Update Potentiation/Depression Terms BASAL
    iaf.basalPotentiation += (iaf.basalPotValue * basal_spikes - 
                             iaf.basalPotentiation * iaf.dt/iaf.potTau)
    iaf.basalDepression -= iaf.basalDepression * iaf.dt/iaf.depTau
    
    # Update Potentiation/Depression Terms APICAL
    iaf.apicalPotentiation += (iaf.apicalPotValue * apical_spikes - 
                              iaf.apicalPotentiation * iaf.dt/iaf.potTau)
    iaf.apicalDepression -= iaf.apicalDepression * iaf.dt/iaf.depTau
    
    # Do Homeostasis
    iaf.homRateEstimate += (iaf.dt/iaf.homTau * 
                           (1 * iaf.spike/iaf.dt - iaf.homRateEstimate))
    iaf.homScale = iaf.homRate / max(0.1, iaf.homRateEstimate) - 1
    iaf.basalWeight += (iaf.homScale * iaf.basalWeight * iaf.dt/iaf.homTau)
    iaf.apicalWeight += (iaf.homScale * iaf.apicalWeight * iaf.dt/iaf.homTau)
    
    # Prevent out of range values
    iaf.basalWeight = np.clip(iaf.basalWeight, 0, iaf.maxBasalWeight)
    iaf.apicalWeight = np.clip(iaf.apicalWeight, 0, iaf.maxApicalWeight)
    
    return iaf