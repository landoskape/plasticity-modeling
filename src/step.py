import numpy as np
from numba import jit
from .IaF import IafNeuron


@jit(nopython=True)
def compute_conductance(spikes, weights, threshold):
    """Compute conductance from spikes and weights with threshold."""
    return np.sum(spikes * weights * (weights > threshold))


@jit(nopython=True)
def clip_weights(weights, min_val, max_val):
    """Clip weights to range [min_val, max_val] in-place."""
    np.clip(weights, min_val, max_val, out=weights)
    return weights


@jit(nopython=True)
def update_membrane_voltage(vm, rest, dt, tau, exc_conductance, inh_conductance, ex_rev, resistance):
    """Update membrane voltage using pre-computed terms."""
    leak_term = (rest - vm) * dt / tau
    exc_df = ex_rev - vm
    inh_df = rest - vm
    exc_dv = exc_conductance * exc_df * resistance * dt / tau
    inh_dv = inh_conductance * inh_df * resistance * dt / tau
    return vm + leak_term + exc_dv + inh_dv


def step_iaf(iaf: IafNeuron, input_rates: np.ndarray) -> IafNeuron:
    """Optimized version of step_iaf that uses pre-allocated arrays and JIT compilation."""

    # Handle spike state
    if iaf.vm > iaf.thresh:
        iaf.spike = True
        iaf.vm = iaf.rest

        # Combined plasticity updates for efficiency
        iaf.basalDepression -= iaf.basalDepValue
        iaf.apicalDepression -= iaf.apicalDepValue

        iaf.basalWeight += iaf.basalPotentiation
        iaf.apicalWeight += iaf.apicalPotentiation

        # Use in-place clipping
        clip_weights(iaf.basalWeight, 0, iaf.maxBasalWeight)
        clip_weights(iaf.apicalWeight, 0, iaf.maxApicalWeight)

    else:
        iaf.spike = False
        # Update membrane voltage using JIT-compiled function
        iaf.vm = update_membrane_voltage(
            iaf.vm,
            iaf.rest,
            iaf.dt,
            iaf.tau,
            iaf.basalConductance + iaf.apicalConductance,
            iaf.gabaConductance,
            iaf.exRev,
            iaf.resistance,
        )

    # Generate all spikes at once using pre-allocated arrays
    basal_rates = input_rates[iaf.basalTuneIdx]
    apical_rates = input_rates[iaf.apicalTuneIdx]

    _basal_spikes = iaf.get_spikes("basal", basal_rates)
    _apical_spikes = iaf.get_spikes("apical", apical_rates)
    _inh_spikes = iaf.get_spikes("inhibitory", iaf.inhRate)

    # Compute conductances using JIT-compiled function
    c_basal = compute_conductance(_basal_spikes, iaf.basalWeight, iaf.basalCondThresh)
    c_apical = compute_conductance(_apical_spikes, iaf.apicalWeight, iaf.apicalCondThresh)

    # Update conductances with pre-computed time constants
    iaf.basalConductance += c_basal - iaf.basalConductance * iaf._dt_exc_tau
    iaf.apicalConductance += c_apical - iaf.apicalConductance * iaf._dt_exc_tau
    iaf.gabaConductance += iaf.inhWeight * np.sum(_inh_spikes) - iaf.gabaConductance * iaf._dt_gaba_tau

    # Batch process depression and replacement
    iaf.basalWeight += _basal_spikes * iaf.basalDepression
    iaf.apicalWeight += _apical_spikes * iaf.apicalDepression

    # Handle replacements efficiently
    basal_replace = iaf.basalWeight < iaf.minBasalWeight
    apical_replace = iaf.apicalWeight < iaf.minApicalWeight

    n_basal_replace = np.sum(basal_replace)
    n_apical_replace = np.sum(apical_replace)

    if n_basal_replace > 0:
        iaf.basalTuneIdx[basal_replace] = np.random.randint(0, iaf.numInputs, size=n_basal_replace)
        iaf.basalWeight[basal_replace] = iaf.basalStartWeight

    if n_apical_replace > 0:
        iaf.apicalTuneIdx[apical_replace] = np.random.randint(0, iaf.numInputs, size=n_apical_replace)
        iaf.apicalWeight[apical_replace] = iaf.apicalStartWeight

    # Update potentiation/depression terms with pre-computed time constants
    iaf.basalPotentiation += iaf.basalPotValue * _basal_spikes - iaf.basalPotentiation * iaf._dt_pot_tau
    iaf.apicalPotentiation += iaf.apicalPotValue * _apical_spikes - iaf.apicalPotentiation * iaf._dt_pot_tau

    iaf.basalDepression -= iaf.basalDepression * iaf._dt_dep_tau
    iaf.apicalDepression -= iaf.apicalDepression * iaf._dt_dep_tau

    # Homeostasis with pre-computed time constant
    spike_term = 1 * iaf.spike / iaf.dt
    iaf.homRateEstimate += iaf._dt_hom_tau * (spike_term - iaf.homRateEstimate)
    iaf.homScale = iaf.homRate / max(0.1, iaf.homRateEstimate) - 1

    # Combined weight updates
    hom_factor = iaf.homScale * iaf._dt_hom_tau
    iaf.basalWeight += hom_factor * iaf.basalWeight
    iaf.apicalWeight += hom_factor * iaf.apicalWeight

    # Final weight clipping
    clip_weights(iaf.basalWeight, 0, iaf.maxBasalWeight)
    clip_weights(iaf.apicalWeight, 0, iaf.maxApicalWeight)

    return iaf
