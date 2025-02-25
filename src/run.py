import numpy as np
from time import time
from typing import Tuple, List
from tqdm import tqdm
from .IaF import IafNeuron
from .step import step_iaf

from line_profiler import LineProfiler

profiler = LineProfiler()
step_iaf = profiler(step_iaf)


def run_iaf(
    ap_dep_idx: int,
    num_state_idx: int,
    basal_dep_follow: int,
    T: int = 19200,
    use_old: bool = False,
) -> Tuple[IafNeuron, List[int], np.ndarray, np.ndarray]:
    """Run the IAF neuron simulation."""
    # Set random seed
    np.random.seed()

    # Initialize parameters
    apic_dep_array = [1.1, 1.05, 1.025, 1.0]  # apical depression parameters
    apical_depression = apic_dep_array[ap_dep_idx]
    basal_depression = 1.1

    num_state_array = [3]  # number of independent components
    num_state = num_state_array[num_state_idx]

    # Set basal depression if following apical
    if basal_dep_follow == 1:
        basal_depression = apical_depression

    t_per_second = 1000

    options = {
        "dt": 0.001,  # time step
        "T": T,  # number of time steps
        "maxBasalWeight": 300e-12,  # can roughly think about in terms of conductance (300 pS)
        "maxApicalWeight": 100e-12,  # (100 pS)
        "loseSynapseRatio": 0.01,  # ratio to max weight for new synaptic connection
        "newSynapseRatio": 0.01,  # starting weight ratio to max of new synapse
        "basalDepression": basal_depression,  # fraction of max depression to max potentiation
        "apicalDepression": apical_depression,  # fraction of max depression to max potentiation
        "numBasal": 300,  # number of basal synapses
        "numApical": 100,  # number of apical synapses
        "plasticityRate": 0.01,  # 0.01 is about the upper limit
        "conductanceThreshold": 0.1,  # threshold for counting synaptic conductance
        # Homeostasis parameters
        "homeostasisTau": 20,  # seconds
        "homeostasisRate": 20,  # spikes/sec
        # Stimulus parameters
        "numInputs": 100,  # 100 input "types" with their own loading
        "numSignals": num_state,  # number of independent components
        "sourceMethod": "gauss",  # 'divide' or 'gauss'
        "sourceStrength": 3,  # SNR ratio
        "rateStd": 10,  # scale rates to have this standard deviation
        "rateMean": 20,  # shift rates to have this mean
    }

    # Create source loading based on method
    if options["sourceMethod"] == "divide":
        num_input_per_signal = options["numInputs"] // options["numSignals"]
        source_loading = np.zeros((options["numSignals"], options["numInputs"]))
        for signal in range(options["numSignals"]):
            start_idx = signal * num_input_per_signal
            end_idx = (signal + 1) * num_input_per_signal
            source_loading[signal, start_idx:end_idx] = options["sourceStrength"]

    elif options["sourceMethod"] == "divideSoft":
        num_input_per_signal = options["numInputs"] // options["numSignals"]
        source_loading = np.zeros((options["numSignals"], options["numInputs"]))
        for signal in range(options["numSignals"]):
            start_idx = signal * num_input_per_signal
            end_idx = (signal + 1) * num_input_per_signal
            source_loading[signal, start_idx:end_idx] = options["sourceStrength"]
        idx_soft = np.arange(options["numInputs"]) // num_input_per_signal % 2 == 1
        source_loading[:, idx_soft] /= 2

    elif options["sourceMethod"] == "gauss":
        shift_input_per_signal = options["numInputs"] // options["numSignals"]
        width_gauss = 2 / 5 * shift_input_per_signal
        idx_gauss = np.arange(options["numInputs"]) - options["numInputs"] // 2
        gauss_loading = np.exp(-(idx_gauss**2) / (2 * width_gauss**2))
        first_input_signal_idx = shift_input_per_signal // 2
        idx_peak_gauss = np.argmax(gauss_loading)
        gauss_loading = np.roll(gauss_loading, first_input_signal_idx - idx_peak_gauss)

        # Vectorized creation of source loading
        shifts = np.arange(options["numSignals"]) * shift_input_per_signal
        source_loading = np.vstack([np.roll(gauss_loading, shift) for shift in shifts])

    else:
        raise ValueError(f"Invalid source method: {options['sourceMethod']}")

    options["sourceLoading"] = source_loading
    options["varAdjustment"] = np.sqrt(np.sum(source_loading**2, axis=0) + 1)

    tau_stim = round(0.01 / options["dt"])  # time constant of stim in samples

    # Initialize storage arrays
    spikes = np.zeros(T * t_per_second, dtype=bool)
    small_basal_weight = np.zeros((options["numInputs"], T))
    small_apical_weight = np.zeros((options["numInputs"], T))

    # Build the model
    iaf = IafNeuron(options, use_old=use_old)

    # Run simulation
    need_input = True

    for second in tqdm(range(T)):
        for millisecond in range(t_per_second):
            t = second * t_per_second + millisecond

            # Generate Input
            if need_input:
                interval = int(np.random.exponential(tau_stim)) + 1
                input_vec = (
                    np.random.randn(options["numInputs"])
                    + np.random.randn(options["numSignals"]).dot(options["sourceLoading"])
                ) / options["varAdjustment"]
                rate = options["rateStd"] * input_vec + options["rateMean"]
                rate = np.maximum(rate, 0)  # No negative rates
                track_interval = interval - 1
                if track_interval > 0:
                    need_input = False
            else:
                track_interval -= 1
                if track_interval == 0:
                    need_input = True

            # Step the model
            iaf = step_iaf(iaf, rate)
            spikes[t] = iaf.spike

        # Record weights periodically
        for input_idx in range(iaf.numInputs):
            small_basal_weight[input_idx, second] = np.sum(iaf.basalWeight[iaf.basalTuneIdx == input_idx])
            small_apical_weight[input_idx, second] = np.sum(iaf.apicalWeight[iaf.apicalTuneIdx == input_idx])

    spk_times = np.where(spikes)[0]

    return iaf, spk_times, small_basal_weight, small_apical_weight, profiler
