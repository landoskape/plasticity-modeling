from pathlib import Path
from typing import Literal
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.files import results_dir, config_dir
from src.iaf.config import SimulationConfig
from src.plotting import errorPlot
from src.iaf.iaf_neuron import IaF
from src.iaf.source_population import SourcePopulationGabor
from src.iaf.synapse_group import SourceParams
from src.utils import roll_along_axis


def get_groupnames() -> list[str]:
    return ["proximal", "distal-simple", "distal-complex"]


def gather_metadata(experiment_folder: Path, experiment_type: Literal["correlation", "hofer"]) -> dict:
    if experiment_type == "hofer":
        return _gather_metadata_hofer(experiment_folder)
    elif experiment_type == "correlation":
        return _gather_metadata_correlation(experiment_folder)
    else:
        raise ValueError(f"Invalid experiment type: {experiment_type}")


def gather_results(metadata: dict) -> list[dict]:
    results = []
    for path in metadata["data_paths"]:
        results.append(joblib.load(path))
    return results


def gather_rates(metadata: dict, experiment_type: Literal["correlation", "hofer"]) -> np.ndarray:
    if experiment_type == "hofer":
        return _gather_rates_hofer(metadata)
    elif experiment_type == "correlation":
        return _gather_rates_correlation(metadata)
    else:
        raise ValueError(f"Invalid experiment type: {experiment_type}")


def gather_weights(
    metadata: dict,
    experiment_type: Literal["correlation", "hofer"],
    average_method: Literal["fraction", "samples"] = "fraction",
    average_window: int | float = 0.2,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    if experiment_type == "hofer":
        return _gather_weights_hofer(metadata, average_method, average_window, normalize)
    elif experiment_type == "correlation":
        return _gather_weights_correlation(metadata, average_method, average_window, normalize)
    else:
        raise ValueError(f"Invalid experiment type: {experiment_type}")


def sort_orientation_preference(
    weights: dict[str, np.ndarray],
    orientation_preference: np.ndarray,
) -> dict[str, np.ndarray]:
    """Sort weights according to orientation preference.

    Parameters
    ----------
    weights : dict[str, np.ndarray]
        Weights to sort.
    """
    orientation_preference = orientation_preference[..., None]
    orientation_preference = np.repeat(orientation_preference, 9, axis=-1)
    weights_preferred = {}
    for sg in get_groupnames():
        sgweights = np.reshape(weights[sg], (*weights[sg].shape[:-1], 9, 4))
        weights_preferred[sg] = np.reshape(
            roll_along_axis(sgweights, -orientation_preference, axis=-1), (*weights[sg].shape[:-1], -1)
        )
    return weights_preferred


def summarize_weights(weights: dict[str, np.ndarray], orientation_preference: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize weights according to orientation preference.

    Groups weights into 6 categories:
    - Central-Preferred: weights in the central position for the preferred orientation
    - Central-Other: weights in the central position for other orientations
    - Edge-Preferred: weights in the edge positions for the preferred orientation
    - Edge-Other: weights in the edge positions for other orientations
    - Outer-Preferred: weights in the outer positions for the preferred orientation
    - Outer-Other: weights in the outer positions for other orientations

    Measures the average weight for each category for each experiment in the three synapse groups.

    Parameters
    ----------
    weights : dict[str, np.ndarray]
        Weights to summarize.
    orientation_preference : np.ndarray
        Orientation preference of the neurons.

    Returns
    -------
    weight_summary : dict[str, np.ndarray]
        Summary of weights.
    """
    meta_shape = orientation_preference.shape
    num_meta = np.prod(meta_shape)
    group_names = get_groupnames()
    weight_groups = [
        "central-preferred",
        "central-other",
        "edge-preferred",
        "edge-other",
        "outer-preferred",
        "outer-other",
    ]
    weight_summary = {weight_group: np.full((len(group_names), num_meta), np.nan) for weight_group in weight_groups}

    central_idx = 4
    orientation_preference = orientation_preference.reshape(-1)
    for isg, sg in enumerate(group_names):
        sgweights = weights[sg].reshape(-1, 9, 4)
        for iexperiment in range(num_meta):
            # Total weight in central position for preferred / other orientations
            central_preferred = sgweights[iexperiment, central_idx, orientation_preference[iexperiment]]
            central_other = np.sum(sgweights[iexperiment, central_idx]) - central_preferred
            # Get indices to edge positions for this orientation preference
            edge0, edge1 = SourcePopulationGabor.stimulus_to_edge_positions(
                orientation_preference[iexperiment], flattened=True
            )
            # Measure edge weights for preferred / other orientations
            preferred0 = sgweights[iexperiment, edge0, orientation_preference[iexperiment]]
            preferred1 = sgweights[iexperiment, edge1, orientation_preference[iexperiment]]
            other0 = np.sum(sgweights[iexperiment, edge0]) - preferred0
            other1 = np.sum(sgweights[iexperiment, edge1]) - preferred1
            # Total weight in all positions for preferred / other orientations
            total_preferred = np.sum(sgweights[iexperiment, :, orientation_preference[iexperiment]])
            total_other = np.sum(sgweights[iexperiment]) - total_preferred
            # Fill up summary matrix
            weight_summary["central-preferred"][isg, iexperiment] = central_preferred
            weight_summary["central-other"][isg, iexperiment] = central_other
            weight_summary["edge-preferred"][isg, iexperiment] = preferred0 + preferred1
            weight_summary["edge-other"][isg, iexperiment] = other0 + other1
            weight_summary["outer-preferred"][isg, iexperiment] = (
                total_preferred - central_preferred - preferred0 - preferred1
            )
            weight_summary["outer-other"][isg, iexperiment] = total_other - central_other - other0 - other1

    # Normalize so we summarize the average weight for each of these groups
    # We can always remultiply it to get the net weight if needed
    weight_summary["central-other"] /= 3  # 3 other orientations
    weight_summary["edge-preferred"] /= 2  # 2 positions in each edge
    weight_summary["edge-other"] /= 2 * 3  # 2 positions in each edge, 3 other orientations
    weight_summary["outer-preferred"] /= 6  # 6 outer positions not in edge
    weight_summary["outer-other"] /= 6 * 3  # 6 outer positions not in edge, 3 other orientations

    # Reshape to match the shape of the orientation preference
    # Which reflects the experiment structure
    for weight_group in weight_groups:
        weight_summary[weight_group] = np.reshape(weight_summary[weight_group], (len(group_names), *meta_shape))

    return weight_summary


def _gather_metadata_correlation(experiment_folder):
    runs = list(experiment_folder.glob("ratio_*_repeat_*.joblib"))
    args = joblib.load(experiment_folder / "args.joblib")

    ratios = []
    repeats = []
    for r in runs:
        stem = r.stem
        _, ratio, _, repeat = stem.split("_")
        ratios.append(int(ratio))
        repeats.append(int(repeat))

    base_config = SimulationConfig.from_yaml(config_dir() / (args.config + ".yaml"))

    dt = joblib.load(runs[0])["sim"].dt
    metadata = dict(
        args=args,
        config_name=args.config,
        base_config=base_config,
        dp_ratios=args.distal_dp_ratios,
        num_neurons=args.num_neurons,
        num_repeats=args.repeats,
        duration=args.duration,
        dt=dt,
        ratios=ratios,
        repeats=repeats,
        data_paths=runs,
    )
    return metadata


def _gather_metadata_hofer(experiment_folder):
    runs = list(experiment_folder.glob("ratio_*_edge_*_repeat_*.joblib"))
    args = joblib.load(experiment_folder / "args.joblib")

    ratios = []
    edges = []
    repeats = []
    for r in runs:
        stem = r.stem
        _, ratio, _, edge, _, repeat = stem.split("_")
        ratios.append(int(ratio))
        edges.append(int(edge))
        repeats.append(int(repeat))

    base_config = SimulationConfig.from_yaml(config_dir() / (args.config + ".yaml"))

    dt = joblib.load(runs[0])["sim"].dt
    metadata = dict(
        args=args,
        config_name=args.config,
        base_config=base_config,
        dp_ratios=args.distal_dp_ratios,
        edge_probabilities=args.edge_probabilities,
        num_neurons=args.num_neurons,
        num_repeats=args.repeats,
        duration=args.duration,
        dt=dt,
        ratios=ratios,
        edges=edges,
        repeats=repeats,
        data_paths=runs,
    )
    return metadata


def _gather_rates_correlation(metadata: dict) -> np.ndarray:
    num_ratios = len(metadata["dp_ratios"])
    firing_rates = np.zeros((num_ratios, metadata["num_repeats"], metadata["num_neurons"], metadata["duration"]))
    for ratio, repeat, path in zip(metadata["ratios"], metadata["repeats"], metadata["data_paths"]):
        spk_times = joblib.load(path)["spike_times"]

        for ineuron in range(metadata["num_neurons"]):
            psth = np.zeros((metadata["duration"] * int(1 / metadata["dt"])))
            psth[spk_times[ineuron]] = 1
            psth = np.sum(np.reshape(psth, (metadata["duration"], -1)), axis=1)
            firing_rates[ratio, repeat, ineuron] = psth
    return firing_rates


def _gather_rates_hofer(metadata: dict) -> np.ndarray:
    num_ratios = len(metadata["dp_ratios"])
    num_edges = len(metadata["edge_probabilities"])
    firing_rates = np.zeros(
        (num_ratios, num_edges, metadata["num_repeats"], metadata["num_neurons"], metadata["duration"])
    )
    for ratio, edge, repeat, path in zip(
        metadata["ratios"],
        metadata["edges"],
        metadata["repeats"],
        metadata["data_paths"],
    ):
        spk_times = joblib.load(path)["spike_times"]

        for ineuron in range(metadata["num_neurons"]):
            psth = np.zeros((metadata["duration"] * int(1 / metadata["dt"])))
            psth[spk_times[ineuron]] = 1
            psth = np.sum(np.reshape(psth, (metadata["duration"], -1)), axis=1)
            firing_rates[ratio, edge, repeat, ineuron] = psth
    return firing_rates


def get_norm_factor(neuron: IaF, normalize: bool = True) -> dict[str, float]:
    def get_num_inputs(source_params: SourceParams) -> int:
        if "restricted" in source_params.source_rule:
            return len(source_params.valid_sources)
        else:
            return source_params.num_presynaptic_neurons

    groups = get_groupnames()
    if normalize:
        max_weight = {sg: neuron.synapse_groups[sg].max_weight for sg in groups}
        num_synapses = {sg: neuron.synapse_groups[sg].num_synapses for sg in groups}
        num_inputs = {sg: get_num_inputs(neuron.synapse_groups[sg].source_params) for sg in groups}
        return {sg: max_weight[sg] * num_synapses[sg] / num_inputs[sg] for sg in groups}
    else:
        return {sg: 1 for sg in groups}


def _gather_weights_correlation(
    metadata: dict,
    average_method: Literal["fraction", "samples"],
    average_window: int | float,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    num_ratios = len(metadata["dp_ratios"])
    num_inputs = metadata["base_config"].sources["excitatory"].num_inputs
    weights = {
        sg: np.zeros((num_ratios, metadata["num_repeats"], metadata["num_neurons"], num_inputs))
        for sg in get_groupnames()
    }

    duration = metadata["duration"]
    if average_method == "fraction":
        if average_window >= 0.5:
            raise ValueError("Average window must be less than 0.5 for fraction averaging")
        num_timesteps = int(duration * average_window)
    else:
        if average_window >= duration / 2:
            raise ValueError("Average window must be less than half the duration for full averaging")
        num_timesteps = duration

    for ratio, repeat, path in zip(metadata["ratios"], metadata["repeats"], metadata["data_paths"]):
        results = joblib.load(path)
        neuron_weights = results["weights"]
        for ineuron in range(metadata["num_neurons"]):
            norm_factor = get_norm_factor(results["sim"].neurons[ineuron], normalize=normalize)
            for sg in get_groupnames():
                weights[sg][ratio, repeat, ineuron] = (
                    np.mean(neuron_weights[ineuron][sg][-num_timesteps:], axis=0) / norm_factor[sg]
                )

    return weights


def _gather_weights_hofer(
    metadata: dict,
    average_method: Literal["fraction", "samples"],
    average_window: int | float,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    num_ratios = len(metadata["dp_ratios"])
    num_edges = len(metadata["edge_probabilities"])
    num_inputs = SourcePopulationGabor.num_inputs
    weights = {
        sg: np.zeros((num_ratios, num_edges, metadata["num_repeats"], metadata["num_neurons"], num_inputs))
        for sg in get_groupnames()
    }

    duration = metadata["duration"]
    if average_method == "fraction":
        if average_window >= 0.5:
            raise ValueError("Average window must be less than 0.5 for fraction averaging")
        num_timesteps = int(duration * average_window)
    else:
        if average_window >= duration / 2:
            raise ValueError("Average window must be less than half the duration for full averaging")
        num_timesteps = duration

    for ratio, edge, repeat, path in zip(
        metadata["ratios"],
        metadata["edges"],
        metadata["repeats"],
        metadata["data_paths"],
    ):
        results = joblib.load(path)
        neuron_weights = results["weights"]
        for ineuron in range(metadata["num_neurons"]):
            norm_factor = get_norm_factor(results["sim"].neurons[ineuron], normalize=normalize)
            for sg in get_groupnames():
                weights[sg][ratio, edge, repeat, ineuron] = (
                    np.mean(neuron_weights[ineuron][sg][-num_timesteps:], axis=0) / norm_factor[sg]
                )

    return weights
