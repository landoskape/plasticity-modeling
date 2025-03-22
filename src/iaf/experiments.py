from src.iaf.config import SimulationConfig
from src.iaf.simulation import Simulation
from src.files import config_dir
from typing import Dict, Any, List, Callable, Iterator, Tuple
import itertools
import copy


def get_experiment(
    config_name: str,
    *,
    base_dp_ratio: float = 1.1,
    distal_dp_ratio: float = 1.1,
    no_distal: bool = False,
    num_simulations: int = 1,
    edge_probability: float = 0.5,
):
    fpath = config_dir() / f"{config_name}.yaml"
    config = SimulationConfig.from_yaml(fpath)
    config.synapses["proximal"].plasticity.depression_potentiation_ratio = base_dp_ratio
    if no_distal:
        config.synapses.pop("distal-simple", None)
        config.synapses.pop("distal-complex", None)
    else:
        config.synapses["distal-simple"].plasticity.depression_potentiation_ratio = base_dp_ratio
        config.synapses["distal-complex"].plasticity.depression_potentiation_ratio = distal_dp_ratio
    config.num_simulations = num_simulations
    if hasattr(config.sources["excitatory"], "edge_probability"):
        config.sources["excitatory"].edge_probability = edge_probability
    return Simulation.from_config(config), config


# def parameter_grid_search(
#     base_config_file: str,
#     parameter_ranges: Dict[str, List[Any]],
# ) -> Iterator[Tuple[Dict[str, Any], SimulationConfig]]:
#     """
#     Generate configurations for a grid search over specified parameters.

#     Args:
#         base_config_file: Name of the base YAML configuration file (e.g., "correlated.yaml")
#         parameter_ranges: Dictionary mapping parameter paths to lists of values to test
#                           Each key is a dot-separated path to the parameter in the config
#                           E.g., "synapses.apical.plasticity.depression_potentiation_ratio": [1.0, 1.1, 1.2]

#     Yields:
#         Tuple containing:
#         - Dictionary of the specific parameter values for this configuration
#         - The configured SimulationConfig object
#     """
#     # Load the base configuration
#     fpath = config_dir() / base_config_file
#     base_config = SimulationConfig.from_yaml(fpath)

#     # Create all combinations of parameter values
#     param_names = list(parameter_ranges.keys())
#     param_values = [parameter_ranges[name] for name in param_names]

#     for value_combination in itertools.product(*param_values):
#         # Create a new configuration for each combination
#         config = copy.deepcopy(base_config)
#         param_dict = {}

#         # Apply each parameter value to the configuration
#         for param_name, param_value in zip(param_names, value_combination):
#             param_dict[param_name] = param_value
#             _set_nested_attribute(config, param_name, param_value)

#         yield param_dict, config


# def run_grid_search(
#     base_config_file: str,
#     parameter_ranges: Dict[str, List[Any]],
#     experiment_fn: Callable[[SimulationConfig], Any],
#     result_handler: Callable[[Dict[str, Any], Any], None],
# ):
#     """
#     Run a grid search over specified parameters.

#     Args:
#         base_config_file: Name of the base YAML configuration file
#         parameter_ranges: Dictionary mapping parameter paths to lists of values to test
#         experiment_fn: Function that takes a SimulationConfig and runs an experiment, returning results
#         result_handler: Function that processes the results for each parameter combination
#     """
#     for params, config in parameter_grid_search(base_config_file, parameter_ranges):
#         # Run the experiment with this configuration
#         results = experiment_fn(config)

#         # Process the results
#         result_handler(params, results)


# def _set_nested_attribute(obj, path, value):
#     """
#     Set a nested attribute on an object using a dot-separated path.

#     Args:
#         obj: The object to modify
#         path: Dot-separated path to the attribute (e.g., "synapses.apical.plasticity.depression_potentiation_ratio")
#         value: The value to set
#     """
#     path_parts = path.split(".")

#     # Navigate to the parent object
#     current = obj
#     for part in path_parts[:-1]:
#         if part in current.__dict__:
#             current = current.__dict__[part]
#         elif hasattr(current, part):
#             current = getattr(current, part)
#         elif isinstance(current, dict) and part in current:
#             current = current[part]
#         else:
#             raise AttributeError(f"Cannot find attribute or key '{part}' in path '{path}'")

#     # Set the value on the final object
#     last_part = path_parts[-1]
#     if hasattr(current, last_part):
#         setattr(current, last_part, value)
#     elif isinstance(current, dict) and last_part in current:
#         current[last_part] = value
#     else:
#         raise AttributeError(f"Cannot find attribute or key '{last_part}' in path '{path}'")
