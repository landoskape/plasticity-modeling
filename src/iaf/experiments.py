from typing import Tuple
from src.iaf.config import SimulationConfig
from src.iaf.simulation import Simulation
from src.files import config_dir


def get_experiment(
    config_name: str,
    *,
    base_dp_ratio: float = 1.1,
    distal_dp_ratio: float = 1.1,
    no_distal: bool = False,
    num_simulations: int = 1,
    edge_probability: float = 0.5,
    independent_noise_rate: float | None = None,
) -> Tuple[Simulation, SimulationConfig]:
    """Create a simulation from a named configuration with customized parameters.

    This function loads a simulation configuration from a YAML file and adjusts
    specific parameters such as depression-potentiation ratios, number of simulations,
    and edge probability. It can also optionally disable distal synapse groups.

    Parameters
    ----------
    config_name : str
        The name of the configuration file (without extension) to load from the
        config directory.
    base_dp_ratio : float, optional
        The depression-potentiation ratio for proximal & distal-simple synapses, default is 1.1.
    distal_dp_ratio : float, optional
        The depression-potentiation ratio for distal-complex synapses, default is 1.1.
    no_distal : bool, optional
        Whether to remove distal synapse groups from the simulation, default is False.
    num_simulations : int, optional
        The number of neurons to simulate, default is 1.
    edge_probability : float, optional
        The probability of generating an edge in Gabor stimuli, default is 0.5.
        Only used if the excitatory source population is a Gabor source.
    independent_noise_rate : float, optional
        The rate of independent noise in units of input rate, default is None.

    Returns
    -------
    tuple[Simulation, SimulationConfig]
        A tuple containing:
        - The configured Simulation object
        - The modified SimulationConfig object
    """
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
    if independent_noise_rate is not None:
        config.synapses["proximal"].independent_noise_rate = independent_noise_rate
        config.synapses["distal-simple"].independent_noise_rate = independent_noise_rate
        config.synapses["distal-complex"].independent_noise_rate = independent_noise_rate
    return Simulation.from_config(config), config


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
