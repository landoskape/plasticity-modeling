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
    return Simulation.from_config(config), config
