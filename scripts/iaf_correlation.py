from time import perf_counter as tic
from src.files import get_config_path
from src.iaf.config import SimulationConfig
from src.iaf.simulation import SimulationArray, Simulation

config = SimulationConfig.from_yaml(get_config_path("correlated.yaml"))

from copy import deepcopy
from src.iaf.source_population import SourcePopulationCorrelation, SourcePopulationPoisson
from src.iaf.iaf_neuron import IaF
from src.iaf.synapse_group import SourcedSynapseGroup, DirectSynapseGroup

source_populations = {
    "excitatory": SourcePopulationCorrelation.from_config(config.sources["excitatory"]),
    "inhibitory": SourcePopulationPoisson.from_config(config.sources["inhibitory"]),
}

neuron = IaF.from_config(config.neuron)
neuron_for_array = deepcopy(neuron)

synapses = {
    "basal": SourcedSynapseGroup.from_config(config.synapses["basal"]),
    "apical": SourcedSynapseGroup.from_config(config.synapses["apical"]),
    "inhibitory": DirectSynapseGroup.from_config(config.synapses["inhibitory"]),
}

sim = Simulation(source_populations, neuron, synapses)
sims = SimulationArray(source_populations, neuron_for_array, synapses, num_simulations=3)
