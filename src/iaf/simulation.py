from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import yaml
from .iaf_neuron import IaF
from .source_population import SourcePopulation, create_source_population
from .synapse_group import SynapseGroup, SourcedSynapseGroup, create_synapse_group
from .config import SimulationConfig


class Simulation:
    """
    A simulation of multiple neurons with synaptic inputs from a source population.

    This class handles the setup and running of a simulation with an integrate-and-fire
    neuron receiving inputs from a source population through synaptic connections.

    Attributes
    ----------
    source_populations : Dict[str, SourcePopulation]
        The source populations providing inputs to the neuron's synapse groups.
    neuron : IaF
        The integrate-and-fire neuron.
    synapses : Dict[str, SynapseGroup]
        The synapse groups providing inputs to the neuron.
    dt : float
        The time step of the simulation in seconds.
    """

    def __init__(
        self,
        source_populations: Dict[str, SourcePopulation],
        neuron: IaF,
        synapses: Dict[str, SynapseGroup],
        num_simulations: int,
        dt: float = 0.001,
    ):
        """
        Initialize the simulation.

        Parameters
        ----------
        source_populations : Dict[str, SourcePopulation]
            The source populations providing inputs to the neuron's synapse groups.
        neuron : IaF
            The neuron to simulate.
        synapses : Dict[str, SourcedSynapseGroup | DirectSynapseGroup]
            The synapse groups providing inputs to the neuron.
        num_simulations : int
            The number of simulations to run with the given configuration.
        dt : float
            The time step of the simulation in seconds (default: 0.001)
        """
        self.dt = dt

        if len(neuron.synapse_groups) != 0:
            raise ValueError("Neuron already has synapse groups")

        neurons = []
        for _ in range(num_simulations):
            new_neuron = deepcopy(neuron)
            neurons.append(new_neuron)

        self.neurons: list[IaF] = neurons
        self.source_populations: Dict[str, SourcePopulation] = {}

        for name, source_population in source_populations.items():
            self.add_source_population(source_population=source_population, name=name)

        for name, synapse_group in synapses.items():
            self.add_synapse_group(synapse_group=synapse_group, name=name)

        self._validate_dt_consistency()

    def __repr__(self) -> str:
        repr_source = ",".join(list(self.source_populations.keys()))
        repr_synapses = ",".join(list(self.neurons[0].synapse_groups.keys()))
        return f"Simulation(num_simulations={len(self.neurons)}, sources=[{repr_source}], synapses=[{repr_synapses}])"

    def _validate_dt_consistency(self) -> None:
        """Verify that dt is consistent across all components."""
        for neuron in self.neurons:
            if self.dt != neuron.dt:
                raise ValueError(f"dt ({self.dt}) does not match neuron dt ({neuron.dt})")
            for name, synapse_group in neuron.synapse_groups.items():
                if self.dt != synapse_group.dt:
                    raise ValueError(f"dt ({self.dt}) does not match synapse group dt ({name}:{synapse_group.dt})")
        for name, source_population in self.source_populations.items():
            if self.dt != source_population.dt:
                raise ValueError(f"dt ({self.dt}) does not match source population dt ({name}:{source_population.dt})")

    @classmethod
    def from_yaml(cls, fpath: Path) -> "Simulation":
        """Create a simulation from a YAML configuration file.

        Parameters
        ----------
        fpath : Path
            The path to the YAML configuration file.

        Returns
        -------
        A new simulation instance.
        """
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(SimulationConfig.model_validate(config))

    @classmethod
    def from_config(cls, config: SimulationConfig) -> "Simulation":
        """Create a simulation from a configuration object.

        Parameters
        ----------
        config : SimulationConfig
            The configuration for the simulation.

        Returns
        -------
        A new simulation instance.
        """
        # Create a new instance with base parameters
        source_populations = {
            name: create_source_population(source_config) for name, source_config in config.sources.items()
        }
        synapses = {name: create_synapse_group(synapse_config) for name, synapse_config in config.synapses.items()}
        sim = cls(
            source_populations=source_populations,
            neuron=IaF.from_config(config.neuron),
            synapses=synapses,
            num_simulations=config.num_simulations,
            dt=config.dt,
        )
        return sim

    def add_source_population(self, source_population: SourcePopulation, name: Optional[str] = None) -> None:
        """Add a source population to the simulation.

        Parameters
        ----------
        source_population : SourcePopulation
            The source population to add.
        name : Optional[str], optional
            The name of the source population. If None, a name will be generated.
        """
        if name is None:
            name = f"source_population_{len(self.source_populations)}"
        self.source_populations[name] = source_population

    def add_synapse_group(
        self, synapse_group: Union[SynapseGroup, SourcedSynapseGroup], name: Optional[str] = None
    ) -> None:
        """Add a synapse group to each neuron in the simulation.

        Uses a deepcopy of each group so that each neuron contains its own
        distinct set of synapses.

        Parameters
        ----------
        synapse_group : Union[SynapseGroup, SourcedSynapseGroup]
            The synapse group to add.
        name : Optional[str], optional
            The name of the synapse group (if None, will be generated by the neuron)

        Raises
        ------
        ValueError
            If the source population of the synapse group is not in the simulation.
        ValueError
            If the number of presynaptic neurons does not match the number of inputs in the source population.
        """
        if synapse_group.source_population not in self.source_populations:
            raise ValueError(f"Source population {synapse_group.source_population} not found")
        if hasattr(synapse_group, "source_params"):
            num_presynaptic_neurons = synapse_group.source_params.num_presynaptic_neurons
            num_inputs = self.source_populations[synapse_group.source_population].num_inputs
            if num_presynaptic_neurons != num_inputs:
                raise ValueError(
                    f"Number of presynaptic neurons {num_presynaptic_neurons} does not match source population '{synapse_group.source_population}' with {num_inputs} inputs"
                )
        for neuron in self.neurons:
            new_synapse_group = deepcopy(synapse_group)
            neuron.add_synapse_group(synapse_group=new_synapse_group, name=name)

    def _prepare_weights(self, duration: int) -> List[Dict[str, Any]]:
        """Prepare arrays to store weights over time.

        Parameters
        ----------
        duration : int
            Duration of the simulation in timesteps.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries with arrays to store weights over time.
        """
        # First figure out which synapse groups have plasticity activated
        plastic_synapse_groups = []
        for name in self.neurons[0].synapse_groups:
            if self.neurons[0].synapse_groups[name].plastic:
                plastic_synapse_groups.append(name)

        weights = [{} for _ in range(len(self.neurons))]
        for name in plastic_synapse_groups:
            for ineuron, neuron in enumerate(self.neurons):
                if hasattr(neuron.synapse_groups[name], "source_params"):
                    num_weights = neuron.synapse_groups[name].source_params.num_presynaptic_neurons
                    weights[ineuron][name] = np.zeros((duration, num_weights))
                else:
                    weights[ineuron][name] = np.zeros((duration, neuron.synapse_groups[name].num_synapses))

        return weights

    def _gather_weights(self, plastic_synapse_groups: List[str]) -> List[Dict[str, Any]]:
        """Gather the weights from the synapse groups for recording.

        Parameters
        ----------
        plastic_synapse_groups : List[str]
            List of names of plastic synapse groups.

        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries with the current weights of the synapses.
        """
        weights = [{} for _ in range(len(self.neurons))]
        for name in plastic_synapse_groups:
            for ineuron, neuron in enumerate(self.neurons):
                if hasattr(neuron.synapse_groups[name], "source_params"):
                    synapse_weights = neuron.synapse_groups[name].weights
                    idx_to_source = neuron.synapse_groups[name].source_params.presynaptic_source
                    synapse_source = neuron.synapse_groups[name].source_population
                    weights[ineuron][name] = np.zeros(self.source_populations[synapse_source].num_inputs)
                    for input_idx in range(self.source_populations[synapse_source].num_inputs):
                        weights[ineuron][name][input_idx] = np.sum(synapse_weights[idx_to_source == input_idx])
                else:
                    weights[ineuron][name] = neuron.synapse_groups[name].weights
        return weights

    def run(
        self,
        duration: int,
        progress_bar: bool = True,
        initialize: bool = True,
        save_source_rates: bool = False,
    ) -> dict:
        """Run the simulation for a specified duration.

        This method executes the simulation, updating the state of all neurons and
        their synapse groups at each timestep. The simulation proceeds as follows:

        1. Initialize the neurons and synapse groups if requested
        2. For each second of simulation:
           - For each timestep within the second:
             - Update source populations to generate new input rates
             - Step each neuron with the current input rates
             - Record spikes
             - Update synapse groups with plasticity if enabled
           - Record weights of plastic synapse groups
        3. Return the simulation results

        Parameters
        ----------
        duration : int
            The duration of the simulation in seconds.
        progress_bar : bool, optional
            Whether to show a progress bar during simulation, by default True.
        initialize : bool, optional
            Whether to initialize the neuron(s) and synapse groups before running.
            If False, will continue from the last state, by default True.
        save_source_rates : bool, optional
            Whether to return the source_rates of the simulation, by default False.
            When True, will include the source rates & source rate intervals.

        Returns
        -------
        dict
            A dictionary containing the simulation results:
            - spike_times: List of arrays containing the timesteps where each neuron spiked
            - weights: List of dictionaries containing the weight history for each neuron's
              plastic synapse groups, recorded once per second. The shape of the weight
              arrays depends on the synapse group type:
              - For sourced synapse groups: (duration, num_presynaptic_neurons)
              - For direct synapse groups: (duration, num_synapses)
            - source_rates: Dictionary of source_rates, if save_source_rates is True.
            - source_intervals: Dictionary of source_intervals, if save_source_rates is True.
        """
        steps_per_second = int(1 / self.dt)
        num_steps = int(duration * steps_per_second)

        spikes = np.zeros((len(self.neurons), num_steps), dtype=bool)
        weights = self._prepare_weights(duration=duration)

        # Initialize the neuron
        if initialize:
            for neuron in self.neurons:
                neuron.initialize(include_synapses=True, reset_weights=True)

        # Manage source populations
        sources_router = [synapse_group.source_population for synapse_group in self.neurons[0].synapse_groups.values()]
        sources_to_update = list(set(sources_router))
        source_needs_input = {source_name: True for source_name in sources_to_update}
        source_track_interval = {source_name: 0 for source_name in sources_to_update}
        source_rates = {source_name: None for source_name in sources_to_update}

        if save_source_rates:
            source_rates_full = {source_name: [] for source_name in sources_to_update}
            source_intervals_full = {source_name: [] for source_name in sources_to_update}

        # Run the simulation
        seconds_progress = tqdm(range(duration), leave=False) if progress_bar else range(duration)

        for second in seconds_progress:
            for subsecond in range(steps_per_second):
                current_step = second * steps_per_second + subsecond

                # Update source populations
                for source_name in sources_to_update:
                    if source_needs_input[source_name]:
                        rates, interval = self.source_populations[source_name].generate_rates()
                        source_rates[source_name] = rates
                        source_track_interval[source_name] = interval - 1

                        if save_source_rates:
                            source_rates_full[source_name].append(rates)
                            source_intervals_full[source_name].append(interval)

                    else:
                        source_track_interval[source_name] -= 1

                    # Figure out if this source needs new input next time
                    source_needs_input[source_name] = source_track_interval[source_name] == 0

                rates = [source_rates[source_name] for source_name in sources_router]

                # Step the neuron with the given rates
                for ineuron, neuron in enumerate(self.neurons):
                    neuron.step(rates)

                    # Record spike
                    spikes[ineuron, current_step] = neuron.spike

            current_weights = self._gather_weights(list(weights[0].keys()))
            for ineuron in range(len(self.neurons)):
                for name in weights[ineuron]:
                    weights[ineuron][name][second] = current_weights[ineuron][name]

        # Get spike times
        spike_times = [np.where(spks)[0] for spks in spikes]

        # Return results
        results = dict(
            spike_times=spike_times,
            weights=weights,
        )
        if save_source_rates:
            results["source_rates"] = {source: np.stack(rates) for source, rates in source_rates_full.items()}
            results["source_intervals"] = {
                source: np.stack(intervals) for source, intervals in source_intervals_full.items()
            }
        return results
