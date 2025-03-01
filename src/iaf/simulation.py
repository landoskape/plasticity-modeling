import numpy as np
from typing import Dict, Any, Tuple, Optional
import time
from tqdm import tqdm

from .iaf_neuron import IaF
from .source_population import SourcePopulationICA


class Simulation:
    """
    A simulation of a neuron with synaptic inputs from a source population.

    This class handles the setup and running of a simulation with an integrate-and-fire
    neuron receiving inputs from a source population through synaptic connections.

    Attributes
    ----------
    neuron : IaF
        The integrate-and-fire neuron.
    source_population : SourcePopulation
        The source population providing inputs to the neuron.
    duration : float
        The duration of the simulation in seconds.
    dt : float
        The time step of the simulation in seconds.
    """

    def __init__(
        self,
        duration: float = 20.0,  # seconds
        dt: float = 0.001,  # seconds
        seed: Optional[int] = None,
        **kwargs,
    ):
        """
        Initialize the simulation.

        Parameters
        ----------
        duration : float
            The duration of the simulation in seconds.
        dt : float
            The time step of the simulation in seconds.
        seed : int, optional
            Random seed for reproducibility.
        **kwargs
            Additional keyword arguments to pass to the neuron and source population.
        """
        self.duration = duration
        self.dt = dt
        self.steps_per_second = int(1 / dt)
        self.num_steps = int(duration * self.steps_per_second)
        self.rng = np.random.RandomState(seed)

        # Extract parameters for different components
        self._setup_parameters(kwargs)

        # Create the source population
        self.source_population = SourcePopulationICA(
            num_inputs=self.source_params["num_inputs"],
            num_signals=self.source_params["num_signals"],
            source_method=self.source_params["source_method"],
            source_strength=self.source_params["source_strength"],
            rate_std=self.source_params["rate_std"],
            rate_mean=self.source_params["rate_mean"],
            gauss_source_width=self.source_params["gauss_source_width"],
            seed=seed,
        )

        # Create the neuron
        self.neuron = IaF(
            time_constant=self.neuron_params["time_constant"],
            resistance=self.neuron_params["resistance"],
            reset_voltage=self.neuron_params["reset_voltage"],
            spike_threshold=self.neuron_params["spike_threshold"],
            dt=dt,
            use_homeostasis=self.neuron_params["use_homeostasis"],
            homeostasis_tau=self.neuron_params["homeostasis_tau"],
            homeostasis_set_point=self.neuron_params["homeostasis_set_point"],
        )

        # Add basal synapses
        self.neuron.add_synapse_group(
            name="basal",
            num_synapses=self.basal_params["num_synapses"],
            max_weight=self.basal_params["max_weight"],
            reversal=self.basal_params["reversal"],
            tau=self.basal_params["tau"],
            dt=dt,
            use_replacement=self.basal_params["use_replacement"],
            num_presynaptic_neurons=self.source_params["num_inputs"],
            lose_synapse_ratio=self.basal_params["lose_synapse_ratio"],
            new_synapse_ratio=self.basal_params["new_synapse_ratio"],
            conductance_threshold=self.basal_params["conductance_threshold"],
            use_stdp=True,
            stdp_rate=self.basal_params["stdp_rate"],
            depression_potentiation_ratio=self.basal_params["depression_potentiation_ratio"],
            potentiation_tau=self.basal_params["potentiation_tau"],
            depression_tau=self.basal_params["depression_tau"],
            use_homeostasis=self.neuron_params["use_homeostasis"],
            homeostasis_tau=self.neuron_params["homeostasis_tau"],
            homeostasis_scale=1.0,
        )

        # Add apical synapses
        self.neuron.add_synapse_group(
            name="apical",
            num_synapses=self.apical_params["num_synapses"],
            max_weight=self.apical_params["max_weight"],
            reversal=self.apical_params["reversal"],
            tau=self.apical_params["tau"],
            dt=dt,
            use_replacement=self.apical_params["use_replacement"],
            num_presynaptic_neurons=self.source_params["num_inputs"],
            lose_synapse_ratio=self.apical_params["lose_synapse_ratio"],
            new_synapse_ratio=self.apical_params["new_synapse_ratio"],
            conductance_threshold=self.apical_params["conductance_threshold"],
            use_stdp=True,
            stdp_rate=self.apical_params["stdp_rate"],
            depression_potentiation_ratio=self.apical_params["depression_potentiation_ratio"],
            potentiation_tau=self.apical_params["potentiation_tau"],
            depression_tau=self.apical_params["depression_tau"],
            use_homeostasis=self.neuron_params["use_homeostasis"],
            homeostasis_tau=self.neuron_params["homeostasis_tau"],
            homeostasis_scale=1.0,
        )

        # Add inhibitory synapses
        self.neuron.add_synapse_group(
            name="inhibitory",
            num_synapses=self.inhibitory_params["num_synapses"],
            max_weight=self.inhibitory_params["max_weight"],
            reversal=self.inhibitory_params["reversal"],
            tau=self.inhibitory_params["tau"],
            dt=dt,
            use_replacement=False,
            use_stdp=False,
            use_homeostasis=False,
        )

        # Initialize storage for results
        self.spikes = np.zeros(self.num_steps, dtype=bool)
        self.basal_weights = np.zeros((self.source_params["num_inputs"], int(self.duration)))
        self.apical_weights = np.zeros((self.source_params["num_inputs"], int(self.duration)))

    def _setup_parameters(self, kwargs: Dict[str, Any]):
        """
        Set up parameters for the simulation components.

        Parameters
        ----------
        kwargs : Dict[str, Any]
            Keyword arguments to override default parameters.
        """
        # Default parameters for the source population
        self.source_params = {
            "num_inputs": 100,
            "num_signals": 3,
            "source_method": "gauss",
            "source_strength": 3.0,
            "rate_std": 10.0,
            "rate_mean": 20.0,
            "gauss_source_width": 2 / 5,
        }

        # Default parameters for the neuron
        self.neuron_params = {
            "time_constant": 20e-3,  # seconds
            "resistance": 100e6,  # Ohms
            "reset_voltage": -70e-3,  # Volts
            "spike_threshold": -50e-3,  # Volts
            "use_homeostasis": True,
            "homeostasis_tau": 20.0,  # seconds
            "homeostasis_set_point": 20.0,  # Hz
        }

        # Default parameters for basal synapses
        self.basal_params = {
            "num_synapses": 300,
            "max_weight": 300e-12,  # Siemens
            "reversal": 0.0,  # Volts
            "tau": 20e-3,  # seconds
            "use_replacement": True,
            "lose_synapse_ratio": 0.01,
            "new_synapse_ratio": 0.01,
            "conductance_threshold": 0.1,
            "stdp_rate": 0.01,
            "depression_potentiation_ratio": 1.1,
            "potentiation_tau": 0.02,  # seconds
            "depression_tau": 0.02,  # seconds
        }

        # Default parameters for apical synapses
        self.apical_params = {
            "num_synapses": 100,
            "max_weight": 100e-12,  # Siemens
            "reversal": 0.0,  # Volts
            "tau": 20e-3,  # seconds
            "use_replacement": True,
            "lose_synapse_ratio": 0.01,
            "new_synapse_ratio": 0.01,
            "conductance_threshold": 0.1,
            "stdp_rate": 0.01,
            "depression_potentiation_ratio": 1.0,  # This can be varied
            "potentiation_tau": 0.02,  # seconds
            "depression_tau": 0.02,  # seconds
        }

        # Default parameters for inhibitory synapses
        self.inhibitory_params = {
            "num_synapses": 200,
            "max_weight": 100e-12,  # Siemens
            "reversal": -70e-3,  # Volts (same as reset)
            "tau": 20e-3,  # seconds
        }

        # Override defaults with provided kwargs
        for key, value in kwargs.items():
            if key in self.source_params:
                self.source_params[key] = value
            elif key in self.neuron_params:
                self.neuron_params[key] = value
            elif key.startswith("basal_") and key[6:] in self.basal_params:
                self.basal_params[key[6:]] = value
            elif key.startswith("apical_") and key[7:] in self.apical_params:
                self.apical_params[key[7:]] = value
            elif key.startswith("inhibitory_") and key[11:] in self.inhibitory_params:
                self.inhibitory_params[key[11:]] = value

    def run(self, progress_bar: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the simulation.

        Parameters
        ----------
        progress_bar : bool
            Whether to show a progress bar.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            The spike times, basal weights, and apical weights.
        """
        # Initialize the neuron
        self.neuron.initialize()

        # Set up stimulus parameters
        tau_stim = 0.01  # seconds
        need_input = True
        track_interval = 0

        # Run the simulation
        seconds_progress = tqdm(range(self.duration)) if progress_bar else range(self.duration)

        for second in seconds_progress:
            for subsecond in range(self.steps_per_second):
                current_step = second * self.steps_per_second + subsecond

                # Generate input if needed
                if need_input:
                    rates, interval = self.source_population.generate_rates(self.dt, tau_stim)
                    track_interval = interval - 1
                    if track_interval > 0:
                        need_input = False
                else:
                    track_interval -= 1
                    if track_interval == 0:
                        need_input = True

                # Create inhibitory input (constant rate independent of source)
                inhibitory_rate = np.ones(self.inhibitory_params["num_synapses"]) * 20.0  # Hz

                # Step the neuron with the given rates
                self.neuron.step([rates, rates, inhibitory_rate], same_input_rates=False)

                # Record spike
                self.spikes[current_step] = self.neuron.spike

            # Record weights for each input (once per second)
            for input_idx in range(self.source_params["num_inputs"]):
                # Sum weights for each input across all synapses
                basal_group = self.neuron.synapse_groups[0]
                apical_group = self.neuron.synapse_groups[1]

                # For basal synapses
                if self.basal_params["use_replacement"]:
                    basal_mask = basal_group.presynaptic_source == input_idx
                else:
                    basal_mask = input_idx
                self.basal_weights[input_idx, second] = np.sum(basal_group.weights[basal_mask])

                # For apical synapses
                if self.apical_params["use_replacement"]:
                    apical_mask = apical_group.presynaptic_source == input_idx
                else:
                    apical_mask = input_idx
                self.apical_weights[input_idx, second] = np.sum(apical_group.weights[apical_mask])

        # Get spike times
        spike_times = np.where(self.spikes)[0]

        return spike_times, self.basal_weights, self.apical_weights


def run_simulation(
    duration: float = 20.0,
    apical_depression_ratio: float = 1.0,
    num_signals: int = 3,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[Simulation, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a simulation with the specified parameters.

    Parameters
    ----------
    duration : float
        The duration of the simulation in seconds.
    apical_depression_ratio : float
        The ratio of depression to potentiation for apical synapses.
    num_signals : int
        The number of independent signals in the source population.
    seed : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional keyword arguments to pass to the simulation.

    Returns
    -------
    Tuple[Simulation, np.ndarray, np.ndarray, np.ndarray]
        The simulation, spike times, basal weights, and apical weights.
    """
    # Set up parameters
    params = {
        "num_signals": num_signals,
        "apical_depression_potentiation_ratio": apical_depression_ratio,
    }
    params.update(kwargs)

    # Create and run the simulation
    sim = Simulation(duration=duration, seed=seed, **params)
    start_time = time.time()
    spike_times, basal_weights, apical_weights = sim.run()
    end_time = time.time()

    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    print(f"Number of spikes: {len(spike_times)}")
    print(f"Average firing rate: {len(spike_times) / duration:.2f} Hz")

    results = dict(
        sim=sim,
        spike_times=spike_times,
        basal_weights=basal_weights,
        apical_weights=apical_weights,
    )
    return results
