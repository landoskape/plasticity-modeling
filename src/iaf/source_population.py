from typing import Literal, List, Tuple, Optional, Union
from pathlib import Path
import yaml
from abc import ABC, abstractmethod
import numpy as np
from ..utils import rng
from .config import (
    SourcePopulationConfig,
    SourceGaborConfig,
    SourceICAConfig,
    SourceCorrelationConfig,
    SourcePoissonConfig,
)


def create_source_population(config: SourcePopulationConfig) -> "SourcePopulation":
    """Create a source population instance based on the configuration type.

    This factory function creates the appropriate source population object
    based on the type of configuration provided.

    Parameters
    ----------
    config : SourcePopulationConfig
        The configuration for the source population. Must be one of the following types:
        - SourceGaborConfig: Creates a SourcePopulationGabor
        - SourceICAConfig: Creates a SourcePopulationICA
        - SourceCorrelationConfig: Creates a SourcePopulationCorrelation
        - SourcePoissonConfig: Creates a SourcePopulationPoisson

    Returns
    -------
    SourcePopulation
        The created source population instance.

    Raises
    ------
    ValueError
        If the configuration type is not supported.
    """
    if isinstance(config, SourceGaborConfig):
        return SourcePopulationGabor.from_config(config)
    elif isinstance(config, SourceICAConfig):
        return SourcePopulationICA.from_config(config)
    elif isinstance(config, SourceCorrelationConfig):
        return SourcePopulationCorrelation.from_config(config)
    elif isinstance(config, SourcePoissonConfig):
        return SourcePopulationPoisson.from_config(config)
    else:
        raise ValueError(f"Invalid source population config: {config}")


class SourcePopulation(ABC):
    """Abstract base class for populations of input sources.

    This class defines the interface for all source populations, which generate
    input rates for the synaptic inputs to neurons in the simulation.

    Each source population generates rates with specific statistical properties
    that remain constant for an exponentially distributed random interval
    (determined by tau_stim).

    Attributes
    ----------
    num_inputs : int
        The number of input neurons in the population.
    dt : float
        The time step of the simulation in seconds.
    tau_stim : float
        The time constant for the stimulus in seconds, which determines
        the average duration for which input rates remain constant.
    _rates_samples_mean : int
        The mean number of time steps for which rates remain constant,
        calculated as tau_stim / dt.
    """

    num_inputs: int
    dt: float
    tau_stim: float
    _rates_samples_mean: int

    def generate_rates(self) -> Tuple[np.ndarray, int]:
        """Generate input rates and an interval for how long to keep them.

        This method generates new input rates by calling the subclass-specific
        _generate_new_rates method, and determines the duration for which these
        rates should remain constant based on an exponential distribution with
        mean tau_stim.

        Returns
        -------
        tuple[np.ndarray, int]
            A tuple containing:
            - rates: Array of shape (num_inputs,) with the firing rates for each input
            - interval: The number of time steps to keep these rates
        """
        # Generate exponential interval (minimum 1)
        sample_interval = rng.exponential(self._rates_samples_mean)
        interval = int(sample_interval) + 1 if sample_interval != np.inf else np.inf

        # Generate rates
        rates = self._generate_new_rates()

        return rates, interval

    @abstractmethod
    def _generate_new_rates(self) -> np.ndarray:
        """Generate new input rates.

        This abstract method must be implemented by subclasses to generate
        input rates according to their specific statistical properties.

        Returns
        -------
        np.ndarray
            Array of shape (num_inputs,) containing the firing rates for each input
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config: SourcePopulationConfig) -> "SourcePopulation":
        """Create a source population from a configuration object.

        Parameters
        ----------
        config : SourcePopulationConfig
            The configuration for the source population.

        Returns
        -------
        SourcePopulation
            A new source population instance.
        """
        pass

    @classmethod
    @abstractmethod
    def from_yaml(cls, fpath: Path) -> "SourcePopulation":
        """Create a source population from a YAML configuration file.

        Parameters
        ----------
        fpath : Path
            The path to the YAML configuration file.

        Returns
        -------
        SourcePopulation
            A new source population instance.
        """
        pass


class SourcePopulationGabor(SourcePopulation):
    """Source population that generates rates based on Gabor-like visual stimuli.

    This class simulates a population of neurons responding to oriented Gabor patterns,
    potentially with edge features. The stimulus is represented as a 3x3 grid of
    orientation values, and neurons respond based on their preferred orientation
    and the orientation of the stimuli.

    Attributes
    ----------
    edge_probability : float
        The probability of generating an edge in the stimulus.
    concentration : float
        The concentration parameter of the von Mises distribution, controlling
        the width of tuning curves.
    baseline_rate : float
        The baseline firing rate of neurons when not driven by a stimulus.
    driven_rate : float
        The maximum additional rate that can be added to the baseline by a stimulus.
    orientations : np.ndarray
        The set of possible orientations (in radians) for the stimulus.
    num_orientations : int
        The number of distinct orientations.
    num_inputs : int
        The number of input neurons.
    tau_stim : float
        The time constant for the stimulus in seconds.
    dt : float
        The time step of the simulation in seconds.
    """

    edge_probability: float
    concentration: float
    baseline_rate: float
    driven_rate: float
    orientations: np.ndarray
    num_orientations: int = 4
    num_inputs: int = 36
    tau_stim: float
    dt: float

    def __init__(
        self,
        edge_probability: float,
        concentration: float,
        baseline_rate: float,
        driven_rate: float,
        tau_stim: float,
        dt: float,
    ):
        """Initialize the Gabor source population.

        Parameters
        ----------
        edge_probability : float
            The probability of generating an edge in the stimulus.
        concentration : float
            The concentration parameter of the von Mises distribution.
        baseline_rate : float
            The baseline firing rate.
        driven_rate : float
            The maximum additional rate added by stimulus.
        tau_stim : float
            The time constant for the stimulus in seconds.
        dt : float
            The time step in seconds.
        """
        self.edge_probability = edge_probability
        self.orientations = np.arange(self.num_orientations) / self.num_orientations * np.pi
        self.concentration = concentration
        self.baseline_rate = baseline_rate
        self.driven_rate = driven_rate
        self.tau_stim = tau_stim
        self.dt = dt

        # Precompute the number of samples that the rates persist for (on average)
        self._rates_samples_mean = self.tau_stim / self.dt

    def generate_stimulus(self, edge_probability: Optional[float] = None) -> np.ndarray:
        """Generate a 3x3 stimulus array with orientations.

        Creates a 3x3 array where each cell contains an orientation index (0-3).
        With probability edge_probability, it will create an edge feature by
        setting the orientations of two opposite cells to match the center cell.

        Parameters
        ----------
        edge_probability : float, optional
            The probability of generating an edge, overriding the instance value.

        Returns
        -------
        np.ndarray
            A 3x3 array of orientation indices (0-3).
        """
        edge_probability = edge_probability or self.edge_probability
        stimulus_orientation = rng.integers(0, 4, size=(3, 3))
        if rng.random() < edge_probability:
            edge_orientation = stimulus_orientation[1, 1]
            x_outer = edge_orientation % 3
            y_outer = int(edge_orientation // 3)
            stimulus_orientation[x_outer, y_outer] = edge_orientation
            stimulus_orientation[-x_outer - 1, -y_outer - 1] = edge_orientation
        return stimulus_orientation

    def vonmises(self, circular_offset: np.ndarray) -> np.ndarray:
        """Calculate von Mises tuning curve values for given orientation offsets.

        Parameters
        ----------
        circular_offset : np.ndarray
            Array of orientation differences (in radians).

        Returns
        -------
        np.ndarray
            Array of same shape as circular_offset containing von Mises values.
        """
        return np.exp(self.concentration * np.cos(2 * circular_offset)) / (2 * np.pi * np.i0(self.concentration))

    def convert_stimulus_to_rates(self, stimulus: np.ndarray) -> np.ndarray:
        """Convert a stimulus array of orientation indices to firing rates.

        Parameters
        ----------
        stimulus : np.ndarray
            A 3x3 array of orientation indices (0-3).

        Returns
        -------
        np.ndarray
            An array of shape (9, 4) containing the firing rates for each
            position in the stimulus (9) and each preferred orientation (4).
        """
        stimulus = self.orientations[stimulus]
        offsets = self.orientations.reshape(1, -1) - np.reshape(stimulus, (-1, 1))
        drive = self.vonmises(offsets)
        rates = self.baseline_rate + self.driven_rate * drive
        return rates

    def _generate_new_rates(self) -> np.ndarray:
        """Generate new input rates based on a Gabor stimulus.

        This method:
        1. Generates a 3x3 stimulus array of orientations
        2. Converts the stimulus to firing rates
        3. Reshapes the rates to a 1D array

        Returns
        -------
        np.ndarray
            Array of shape (num_inputs,) containing firing rates.
        """
        stimori = self.generate_stimulus()
        rates = self.convert_stimulus_to_rates(stimori)
        return np.reshape(rates, -1)

    @classmethod
    def from_yaml(cls, fpath: Path) -> "SourcePopulationGabor":
        """Create a Gabor source population from a YAML configuration file.

        Parameters
        ----------
        fpath : Path
            The path to the YAML configuration file.

        Returns
        -------
        SourcePopulationGabor
            A new Gabor source population instance.
        """
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(SourceGaborConfig.model_validate(config))

    @classmethod
    def from_config(cls, config: SourceGaborConfig) -> "SourcePopulationGabor":
        """Create a Gabor source population from a configuration object.

        Parameters
        ----------
        config : SourceGaborConfig
            The configuration for the Gabor source population.

        Returns
        -------
        SourcePopulationGabor
            A new Gabor source population instance.
        """
        return cls(
            edge_probability=config.edge_probability,
            concentration=config.concentration,
            baseline_rate=config.baseline_rate,
            driven_rate=config.driven_rate,
            tau_stim=config.tau_stim,
            dt=config.dt,
        )


class SourceFromLoadingMixin:
    """A mixin for source populations that generate rates from a loading matrix.

    This mixin provides functionality for source populations that generate input rates
    by combining latent signals with noise using a loading matrix. The loading matrix
    determines how each latent signal contributes to the activity of each input neuron.

    Attributes
    ----------
    num_inputs : int
        The number of input neurons.
    num_signals : int
        The number of latent signals (independent components).
    source_loading : np.ndarray
        The loading matrix of shape (num_signals, num_inputs) that maps signals to inputs.
    var_adjustment : np.ndarray
        The variance adjustment factor for each input, used to normalize the variance.
    rate_mean : float
        The mean firing rate to center the distribution around.
    rate_std : float
        The standard deviation of firing rates.
    """

    num_inputs: int
    num_signals: int
    source_loading: np.ndarray
    var_adjustment: np.ndarray
    rate_mean: float
    rate_std: float
    rng: np.random.Generator

    def _generate_source_loading(self) -> None:
        """Generate the source loading matrix and compute the variance adjustment.

        This method must be implemented by subclasses to create the appropriate
        source loading matrix based on their specific requirements.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _generate_new_rates(self) -> np.ndarray:
        """Generate new input rates by combining latent signals with noise.

        This method:
        1. Generates random signal components for each latent signal
        2. Generates random noise components for each input neuron
        3. Combines signals and noise using the source loading matrix
        4. Scales and shifts the result to get the desired rate distribution
        5. Clips negative rates to zero

        Returns
        -------
        np.ndarray
            Array of shape (num_inputs,) containing the firing rates for each input neuron.
        """
        # Generate random signal components
        signal_components = rng.standard_normal(self.num_signals)

        # Generate random noise components
        noise_components = rng.standard_normal(self.num_inputs)

        # Combine signal and noise
        input_vec = (noise_components + signal_components.dot(self.source_loading)) / self.var_adjustment

        # Scale and shift to get rates
        rate = self.rate_std * input_vec + self.rate_mean

        # No negative rates
        rate = np.maximum(rate, 0)

        return rate

    @classmethod
    def estimate_correlation(cls, source_loading: np.ndarray) -> np.ndarray:
        """Estimate the correlation of each neuron with each latent signal.

        The input rates are generated by combining the latent signal with noise,
        proportional to the source loading for each input neuron. If there was only
        one signal, the total variance of the neuron is a sum of the squared loadings
        plus 1 (for the noise). Therefore, the correlation of neuron 0 with signal i
        is equal to:

        .. math::

            corr(x_0, s_i) = \\sqrt{\\frac{w_{0,i}^2}{\\sum_{j} w_{0,j}^2 + 1}}

        where :math:`w_{0,i}` is the source loading of neuron 0 for signal i.

        In practice, this isn't the true correlation because we have limited samples
        and the input rates are clipped to be nonnegative.

        Parameters
        ----------
        source_loading : np.ndarray
            The loading matrix of shape (num_signals, num_inputs).

        Returns
        -------
        np.ndarray
            The estimated correlation matrix of same shape as source_loading.
        """
        estimate = np.zeros_like(source_loading)
        for isource, loading in enumerate(source_loading):
            estimate[isource] = np.sqrt(loading**2 / (np.sum(source_loading**2, axis=0) + 1))
        return estimate


class SourcePopulationICA(SourceFromLoadingMixin, SourcePopulation):
    """
    A population of input sources with shared properties.

    This class generates input rates for a population of neurons based on different
    source loading methods. The input rates are generated as a combination of
    independent signals and noise.

    Attributes
    ----------
    num_inputs : int
        The number of input neurons.
    num_signals : int
        The number of independent components/signals.
    source_method : str
        The method for generating source loading ('divide' or 'gauss').
    source_strength : float
        The signal-to-noise ratio.
    rate_std : float
        The standard deviation of the input rates.
    rate_mean : float
        The mean of the input rates.
    source_loading : np.ndarray
        The loading matrix of shape (num_signals, num_inputs).
    var_adjustment : np.ndarray
        The variance adjustment factor for each input.
    tau_stim : float
        The time constant for the stimulus in seconds.
    dt : float
        The time step in seconds.
    """

    def __init__(
        self,
        num_inputs: int = 100,
        num_signals: int = 3,
        source_method: Literal["divide", "gauss", "correlated"] = "gauss",
        source_strength: float = 3.0,
        rate_std: float = 10.0,
        rate_mean: float = 20.0,
        gauss_source_width: float = 2 / 5,
        tau_stim: float = 0.01,
        dt: float = 0.001,
    ):
        """
        Initialize the source population.

        Parameters
        ----------
        num_inputs : int
            The number of input neurons.
        num_signals : int
            The number of independent components/signals.
        source_method : str
            The method for generating source loading ('divide' or 'gauss').
        source_strength : float
            The signal-to-noise ratio.
        rate_std : float
            The standard deviation of the input rates.
        rate_mean : float
            The mean of the input rates.
        gauss_source_width : float
            The width of the Gaussian source.
        tau_stim : float
            The time constant for the stimulus in seconds.
        dt : float
            The time step in seconds.
        """
        self.num_inputs = num_inputs
        self.num_signals = num_signals
        self.source_method = source_method
        self.source_strength = source_strength
        self.rate_std = rate_std
        self.rate_mean = rate_mean
        self.gauss_source_width = gauss_source_width
        self.tau_stim = tau_stim
        self.dt = dt

        # Create source loading based on method
        self._generate_source_loading()

        # Precompute the number of samples that the rates persist for (on average)
        self._rates_samples_mean = self.tau_stim / self.dt

    @classmethod
    def from_yaml(cls, fpath: Path) -> "SourcePopulationICA":
        """Create a source population from a YAML configuration file.

        Args:
            fpath: The path to the YAML configuration file.
        """
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(SourceICAConfig.model_validate(config))

    @classmethod
    def from_config(cls, config: SourceICAConfig) -> "SourcePopulationICA":
        """Create a source population from a configuration object.

        Args:
            config: The configuration for the source population.

        Returns:
            A new source population instance.
        """
        return cls(
            num_inputs=config.num_inputs,
            num_signals=config.num_signals,
            source_method=config.source_method,
            source_strength=config.source_strength,
            rate_std=config.rate_std,
            rate_mean=config.rate_mean,
            gauss_source_width=config.gauss_source_width,
            tau_stim=config.tau_stim,
            dt=config.dt,
        )

    def _generate_source_loading(self) -> None:
        """
        Generate the source loading by the specified method and compute the variance adjustment.
        """
        if self.source_method == "divide":
            self.source_loading = self._create_divide_loading()
        elif self.source_method == "gauss":
            self.source_loading = self._create_gauss_loading()
        else:
            raise ValueError(f"Invalid source method: {self.source_method}")

        self.var_adjustment = np.sqrt(np.sum(self.source_loading**2, axis=0) + 1)

    def _create_divide_loading(self) -> np.ndarray:
        """
        Create source loading using the 'divide' method.

        This method divides the inputs evenly among the signals.

        Returns
        -------
        np.ndarray
            The source loading matrix of shape (num_signals, num_inputs).
        """
        num_input_per_signal = self.num_inputs // self.num_signals
        source_loading = np.zeros((self.num_signals, self.num_inputs))

        for signal in range(self.num_signals):
            start_idx = signal * num_input_per_signal
            end_idx = (signal + 1) * num_input_per_signal
            source_loading[signal, start_idx:end_idx] = self.source_strength

        return source_loading

    def _create_gauss_loading(self) -> np.ndarray:
        """
        Create source loading using the 'gauss' method.

        This method creates Gaussian-shaped loadings for each signal,
        with peaks evenly spaced across the input space.

        Returns
        -------
        np.ndarray
            The source loading matrix of shape (num_signals, num_inputs).
        """
        shift_input_per_signal = self.num_inputs // self.num_signals
        width_gauss = self.gauss_source_width * shift_input_per_signal

        # Create a Gaussian centered at the middle of the input space
        idx_gauss = np.arange(self.num_inputs) - self.num_inputs // 2
        gauss_loading = np.exp(-(idx_gauss**2) / (2 * width_gauss**2))

        # Shift the Gaussian to the first signal position
        first_input_signal_idx = shift_input_per_signal // 2
        idx_peak_gauss = np.argmax(gauss_loading)
        gauss_loading = np.roll(gauss_loading, first_input_signal_idx - idx_peak_gauss)

        # Create shifted versions for each signal
        shifts = np.arange(self.num_signals) * shift_input_per_signal
        source_loading = np.vstack([np.roll(gauss_loading, shift) for shift in shifts])

        # Scale by source strength
        source_loading *= self.source_strength

        return source_loading


class SourcePopulationCorrelation(SourceFromLoadingMixin, SourcePopulation):
    """
    A population of input sources with a single latent signal with a tunable
    correlation throughout the population.

    Attributes
    ----------
    num_inputs : int
        The number of input neurons.
    max_correlation : float
        The maximum correlation for the signal.
    decay_function : Literal["linear"]
        The function to use to decay the correlation from the center to the
        edges of the population.
    rate_std : float
        The standard deviation of the input rates.
    rate_mean : float
        The mean of the input rates.
    tau_stim : float
        The time constant for the stimulus in seconds.
    dt : float
        The time step in seconds.
    """

    num_signals: int = 1

    def __init__(
        self,
        num_inputs: int,
        max_correlation: float,
        decay_function: Literal["linear"] = "linear",
        rate_std: float = 10.0,
        rate_mean: float = 20.0,
        tau_stim: float = 0.01,
        dt: float = 0.001,
    ):
        """
        Initialize the Poisson source population.

        Parameters
        ----------
        num_inputs : int
            The number of input neurons.
        max_correlation : float
            The maximum correlation for the signal.
        decay_function : Literal["linear"]
            The function to use to decay the correlation from the center to the
            edges of the population.
        rate_std : float
            The standard deviation of the input rates.
        rate_mean : float
            The mean of the input rates.
        tau_stim : float
            The time constant for the stimulus in seconds.
        dt : float
            The time step in seconds.
        """
        self.num_inputs = num_inputs
        self.max_correlation = max_correlation
        self.decay_function = decay_function
        self.rate_std = rate_std
        self.rate_mean = rate_mean
        self.tau_stim = tau_stim
        self.dt = dt

        self._generate_source_loading()

        # Precompute the number of samples that the rates persist for (on average)
        self._rates_samples_mean = self.tau_stim / self.dt

    def _generate_source_loading(self) -> None:
        """
        Generate the source loading by the specified method and compute the variance adjustment.
        """
        if self.decay_function == "linear":
            self.corr_function = np.linspace(self.max_correlation, 0, self.num_inputs)
        else:
            raise ValueError(f"Invalid decay function: {self.decay_function}")

        # Generate loadings and variance adjustment
        self.source_loading = np.reshape(np.sqrt(self.corr_function**2 / (1 - self.corr_function**2)), (1, -1))
        self.var_adjustment = np.sqrt(np.sum(self.source_loading**2, axis=0) + 1)

    @classmethod
    def from_yaml(cls, fpath: Path) -> "SourcePopulationCorrelation":
        """Create a source population from a YAML configuration file.

        Args:
            fpath: The path to the YAML configuration file.
        """
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(SourceCorrelationConfig.model_validate(config))

    @classmethod
    def from_config(cls, config: SourceCorrelationConfig) -> "SourcePopulationCorrelation":
        """Create a Poisson source population from a configuration object.

        Args:
            config: The configuration for the source population.

        Returns:
            A new source population instance.
        """
        return cls(
            num_inputs=config.num_inputs,
            max_correlation=config.max_correlation,
            decay_function=config.decay_function,
            rate_std=config.rate_std,
            rate_mean=config.rate_mean,
            tau_stim=config.tau_stim,
            dt=config.dt,
        )


class SourcePopulationPoisson(SourcePopulation):
    """
    A population of input sources with independent Poisson firing rates.

    This class represents a simpler source population where each input
    neuron has a fixed firing rate which represents the mean rate of the
    Poisson process governing each inputs spike generation.

    Attributes
    ----------
    num_inputs : int
        The number of input neurons.
    rates : np.ndarray | float | List[float]
        The firing rates for each input neuron.
    """

    def __init__(
        self,
        num_inputs: int,
        rates: float | List[float] | np.ndarray,
        tau_stim: float = 0.01,
        dt: float = 0.001,
    ):
        """
        Initialize the Poisson source population.

        Parameters
        ----------
        num_inputs : int
            The number of input neurons.
        rates : List[float], optional
            The base firing rates for each input neuron. If None, all inputs
            will have a default rate of 20 Hz.
        tau_stim : float
            The time constant for the stimulus in seconds.
        dt : float
            The time step in seconds.
        """
        self.num_inputs = num_inputs
        self.rates = self._validate_rates(rates)
        self.tau_stim = tau_stim
        self.dt = dt

        # Precompute the number of samples that the rates persist for (on average)
        self._rates_samples_mean = self.tau_stim / self.dt

    @classmethod
    def from_yaml(cls, fpath: Path) -> "SourcePopulationPoisson":
        """Create a source population from a YAML configuration file.

        Args:
            fpath: The path to the YAML configuration file.
        """
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(SourcePoissonConfig.model_validate(config))

    @classmethod
    def from_config(cls, config: SourcePoissonConfig) -> "SourcePopulationPoisson":
        """Create a Poisson source population from a configuration object.

        Args:
            config: The configuration for the source population.

        Returns:
            A new source population instance.
        """
        return cls(
            num_inputs=config.num_inputs,
            rates=config.rates,
            tau_stim=config.tau_stim,
            dt=config.dt,
        )

    def _validate_rates(self, rates: Union[float, List[float], np.ndarray]) -> np.ndarray:
        if isinstance(rates, (float, int)):
            return np.ones(self.num_inputs) * rates
        elif isinstance(rates, list):
            return np.array(rates)
        elif isinstance(rates, np.ndarray):
            return rates
        else:
            raise ValueError(f"Invalid rates: {rates}")

    def _generate_new_rates(self) -> np.ndarray:
        """
        Generate new input rates (the rates are constant and predefined).
        """
        return self.rates
