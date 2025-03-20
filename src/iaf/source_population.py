from typing import Literal, List
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
    """
    A population of input sources with shared properties.
    """

    num_inputs: int
    dt: float
    tau_stim: float
    _rates_samples_mean: int

    def generate_rates(self) -> tuple[np.ndarray, int]:
        """
        Generate input rates and an interval for how long to keep them.

        Returns
        -------
        tuple[np.ndarray, int]
            The input rates and the number of time steps to keep them.
        """
        # Generate exponential interval (minimum 1)
        sample_interval = rng.exponential(self._rates_samples_mean)
        interval = int(sample_interval) + 1 if sample_interval != np.inf else np.inf

        # Generate rates
        rates = self._generate_new_rates()

        return rates, interval

    @abstractmethod
    def _generate_new_rates(self) -> np.ndarray:
        """
        Generate new input rates.
        """
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config):
        """Create a source population from a configuration object.

        Args:
            config: The configuration for the source population.

        Returns:
            A new source population instance.
        """
        pass

    @classmethod
    @abstractmethod
    def from_yaml(cls, fpath: Path):
        """Create a source population from a YAML configuration file.

        Args:
            fpath: The path to the YAML configuration file.
        """


class SourcePopulationGabor(SourcePopulation):
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
        self.edge_probability = edge_probability
        self.orientations = np.arange(self.num_orientations) / self.num_orientations * np.pi
        self.orientation_preference = self.orientations.reshape(1, -1)
        self.concentration = concentration
        self.baseline_rate = baseline_rate
        self.driven_rate = driven_rate
        self.tau_stim = tau_stim
        self.dt = dt

        # Precompute the number of samples that the rates persist for (on average)
        self._rates_samples_mean = self.tau_stim / self.dt

    def generate_stimulus(self, edge_probability: float = None) -> np.ndarray:
        edge_probability = edge_probability or self.edge_probability
        stimulus_orientation = rng.integers(0, 4, size=(3, 3))
        if rng.random() < edge_probability:
            edge_orientation = stimulus_orientation[1, 1]
            x_outer = edge_orientation % 3
            y_outer = int(edge_orientation // 3)
            stimulus_orientation[x_outer, y_outer] = edge_orientation
            stimulus_orientation[-x_outer - 1, -y_outer - 1] = edge_orientation
        return stimulus_orientation

    def _generate_new_rates(self) -> np.ndarray:
        stimori = self.generate_stimulus()
        stimulus = self.orientations[stimori]
        offsets = self.orientation_preference - np.reshape(stimulus, (-1, 1))
        drive = np.exp(self.concentration * np.cos(2 * offsets)) / (2 * np.pi * np.i0(self.concentration))
        rates = self.baseline_rate + self.driven_rate * drive
        return np.reshape(rates, -1)

    @classmethod
    def from_yaml(cls, fpath: Path):
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(SourceGaborConfig.model_validate(config))

    @classmethod
    def from_config(cls, config: SourceGaborConfig):
        return cls(
            edge_probability=config.edge_probability,
            concentration=config.concentration,
            baseline_rate=config.baseline_rate,
            driven_rate=config.driven_rate,
            tau_stim=config.tau_stim,
            dt=config.dt,
        )


class SourceFromLoadingMixin:
    """
    A mixin for source populations that are generated from a loading matrix.
    """

    num_inputs: int
    num_signals: int
    source_loading: np.ndarray
    var_adjustment: np.ndarray
    rate_mean: float
    rate_std: float
    rng: np.random.Generator

    def _generate_source_loading(self) -> None:
        """
        Generate the source loading and the variance adjustment.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _generate_new_rates(self) -> np.ndarray:
        """
        Generate new input rates.

        Returns
        -------
        np.ndarray
            The input rates for each input neuron in the source population.
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
    def from_yaml(cls, fpath: Path):
        """Create a source population from a YAML configuration file.

        Args:
            fpath: The path to the YAML configuration file.
        """
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(SourceICAConfig.model_validate(config))

    @classmethod
    def from_config(cls, config: SourceICAConfig):
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
    def from_yaml(cls, fpath: Path):
        """Create a source population from a YAML configuration file.

        Args:
            fpath: The path to the YAML configuration file.
        """
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(SourceCorrelationConfig.model_validate(config))

    @classmethod
    def from_config(cls, config: SourceCorrelationConfig):
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
    def from_yaml(cls, fpath: Path):
        """Create a source population from a YAML configuration file.

        Args:
            fpath: The path to the YAML configuration file.
        """
        with open(fpath, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_config(SourcePoissonConfig.model_validate(config))

    @classmethod
    def from_config(cls, config: SourcePoissonConfig):
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

    def _validate_rates(self, rates: float | List[float] | np.ndarray) -> np.ndarray:
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
