from typing import Optional
import numpy as np
import pandas as pd
from scipy.constants import R, physical_constants
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType

from src.files import get_figure_dir
from src.plotting import save_figure, FigParams

F = physical_constants["Faraday constant"][0]


class VGCC:
    """Voltage-Gated Calcium Channel (VGCC) model with activation and inactivation gates.

    Rate constants for activation (m) and inactivation (h) gates:
    alpha_m(V) = 0.055(-27-V)/(e^((-27-V)/3.8)-1) [ms^-1 mV^-1]
    beta_m(V) = 0.94e^((-75-V)/17) [ms^-1]
    alpha_h(V) = 0.000457e^((-13-V)/50) [ms^-1]
    beta_h(V) = 0.0065/(e^((-V-15)/28)+1) [ms^-1]

    Time constant and open probability:
    tau(V) = 1/(alpha_V + beta_V)
    P_open(V) = m^2 * h
    """

    @classmethod
    def color(self) -> list[float]:
        """Return the color representation of the VGCC.

        Returns
        -------
        list[float]
            RGB color values normalized to [0, 1] range
        """
        return [x / 255 for x in [66, 135, 135]]

    def __init__(self) -> None:
        """Initialize VGCC with parameters."""
        pass

    def _alpha_m(self, V: float) -> float:
        """Forward rate constant for activation gate.

        Parameters
        ----------
        V : float
            Membrane potential in mV

        Returns
        -------
        float
            Forward rate constant in ms^-1
        """
        return 0.055 * (-27 - V) / (np.exp((-27 - V) / 3.8) - 1)

    def _beta_m(self, V: float) -> float:
        """Backward rate constant for activation gate.

        Parameters
        ----------
        V : float
            Membrane potential in mV

        Returns
        -------
        float
            Backward rate constant in ms^-1
        """
        return 0.94 * np.exp((-75 - V) / 17)

    def _alpha_h(self, V: float) -> float:
        """Forward rate constant for inactivation gate.

        Parameters
        ----------
        V : float
            Membrane potential in mV

        Returns
        -------
        float
            Forward rate constant in ms^-1
        """
        return 0.000457 * np.exp((-13 - V) / 50)

    def _beta_h(self, V: float) -> float:
        """Backward rate constant for inactivation gate.

        Parameters
        ----------
        V : float
            Membrane potential in mV

        Returns
        -------
        float
            Backward rate constant in ms^-1
        """
        return 0.0065 / (np.exp((-V - 15) / 28) + 1)

    def time_constant(self, V: float) -> tuple[float, float]:
        """Compute the time constant of the VGCC activation & inactivation gates.

        Parameters
        ----------
        V : float
            Membrane potential in mV

        Returns
        -------
        tuple[float, float]
            Time constants for activation and inactivation gates in ms
        """
        alpha_V_m = self._alpha_m(V)
        beta_V_m = self._beta_m(V)
        alpha_V_h = self._alpha_h(V)
        beta_V_h = self._beta_h(V)
        time_constant_m = 1 / (alpha_V_m + beta_V_m)
        time_constant_h = 1 / (alpha_V_h + beta_V_h)
        return time_constant_m, time_constant_h

    def open_probability_activation(self, V: float) -> float:
        """Compute the open probability of the VGCC activation gate.

        Parameters
        ----------
        V : float
            Membrane potential in mV

        Returns
        -------
        float
            Open probability of activation gate
        """
        alpha_V_m = self._alpha_m(V)
        beta_V_m = self._beta_m(V)
        return alpha_V_m / (alpha_V_m + beta_V_m)

    def open_probability_inactivation(self, V: float) -> float:
        """Compute the open probability of the VGCC inactivation gate.

        Parameters
        ----------
        V : float
            Membrane potential in mV

        Returns
        -------
        float
            Open probability of inactivation gate
        """
        alpha_V_h = self._alpha_h(V)
        beta_V_h = self._beta_h(V)
        return alpha_V_h / (alpha_V_h + beta_V_h)

    def open_probability(self, V: float) -> float:
        """Compute the open probability of the VGCC.

        Parameters
        ----------
        V : float
            Membrane potential in mV

        Returns
        -------
        float
            Open probability
        """
        m = self.open_probability_activation(V)
        h = self.open_probability_inactivation(V)
        return m**2 * h

    def dmdt(self, V: float, m: float) -> float:
        """Compute the time derivative of the activation gate.

        Parameters
        ----------
        V : float
            Membrane potential in mV
        m : float
            Current activation gate value

        Returns
        -------
        float
            Time derivative of activation gate
        """
        alpha_V_m = self._alpha_m(V)
        beta_V_m = self._beta_m(V)
        return alpha_V_m * (1 - m) - beta_V_m * m

    def dhdt(self, V: float, h: float) -> float:
        """Compute the time derivative of the inactivation gate.

        Parameters
        ----------
        V : float
            Membrane potential in mV
        h : float
            Current inactivation gate value

        Returns
        -------
        float
            Time derivative of inactivation gate
        """
        alpha_V_h = self._alpha_h(V)
        beta_V_h = self._beta_h(V)
        return alpha_V_h * (1 - h) - beta_V_h * h


class NMDAR:
    """N-Methyl-D-Aspartate Receptor (NMDAR) model with voltage-dependent Mg2+ block.

    Magnesium block kinetics:
    k_off = e^(0.017V + 0.96) [ms^-1]
    k_on = [Mg^2+]e^(-0.045V - 6.97) [ms^-1 µM^-1]

    Time constant and open probability:
    tau(V) = 1/(k_on + k_off)
    P_open(V) = k_off/(k_on + k_off)
    """

    @classmethod
    def color(self) -> str:
        """Return the color representation of the NMDAR.

        Returns
        -------
        str
            Color name
        """
        return "black"

    def __init__(self, mg_conc: float = 1000) -> None:
        """Initialize NMDAR with parameters.

        Parameters
        ----------
        mg_conc : float, optional
            Magnesium concentration in µM, by default 1000
        """
        self.mg_conc = mg_conc

    def _k_off(self, V: float) -> float:
        """Off rate for Mg2+ block.

        Parameters
        ----------
        V : float
            Membrane potential in mV

        Returns
        -------
        float
            Off rate in ms^-1
        """
        return np.exp(0.017 * V + 0.96)

    def _k_on(self, V: float) -> float:
        """On rate for Mg2+ block.

        Parameters
        ----------
        V : float
            Membrane potential in mV

        Returns
        -------
        float
            On rate in ms^-1 µM^-1
        """
        return self.mg_conc * np.exp(-0.045 * V - 6.97)

    def time_constant(self, V: float) -> float:
        """Compute the time constant of the NMDAR.

        Parameters
        ----------
        V : float
            Membrane potential in mV

        Returns
        -------
        float
            Time constant in ms
        """
        return 1 / (self._k_on(V) + self._k_off(V))

    def open_probability(self, V: float) -> float:
        """Compute the open probability of the NMDAR.

        Parameters
        ----------
        V : float
            Membrane potential in mV

        Returns
        -------
        float
            Open probability
        """
        return self._k_off(V) / (self._k_on(V) + self._k_off(V))

    def dndt(self, V: float, n: float) -> float:
        """Compute the time derivative of the NMDAR activation gate.

        Parameters
        ----------
        V : float
            Membrane potential in mV
        n : float
            Current activation gate value

        Returns
        -------
        float
            Time derivative of activation gate
        """
        k_off = self._k_off(V)
        k_on = self._k_on(V)
        return k_off * (1 - n) - k_on * n


def compute_current(V: float, p_open: float, ca_in: float, ca_out: float, temp: float = 310.15) -> float:
    """Compute relative calcium current using modified Goldman-Hodgkin-Katz equation.

    I_Ca = P_open * V * ([Ca]_in - [Ca]_out * e^(-2VF/RT))/(1 - e^(-2VF/RT))

    This function does not include the maximum conductance of the channel, which is
    technically required to compute the current. We can think of this as the current
    per unit of conductance -- similar to the current density (but not quite). It's
    posed like this because the simulations that use this model are concerned with
    what conditions open the channel, rather than the specific conductance.

    Parameters
    ----------
    V : float
        Membrane potential in mV
    p_open : float
        Open probability of the channel
    ca_in : float
        Intracellular calcium concentration
    ca_out : float
        Extracellular calcium concentration
    temp : float, optional
        Temperature in Kelvin, by default 310.15

    Returns
    -------
    float
        Relative calcium current
    """
    V_volts = V / 1000  # Convert mV to V
    numer = ca_in - ca_out * np.exp(-2 * V_volts * F / (R * temp))
    denom = 1 - np.exp(-2 * V_volts * F / (R * temp))
    return p_open * V_volts * (numer / denom)


class AP:
    """Action potential waveform model using quadratic function.

    Voltage equation:
    V(t) = V_base + V_amp - (at)^2  for -V_dur/2 < t < V_dur/2
    V(t) = V_base                    otherwise

    where 'a' is calculated as:
    a = sqrt(4*V_amp/V_dur^2)
    """

    def __init__(self, v_amp: float, v_dur: float, v_base: float = -70) -> None:
        """Initialize action potential waveform.

        Parameters
        ----------
        v_amp : float
            AP amplitude in mV
        v_dur : float
            AP duration in ms
        v_base : float, optional
            Baseline voltage in mV, by default -70
        """
        self.v_amp = v_amp
        self.v_dur = v_dur
        self.v_base = v_base
        self.a = np.sqrt(v_amp) * 2 / v_dur

    def voltage(self, t: np.ndarray, t_peak: float) -> np.ndarray:
        """Compute the voltage at times t.

        Parameters
        ----------
        t : np.ndarray
            Time in ms
        t_peak : float
            Time of peak voltage in ms

        Returns
        -------
        np.ndarray
            Voltage in mV
        """
        voltage = self.v_base * np.ones_like(t)
        t_ap_idx = np.where(abs(t - t_peak) <= self.v_dur / 2)[0]
        voltage[t_ap_idx] += self.v_amp - (self.a * (t[t_ap_idx] - t_peak)) ** 2
        return voltage


def get_csv_data(file_path: str) -> pd.DataFrame:
    """Read CSV data from Nevian-sakmann-2006 data.

    CSV Files are no header two-column csvs with x data and y data in each column.
    They are generated by using the WebPlotDigitizer tool on Figure 4, Panels B & D
    from this paper: https://www.jneurosci.org/content/26/43/11001.long

    Parameters
    ----------
    file_path : str
        Path to the CSV file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'x' and 'y'
    """
    df = pd.read_csv(file_path, header=None)
    df.columns = ["x", "y"]
    return df


def sigmoid(x: np.ndarray, L: float, x0: float, k: float, b: float) -> np.ndarray:
    """Sigmoid function of the form L/(1 + e^(-k(x-x0))) + b.

    Parameters
    ----------
    x : np.ndarray
        Input values
    L : float
        Maximum value of the curve
    x0 : float
        x-value of the sigmoid's midpoint
    k : float
        Steepness of the curve
    b : float
        y-offset of the curve

    Returns
    -------
    np.ndarray
        Sigmoid function values
    """
    return L / (1 + np.exp(-k * (x - x0))) + b


def fit_sigmoid(df: pd.DataFrame) -> np.ndarray:
    """Fit a sigmoid function to the data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'x' and 'y' columns containing the data to fit

    Returns
    -------
    np.ndarray
        Array of fitted parameters [L, x0, k, b]
    """
    # Extract x and y data
    x_data = df["x"].values
    y_data = df["y"].values

    # Figure out if it's increasing or decreasing
    idx_increasing = np.argsort(x_data)
    y_sorted = y_data[idx_increasing]
    increasing = y_sorted[-1] > y_sorted[0]

    # Initial parameter guesses
    L0 = (max(y_data) - min(y_data)) * (1 if increasing else -1)
    x00 = np.median(x_data)
    k0 = 1.0  # why not
    b0 = y_sorted[0]

    # Format for curve_fit
    p0 = [L0, x00, k0, b0]

    # Define bounds -- it's all unbounde except for K which needs to be positive
    lower_bounds = [-np.inf, -np.inf, 0, -np.inf]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf]

    # Fit the sigmoid function to the data
    params = curve_fit(
        sigmoid,
        x_data,
        y_data,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
        maxfev=10000,
    )[0]

    return params


def plasticity_transfer_function(
    params: tuple[float],
    x_values: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a transfer function for plasticity based on sigmoid fit of buffer effects.

    Our model is that the reduction in plasticity caused by adding more buffer linearly
    corresponds to the relationship between the amount of calcium influx and the amount
    of plasticity. So, if a buffer concentration of 0.5mM causes a 50% reduction in the
    amount of LTP/LTD, then we assume that the effective dose of [Ca] at 0.5mM buffer
    corresponds to 50% of the maximum possible LTP/LTD.

    Since we assume linearity, we can use the hill coefficient (k in the sigmoid function)
    and the half-effective concentration (x0 in the sigmoid function) to produce a transfer
    function between calcium concentration and plasticity magnitude where we assume that
    the curve goes from 0% to 100%. Then, the only free parameter is the scaling between
    effective dose of calcium and buffer concentration.

    Parameters
    ----------
    params : tuple[float]
        Parameters from sigmoid fit (L, x0, k, b) where:
        L: Maximum value (will be set to 1.0)
        x0: x-value of the sigmoid's midpoint (half-effective concentration)
        k: Steepness of the curve (hill coefficient)
        b: y-offset (will be set to 0)
    x_values : np.ndarray, optional
        Values to use for the transfer function, by default None
        If None, will use np.linspace(0, 2, 101)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - transfer_function: Normalized sigmoid values (0-1 range)
        - x: Corresponding x values (effective calcium dose) from 0 to 2
    """
    # Set up effective calcium dose values
    if x_values is None:
        x_values = np.linspace(0, 2, 101)

    transfer_function = sigmoid(x_values, *params)

    # Clamp to 100% -> 0%
    initial = sigmoid(0, *params)
    baseline = sigmoid(50, *params)
    scale = initial - baseline
    transfer_function = (transfer_function - baseline) / scale
    transfer_function = 1 - transfer_function
    return transfer_function, x_values


def plot_channel_properties(
    v_min: float = -80,
    v_max: float = 40,
    num_points: int = 200,
    show_fig: bool = True,
    save_fig: bool = False,
) -> plt.Figure:
    """Plot the channel properties of the NMDAR and VGCC.

    Parameters
    ----------
    v_min : float, optional
        Minimum voltage in mV, by default -80
    v_max : float, optional
        Maximum voltage in mV, by default 40
    num_points : int, optional
        Number of points to use for the plot, by default 200
    show_fig : bool, optional
        Whether to show the figure, by default True
    save_fig : bool, optional
        Whether to save the figure, by default False

    Returns
    -------
    plt.Figure
        The figure containing the channel properties plot
    """
    fig, (ax_open_prob, ax_time_const) = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
    build_axes_channel_properties(ax_open_prob, ax_time_const, v_min, v_max, num_points)

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("biophysics") / "channel_properties"
        save_figure(fig, fig_path)

    return fig


def plot_simulations(
    t_start: float = 0,
    t_end: float = 3,
    dt: float = 0.01,
    ap_peak_time: float = 1.0,
    ap_amplitudes: np.ndarray = np.linspace(10, 100, 10),
    colors: list[ColorType] | ColorType = "black",
    labels: list[str] | None = None,
    show_fig: bool = True,
    save_fig: bool = False,
):
    """Plot the simulations of the NMDAR and VGCC.

    Parameters
    ----------
    t_start : float, optional
        Start time in ms, by default 0
    t_end : float, optional
        End time in ms, by default 3
    dt : float, optional
        Time step for numerical integration (ms), by default 0.01
    ap_peak_time : float, optional
        Time of AP peak in ms, by default 1.0
    ap_amplitudes : np.ndarray, optional
        AP amplitudes to test, by default np.linspace(10, 100, 10)
    colors : list[ColorType] | ColorType, optional
        Colors to use for the traces, by default "black". If a list,
        must be the same length as ap_amplitudes.
    labels : list[str] | str, optional
        Labels to use for the traces, by default None. If a list,
        must be the same length as ap_amplitudes.
    show_fig : bool, optional
        Whether to show the figure, by default True
    save_fig : bool, optional
        Whether to save the figure, by default False

    Returns
    -------
    plt.Figure
        The figure containing the simulations plot
    """
    fig, (ax_voltage, ax_nmdar, ax_vgcc) = plt.subplots(1, 3, figsize=(8, 3.5), sharex=True)
    build_axes_simulations(
        ax_voltage,
        ax_nmdar,
        ax_vgcc,
        t_start,
        t_end,
        dt,
        ap_peak_time,
        ap_amplitudes,
        colors,
        labels,
    )

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("biophysics") / "simulations"
        save_figure(fig, fig_path)

    return fig


def build_axes_channel_properties(
    ax_open_prob: plt.Axes,
    ax_time_const: plt.Axes,
    v_min: float = -80,
    v_max: float = 40,
    num_points: int = 200,
) -> tuple[plt.Axes, plt.Axes]:
    """Build the axes for the channel properties plot.

    Parameters
    ----------
    ax_open_prob : plt.Axes
        Axes for the open probability plot
    ax_time_const : plt.Axes
        Axes for the time constant plot

    Returns
    -------
    tuple[plt.Axes, plt.Axes]
        A tuple containing the open probability and time constant axes
    """
    # Simulation parameters
    v_range = np.linspace(v_min, v_max, num_points)  # Voltage range for steady-state plots

    # Initialize channels
    nmdar = NMDAR()
    vgcc = VGCC()

    ax_open_prob.plot(v_range, nmdar.open_probability(v_range), "k", label="NMDAR")
    ax_open_prob.plot(v_range, vgcc.open_probability_activation(v_range), "b", label="VGCC")
    ax_open_prob.set_xlabel("Membrane Potential (mV)")
    ax_open_prob.set_ylabel("Open Probability")
    ax_open_prob.set_title("Channel Open Probabilities")
    ax_open_prob.legend()

    ax_time_const.plot(v_range, nmdar.time_constant(v_range), "k", label="NMDAR")
    tau_m, _ = vgcc.time_constant(v_range)
    ax_time_const.plot(v_range, tau_m, "b", label="VGCC")
    ax_time_const.set_xlabel("Membrane Potential (mV)")
    ax_time_const.set_ylabel("Time Constant (ms)")
    ax_time_const.set_title("Channel Time Constants")
    ax_time_const.legend()


def build_axes_simulations(
    ax_voltage: plt.Axes,
    ax_nmdar: plt.Axes,
    ax_vgcc: plt.Axes,
    t_start: float = 0,
    t_end: float = 3,
    dt: float = 0.01,
    ap_peak_time: float = 1.0,
    ap_amplitudes: np.ndarray = np.linspace(10, 100, 10),
    v_base: float = -70,
    colors: list[ColorType] | ColorType = "black",
    labels: list[str] | None = None,
    linewidth: float = 1.0,
):
    """Build the axes for the simulations plot.

    Parameters
    ----------
    ax_voltage : plt.Axes
        Axes for the voltage plot
    ax_nmdar : plt.Axes
        Axes for the NMDAR plot
    ax_vgcc : plt.Axes
        Axes for the VGCC plot
    t_start : float, optional
        Start time in ms, by default 0
    t_end : float, optional
        End time in ms, by default 3
    dt : float, optional
        Time step for numerical integration (ms), by default 0.01
    ap_peak_time : float, optional
        Time of AP peak in ms, by default 1.0
    ap_amplitudes : np.ndarray, optional
        AP amplitudes to test, by default np.linspace(10, 100, 10)
    v_base : float, optional
        Base voltage in mV, by default -70
    colors : list[ColorType] | ColorType, optional
        Colors to use for the traces, by default "black". If a list,
        must be the same length as ap_amplitudes.
    labels : list[str] | str, optional
        Labels to use for the traces, by default None. If a list,
        must be the same length as ap_amplitudes.
    linewidth : float, optional
        Linewidth for the traces, by default 1.0

    Returns
    -------
    tuple[plt.Axes, plt.Axes]
        A tuple containing the open probability and time constant axes
    """
    if isinstance(colors, list):
        if len(colors) != len(ap_amplitudes):
            raise ValueError("If colors is a list, it must be the same length as ap_amplitudes.")
    else:
        colors = [colors] * len(ap_amplitudes)

    if labels is None:
        labels = [None] * len(ap_amplitudes)
    elif isinstance(labels, list):
        if len(labels) != len(ap_amplitudes):
            raise ValueError("If labels is a list, it must be the same length as ap_amplitudes.")

    nmdar = NMDAR()
    vgcc = VGCC()

    t_range = np.linspace(t_start, t_end, int((t_end - t_start) / dt))

    # Generate voltage traces and responses for different AP amplitudes
    for amp, color, label in zip(ap_amplitudes, colors, labels):
        # Create AP waveform
        ap = AP(v_amp=amp, v_dur=1, v_base=v_base)
        v_trace = ap.voltage(t_range, ap_peak_time)

        # Initialize state variables to steady state at baseline voltage
        m = vgcc.open_probability_activation(v_trace[0])
        h = vgcc.open_probability_inactivation(v_trace[0])
        n = nmdar.open_probability(v_trace[0])

        # Arrays to store results
        m_trace = np.zeros_like(t_range)
        h_trace = np.zeros_like(t_range)
        n_trace = np.zeros_like(t_range)
        m_trace[0] = m
        h_trace[0] = h
        n_trace[0] = n

        # Numerical integration using Euler's method
        for i in range(1, len(t_range)):
            m += dt * vgcc.dmdt(v_trace[i - 1], m)
            h += dt * vgcc.dhdt(v_trace[i - 1], h)
            n += dt * nmdar.dndt(v_trace[i - 1], n)

            m_trace[i] = m
            h_trace[i] = h
            n_trace[i] = n

        # Calculate open probabilities
        vgcc_p = m_trace**2 * h_trace  # VGCC open probability
        nmdar_p = n_trace  # NMDAR open probability

        ax_voltage.plot(t_range, v_trace, color=color, label=label, linewidth=linewidth)
        ax_nmdar.plot(t_range, nmdar_p, color=color, label=label, linewidth=linewidth)
        ax_vgcc.plot(t_range, vgcc_p, color=color, label=label, linewidth=linewidth)

    return ax_voltage, ax_nmdar, ax_vgcc
