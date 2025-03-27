from typing import Optional, Literal
import numpy as np
import pandas as pd
from scipy.constants import R, physical_constants
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.typing import ColorType

from src.files import get_figure_dir, data_dir
from src.plotting import save_figure, FigParams, Proximal, DistalSimple, DistalComplex
from src.experimental import ElifeData
from src.utils import get_closest_idx

F = physical_constants["Faraday constant"][0]


# From memory need to look up
BUFFER_KD = 250e-3


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
        return "mediumpurple"  # [x / 255 for x in [66, 135, 135]]

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


def get_nevian_params(buffer: Literal["BAPTA", "EGTA", "AVERAGE"] = "AVERAGE"):
    """Retrieve nevian reconstruction data and measure sigmoidal fits to data.

    Parameters
    ----------
    buffer : Literal["BAPTA", "EGTA", "AVERAGE"], optional
        Which buffer data to use to make the data, default is AVERAGE
        which means combining them and making the sigmoid of both datasets.

    Returns
    -------
    dfs : list[pd.DataFrame]
        dataframe of the data for LTP[0] and LTD[1]
    params : list[tuple]
        sigmoidal fit to the data in dfs
    """
    # Read CSVs for STDP Data
    stdp_data_path = data_dir() / "nevian-sakmann-2006"
    stdp_dfs = {}
    for plasticity_type in ["LTP", "LTD"]:
        stdp_dfs[plasticity_type] = {}
        for buffer_type in ["BAPTA", "EGTA"]:
            path = stdp_data_path / f"{plasticity_type}-{buffer_type}.csv"
            stdp_dfs[plasticity_type][buffer_type] = get_csv_data(path)

    if buffer == "BAPTA":
        ltp_df = stdp_dfs["LTP"]["BAPTA"]
        ltd_df = stdp_dfs["LTD"]["BAPTA"]
    elif buffer == "EGTA":
        ltp_df = stdp_dfs["LTP"]["EGTA"]
        ltd_df = stdp_dfs["LTD"]["EGTA"]
    elif buffer == "AVERAGE":
        ltp_df = pd.concat([stdp_dfs["LTP"]["BAPTA"], stdp_dfs["LTP"]["EGTA"]])
        ltd_df = pd.concat([stdp_dfs["LTD"]["BAPTA"], stdp_dfs["LTD"]["EGTA"]])
    else:
        raise ValueError(f"Did not recognize buffer, received: {buffer}")

    dfs = dict(LTP=ltp_df, LTD=ltd_df)
    params = dict(LTP=fit_sigmoid(ltp_df), LTD=fit_sigmoid(ltd_df))

    return dfs, params


def measure_transfer_functions(
    data: dict,
    flip_ltd: bool = True,
):
    # Calculate peak calcium influx evoked by each channel
    # Note: it's appropriate to do this independently, because the two curves represent relative
    # calcium influx as a function of the voltage stimulus - but since we don't know what the
    # maximum channel conductance is in a dendritic spine, we can't compare NMDAR to VGCC. So,
    # to keep things simple, I assume that the maximum calcium influx that enters via a 1ms AP
    # reflects the maximum calcium possible from these channels. We're ignoring lots of things,
    # including but not limited to: buffering properties, extrusion properties, glutamate binding
    # dynamics, etc etc etc.

    # Get elife data
    elife_data = ElifeData()

    # Estimate relative peak concentration of NMDAR vs VGCC based on maximum
    # measurements of calcium concentration for AP only or NL component (which
    # mostly from NMDARs)
    max_nl_component = np.max(elife_data.nl_component)  # to estimate how much calcium can come in due to NMDARs
    max_ap_only = np.max(elife_data.spk_new[elife_data.idx_ap])  # to estimate how much calcium can come in due to VGCCs
    relative_ltp_scale = max_nl_component / max_ap_only  # to relate VGCC and NMDAR conductance to calcium

    # ~~~ The only thing to consider is that estimating max requires the data to have picked an
    # ~~~ AP amplitude that evokes near maximal concentration!!!
    nmdar_peak = np.max(data["nmdar_integral_ca"])
    vgcc_peak = np.max(data["vgcc_integral_ca"])
    ltp_ca_to_buffer = 1.0 / nmdar_peak / relative_ltp_scale
    ltd_ca_to_buffer = 1.0 / vgcc_peak
    effective_ca_ltp = ltp_ca_to_buffer * np.array(data["nmdar_integral_ca"])
    effective_ca_ltd = ltd_ca_to_buffer * np.array(data["vgcc_integral_ca"])

    # This measures the transfer function from relative calcium to plasticity magnitude
    # with the maximum calcium value being 1.0 or whatever the average calcium influx
    # for a typical pre/post or post/pre protocol is that drives plasticity.
    # --
    # So, to convert the "*_integral_ca" into units of relative calcium, we should divide
    # integral calcium by the maximum observed (at least over a valid range of AP amplitudes
    # to capture the full range). To attempt to compare NMDARs and VGCCs (which we don't
    # have max channel conductance data for), we update based on the peak NMDAR/VGCC ca influx.
    ca_concentration, ltp_transfer = plasticity_transfer_function(
        data["params"]["LTP"],
        LTP=True,
        max_buffer_concentration=10.0,
        num_points=10001,
    )
    ltd_transfer = plasticity_transfer_function(
        data["params"]["LTD"],
        LTP=False,
        max_buffer_concentration=10.0,
        num_points=10001,
    )[1]

    # Get closest index and error for the "effective" concentration
    # and the "measured" concentration
    idx_ltp, error_ltp = get_closest_idx(ca_concentration, effective_ca_ltp)
    idx_ltd, error_ltd = get_closest_idx(ca_concentration, effective_ca_ltd)
    ltp_transfer = ltp_transfer[idx_ltp]
    ltd_transfer = ltd_transfer[idx_ltd]

    if flip_ltd:
        # flip sign because transfer is negative
        ltd_transfer = -1 * ltd_transfer

    results = dict(
        LTP=ltp_transfer,
        LTD=ltd_transfer,
        error_ltp_ca_estimate=error_ltp,
        error_ltd_ca_estimate=error_ltd,
    )
    return results


def simplistic_plasticity_transfer_function(
    params: tuple[float],
    x_values: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a transfer function for plasticity based on sigmoid fit of buffer effects.

    In this model, we assume the reduction in plasticity caused by adding more buffer linearly
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


def plasticity_transfer_function(
    params: tuple[float],
    LTP: bool,
    max_buffer_concentration: float = 5.0,
    kd: float = BUFFER_KD,
    num_points: int = 1001,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a transfer function for plasticity based on sigmoid fit of buffer effects.

    We model how buffers affect calcium dynamics by assuming the buffer is in the buffer
    capcity regime (e.g. [CaB]/[Ca] = [B]/Kd). Therefore, an increase in the concentration
    of the buffer linearly corresponds to a reduction in calcium concentration.

    From the Nevian pharmacology data, we know the relationship between the magnitude of
    plasticity and the concentration of calcium ~above a certain threshold~ for plasticity.

    Let B_{threshold} be the lowest buffer concentration at which plasticity is fully
    blocked (e.g. just over 1 mM for LTP data, see Nevian reconstruction). Let the associated
    calcium at this buffer concentration be Ca_{threshold}.

    Parameters
    ----------
    params : tuple[float]
        Parameters from sigmoid fit (L, x0, k, b) where:
        L: Maximum value (will be set to 1.0)
        x0: x-value of the sigmoid's midpoint (half-effective concentration)
        k: Steepness of the curve (hill coefficient)
        b: y-offset (will be set to 0)
    LTP : boolean
        True if we're modeling LTP (assumes the sigmoid model is positive, and
        negative for LTD).
    max_buffer_concentration : float
        maximum buffer concentration to model, default=5.0
    kd : float
        The binding coefficient of the buffer we're modeling.
    num_points : int
        The number of points to use for the concentration values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - transfer_function: Normalized sigmoid values (0-1 range)
        - buffer_concentration: Buffer concentration values
        -
    """
    # Set up effective calcium dose values
    buffer_concentration = np.linspace(0, max_buffer_concentration, num_points)
    kappa = buffer_concentration / kd

    L, x0, k, b = params
    transfer = sigmoid(buffer_concentration, L, x0, k, b)
    transfer = np.maximum(transfer, 0) if LTP else np.minimum(transfer, 0)

    # Measure calcium remaining after adding buffer
    Ca_remaining = 1.0 / (1 + kappa)

    # Return calcium and plasticity (in order from lowest to highest calcium)
    return Ca_remaining[::-1], transfer[::-1]


def run_simulations(
    buffer: Literal["BAPTA", "EGTA", "AVERAGE"] = "AVERAGE",
    max_ap_amplitude: float = 100,
    num_ap_amplitudes: int = 100,
) -> plt.Axes:
    """Run simulations of conductance data and expected plasticity curves."""

    # Get Nevian params
    params = get_nevian_params(buffer)[1]

    # Simulation parameters
    v_range = np.linspace(-80, 40, 200)  # Voltage range for steady-state plots
    dt = 0.01  # Time step for numerical integration (ms)
    t_start = 0
    t_end = 5
    t_range = np.linspace(t_start, t_end, int((t_end - t_start) / dt))
    v_base = -70  # base voltage of APs
    ap_peak_time = 1  # Time of AP peak in ms
    ap_amplitudes = np.linspace(0, max_ap_amplitude, num_ap_amplitudes)  # AP amplitudes to test
    ap_peaks = v_base + ap_amplitudes
    ca_in = 75e-9
    ca_out = 1.5e-6

    # Initialize channels
    nmdar = NMDAR()
    vgcc = VGCC()

    # Data dictionary for storing all results
    data = dict(
        params=params,
        t_range=t_range,
        v_range=v_range,
        ap_amplitudes=ap_amplitudes,
        ap_peaks=ap_peaks,
        ca_in=ca_in,
        ca_out=ca_out,
        nmdar=vars(nmdar),
        vgcc=vars(vgcc),
        v_trace=[],
        nmdar_p=[],
        vgcc_p=[],
        nmdar_ica=[],
        vgcc_ica=[],
        LTP=[],
        LTD=[],
    )

    # Generate voltage traces and responses for different AP amplitudes
    for amp in ap_amplitudes:
        # Create AP waveform
        ap = AP(v_amp=amp, v_dur=1, v_base=-70)
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

        # Calculate calcium current
        vgcc_ica = -compute_current(v_trace, vgcc_p, ca_in, ca_out)
        nmdar_ica = -compute_current(v_trace, nmdar_p, ca_in, ca_out)

        data["v_trace"].append(v_trace)
        data["nmdar_p"].append(nmdar_p)
        data["vgcc_p"].append(vgcc_p)
        data["nmdar_ica"].append(nmdar_ica)
        data["vgcc_ica"].append(vgcc_ica)

    data["nmdar_peak_p"] = [np.max(p) for p in data["nmdar_p"]]
    data["vgcc_peak_p"] = [np.max(p) for p in data["vgcc_p"]]
    data["nmdar_integral_ca"] = [np.sum(ica - ica[0]) for ica in data["nmdar_ica"]]
    data["vgcc_integral_ca"] = [np.sum(ica - ica[0]) for ica in data["vgcc_ica"]]

    # Calculate plasticity transfer functions
    transfer_function_results = measure_transfer_functions(data, flip_ltd=True)
    data = data | transfer_function_results

    return data


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


def build_axes_nevian_reconstruction(
    ax: plt.Axes,
    buffer: Literal["BAPTA", "EGTA", "AVERAGE"] = "AVERAGE",
    x_legend: float = 1.8,
    y_legend: float = 0.95,
    y_offset: float = -0.15,
    ha_legend: str = "right",
    va_legend: str = "center",
) -> plt.Axes:

    dfs, params = get_nevian_params(buffer=buffer)
    keys = ["LTP", "LTD"]
    names = ["LTP ($\propto I_{NMDAR}$)", "LTD ($\propto I_{VGCC}$)"]
    colors = [NMDAR.color(), VGCC.color()]

    # Plot results
    xfit = np.linspace(0.0, 2.0, 101)
    for key, name, color in zip(keys, names, colors):
        xdata = dfs[key]["x"]
        ydata = dfs[key]["y"]
        kwargs = dict(
            marker=".",
            markerfacecolor=color,
            markersize=FigParams.scattersize,
            markeredgecolor="none",
            linestyle="none",
            label=name,
        )
        if key == "LTD":
            ydata = -ydata
        ax.plot(xdata, ydata, **kwargs, zorder=1000)

        L, x0, k, b = params[key]
        yfit = sigmoid(xfit, L, x0, k, b)
        if key == "LTD":
            yfit = -yfit
        kwargs = dict(linewidth=FigParams.linewidth, linestyle="-", color=color)
        ax.plot(xfit, yfit, **kwargs, zorder=900)

    ax.axhline(0, color="black", linewidth=FigParams.thinlinewidth, linestyle="--", zorder=-1000)
    ax.set_ylim(-0.3, 1.3)

    for iname, name in enumerate(names):
        ax.text(
            x_legend,
            y_legend + iname * y_offset,
            name,
            color=colors[iname],
            ha=ha_legend,
            va=va_legend,
            fontsize=FigParams.fontsize,
        )


def build_axes_transfer_functions(
    ax_transfer: plt.Axes,
    ax_prediction: plt.Axes,
    data: dict,
    ap_amplitudes: np.ndarray,
    x_legend: float = 1.8,
    y_legend: float = 0.95,
    y_offset: float = -0.15,
    ha_legend: str = "right",
    va_legend: str = "center",
) -> tuple[plt.Axes, plt.Axes]:
    ax_transfer.plot(data["ap_peaks"], data["LTP"], color=NMDAR.color(), linewidth=FigParams.thinlinewidth)
    ax_transfer.plot(data["ap_peaks"], data["LTD"], color=VGCC.color(), linewidth=FigParams.thinlinewidth)
    ax_transfer.set_ylim(-0.1, 1.1)

    idx_to_aps, error = get_closest_idx(data["ap_amplitudes"], np.array(ap_amplitudes))
    if np.any(error > 0.1):
        raise ValueError(
            "Failed to find APs with desired amplitude, run_simulations with a wider range and precision!!"
        )

    if len(ap_amplitudes) != 3:
        raise ValueError("This function expects 3 AP amplitudes")

    ltp_groups = data["LTP"][idx_to_aps]
    ltd_groups = data["LTD"][idx_to_aps]
    colors = [Proximal.color, DistalSimple.color, DistalComplex.color]

    plasticity = np.stack([ltp_groups, ltd_groups], axis=1)
    plasticity = np.clip(plasticity, 0.0, 1.0)
    for pp, color in zip(plasticity, colors):
        ax_prediction.plot(
            [0, 1],
            pp,
            color=color,
            linewidth=FigParams.thinlinewidth,
            marker=".",
            markersize=FigParams.scattersize,
        )

    ax_prediction.set_xlim(-0.2, 1.2)
    ax_prediction.set_ylim(-0.1, 1.1)
