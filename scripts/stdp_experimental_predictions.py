from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from src.conductance import (
    NMDAR,
    VGCC,
    AP,
    get_csv_data,
    sigmoid,
    fit_sigmoid,
    compute_current,
    plasticity_transfer_function,
    simplistic_plasticity_transfer_function,
)
from src.files import get_figure_dir, data_dir
from src.plotting import save_figure


def show_nevian_reconstruction(show_fig: bool = True, save_fig: bool = False) -> plt.Figure:
    # Read CSVs for STDP Data
    stdp_data_path = data_dir() / "nevian-sakmann-2006"
    stdp_files = [
        f"{plasticity_type}-{buffer_type}.csv"
        for plasticity_type in ["LTP", "LTD"]
        for buffer_type in ["BAPTA", "EGTA"]
    ]
    stdp_file_paths = [stdp_data_path / stdp_file for stdp_file in stdp_files]

    dfs = [get_csv_data(file_path) for file_path in stdp_file_paths]
    params = [fit_sigmoid(df) for df in dfs]

    # Plot results
    idx_plot = [0, 0, 1, 1]
    filled = [True, False, True, False]
    fit_color = ["black", "gray", "black", "gray"]
    label = ["BAPTA", "EGTA", "BAPTA", "EGTA"]
    xfit = np.linspace(0.0, 2.0, 101)

    fig, ax = plt.subplots(1, 2, figsize=(7, 3), layout="constrained", sharex=True)
    for idata in range(len(stdp_files)):
        xdata = dfs[idata]["x"]
        ydata = dfs[idata]["y"]
        kwargs = dict(
            marker="o",
            markerfacecolor="k" if filled[idata] else "w",
            color="k",
            linestyle="none",
            label=label[idata],
        )
        ax[idx_plot[idata]].plot(xdata, ydata, **kwargs, zorder=1000)

        L, x0, k, b = params[idata]
        yfit = sigmoid(xfit, L, x0, k, b)
        kwargs = dict(linewidth=2, linestyle="-", color=fit_color[idata])
        ax[idx_plot[idata]].plot(xfit, yfit, **kwargs)

    ax[0].axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax[1].axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax[0].set_ylim(-0.5, 1.5)
    ax[1].set_ylim(-1.5, 0.5)
    ax[0].set_xticks(np.linspace(0.0, 2.0, 5))
    ax[1].set_xticks(np.linspace(0.0, 2.0, 5))
    ax[0].set_yticks(np.linspace(-0.5, 1.5, 5))
    ax[1].set_yticks(np.linspace(-1.5, 0.5, 5))
    ax[0].legend(loc="upper right")
    ax[1].legend(loc="lower right")
    ax[0].set_xlabel("[buffer] (mM)")
    ax[1].set_xlabel("[buffer] (mM)")
    ax[0].set_ylabel("relative magnitude of LTP")
    ax[1].set_ylabel("relative magnitude of LTD")

    if show_fig:
        plt.show()

    if save_fig:
        fig_path = get_figure_dir("stdp_prediction") / "nevian_reconstruction"
        save_figure(fig, fig_path)

    return fig


def show_plasticity_transfer_function(show_fig: bool = True, save_fig: bool = False) -> plt.Figure:
    # Read CSVs for STDP Data
    stdp_data_path = data_dir() / "nevian-sakmann-2006"

    plasticity_types = ["LTP", "LTD"]
    colors = [NMDAR.color(), VGCC.color()]
    buffer_type = "EGTA"
    stdp_files = [f"{ptype}-{buffer_type}.csv" for ptype in plasticity_types]
    stdp_file_paths = [stdp_data_path / stdp_file for stdp_file in stdp_files]

    dfs = [get_csv_data(file_path) for file_path in stdp_file_paths]
    params = [fit_sigmoid(df) for df in dfs]

    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    for ii, ptype in enumerate(plasticity_types):
        LTP = ptype == "LTP"
        ca_signal, transfer_function = plasticity_transfer_function(
            params[ii],
            LTP=LTP,
            max_buffer_concentration=5.0,
            kd=0.25,
        )
        kwargs = dict(linewidth=2, linestyle="-", color=colors[ii], label=ptype)
        ax.plot(ca_signal, transfer_function, **kwargs)

    ax.legend(loc="upper left")
    ax.set_xlabel("[Ca] Concentration (AU)")
    ax.set_ylabel("Fractional Plasticity Magnitude (AU)")

    if show_fig:
        plt.show()

    if save_fig:
        fig_path = get_figure_dir("stdp_prediction") / "plasticity_transfer_function"
        save_figure(fig, fig_path)

    return fig


def run_simulations(num_ap_amplitudes: int = 10):
    # Do simulations of AP-evoked calcium conductance, current, and evoked LTP / LTD

    # Read CSVs for STDP Data
    plasticity_type = ["LTP", "LTD"]
    buffer_type = "EGTA"
    stdp_data_path = data_dir() / "nevian-sakmann-2006"
    stdp_files = [f"{ptype}-{buffer_type}.csv" for ptype in plasticity_type]
    stdp_file_paths = [stdp_data_path / stdp_file for stdp_file in stdp_files]

    dfs = [get_csv_data(file_path) for file_path in stdp_file_paths]
    params = {ptype: fit_sigmoid(df) for ptype, df in zip(plasticity_type, dfs)}

    # Simulation parameters
    v_range = np.linspace(-80, 40, 200)  # Voltage range for steady-state plots
    dt = 0.01  # Time step for numerical integration (ms)
    t_start = 0
    t_end = 5
    t_range = np.linspace(t_start, t_end, int((t_end - t_start) / dt))
    ap_peak_time = 1  # Time of AP peak in ms
    ap_amplitudes = np.linspace(0, 100, num_ap_amplitudes)  # AP amplitudes to test
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
    buffer_scale = 1.0
    nmdar_peak = np.max(data["nmdar_integral_ca"])
    vgcc_peak = np.max(data["vgcc_integral_ca"])
    ltp_ca_to_buffer = buffer_scale / nmdar_peak
    ltd_ca_to_buffer = buffer_scale / vgcc_peak

    effective_ca_ltp = ltp_ca_to_buffer * np.array(data["nmdar_integral_ca"])
    effective_ca_ltd = ltd_ca_to_buffer * np.array(data["vgcc_integral_ca"])
    data["LTP"] = simplistic_plasticity_transfer_function(data["params"]["LTP"], x_values=effective_ca_ltp)[0]
    data["LTD"] = simplistic_plasticity_transfer_function(data["params"]["LTD"], x_values=effective_ca_ltd)[0]

    return data


def show_estimated_plasticity(data: dict, buffer_scale: float = 1.0, show_fig: bool = True, save_fig: bool = False):
    """Show plasticity kernels based on simulations

    The estimate of plasticity kernels depends on a few hard-coded free parameters
    that we can't really measure. To be thorough and rigorous, this function allows
    us to generate the plasticity kernels using a variety of those free parameters.

    There are only two parameters to set (that are essentially equivalent...):
    max_ica: the value (float) of the maximum calcium influx assumed to enter
             via a particular channel
    buffer_scale: the value (float) of buffer corresponding to the maximum effective
                  dose of calcium (see :func:`~src.conductance.plasticity_transfer_function`)
                  for more explanation.
    """
    # Calculate peak calcium influx evoked by each channel
    # Note: it's appropriate to do this independently, because the two curves represent relative
    # calcium influx as a function of the voltage stimulus - but since we don't know what the
    # maximum channel conductance is in a dendritic spine, we can't compare NMDAR to VGCC. So,
    # to keep things simple, I assume that the maximum calcium influx that enters via a 1ms AP
    # reflects the maximum calcium possible from these channels. We're ignoring lots of things,
    # including but not limited to: buffering properties, extrusion properties, glutamate binding
    # dynamics, etc etc etc.

    # ~~~ The only thing to consider is that estimating max requires the data to have picked an
    # ~~~ AP amplitude that evokes near maximal concentration!!!
    nmdar_peak = np.max(data["nmdar_integral_ca"])
    vgcc_peak = np.max(data["vgcc_integral_ca"])
    common_peak = max([nmdar_peak, vgcc_peak])
    ltp_ca_to_buffer = buffer_scale / common_peak
    ltd_ca_to_buffer = buffer_scale / common_peak

    effective_ca_ltp = ltp_ca_to_buffer * np.array(data["nmdar_integral_ca"])
    effective_ca_ltd = ltd_ca_to_buffer * np.array(data["vgcc_integral_ca"])
    ca_concentration, ltp_transfer = plasticity_transfer_function(
        data["params"]["LTP"],
        LTP=True,
        max_buffer_concentration=10.0,
        kd=0.25,
        num_points=10001,
    )
    ltd_transfer = plasticity_transfer_function(
        data["params"]["LTD"],
        LTP=False,
        max_buffer_concentration=10.0,
        kd=0.25,
        num_points=10001,
    )[1]

    def get_closest_idx(reference, comparison):
        idx = np.zeros(comparison.shape, dtype=np.int32)
        for ic, c in enumerate(comparison):
            idx[ic] = np.argmin(np.abs(reference - c))
        error = comparison - reference[idx]
        return idx, error

    # Get closest index and error for the "effective" concentration
    # and the "measured" concentration
    idx_ltp, error_ltp = get_closest_idx(ca_concentration, effective_ca_ltp)
    idx_ltd, error_ltd = get_closest_idx(ca_concentration, effective_ca_ltd)
    ltp_transfer = ltp_transfer[idx_ltp]
    ltd_transfer = -1 * ltd_transfer[idx_ltd]  # flip sign because transfer is negative

    ap_peaks = [np.max(v_trace) for v_trace in data["v_trace"]]

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), layout="constrained")
    ax.plot(ap_peaks, ltp_transfer / np.max(ltp_transfer), color=NMDAR.color(), linewidth=1.5, label="LTP")
    ax.plot(ap_peaks, ltd_transfer / np.max(ltd_transfer), color=VGCC.color(), linewidth=1.5, label="LTD")
    ax.set_xlabel("AP Peak Voltage (mV)")
    ax.set_ylabel("Plasticity Magnitude (AU)")
    ax.legend(loc="upper left")

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("stdp_prediction") / "estimated_plasticity_kernel"
        save_figure(fig, fig_path)

    return fig


if __name__ == "__main__":
    show_fig = False
    save_fig = True
    data = run_simulations(num_ap_amplitudes=100)

    # Show our reconstruction of the Nevian data
    fig = show_nevian_reconstruction(show_fig=show_fig, save_fig=save_fig)

    # Show the relative plasticity trasnfer functions
    fig = show_plasticity_transfer_function(show_fig=show_fig, save_fig=save_fig)

    # Show the estimated plasticity for LTP and LTD based on our biophysical simulations
    fig = show_estimated_plasticity(data, show_fig=show_fig, save_fig=save_fig)
