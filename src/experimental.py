from dataclasses import dataclass
import numpy as np
from scipy.io import loadmat
from scipy.signal import medfilt
import matplotlib.pyplot as plt

from .files import data_dir, get_figure_dir
from .utils import resolve_dataclass, nanshift
from .plotting import save_figure, beeswarm, format_spines
from .plotting import Proximal, DistalSimple, DistalComplex


def get_elife_data() -> dict:
    """Load the data that was used to generate figure 2 of the ELife paper.

    Returns a dictionary of the matlab structure using scipy's loadmat method
    with simplicy_cells=True.
    """
    data_path = data_dir() / "elife-paper-fig2.mat"
    return loadmat(data_path, simplify_cells=True)["res"]


def correct_pmt(x):
    """Correct PMT closure."""
    baseline = np.mean(x[60:98], axis=0)
    std = np.std(x[60:98], axis=0)
    x[98:101] = baseline + std * np.random.randn(*baseline.shape)
    return x


@dataclass
class DendriticSiteParams:
    """Parameters used to classify each dendritic site.

    Proximal vs distal comparisons are simply based on the distance
    of the site from the soma. "Simple" vs "Complex" sites are based
    on the amplitude of the AP response because we demonstrated that
    this was determined by the dendritic branching structure within
    a given cell.
    """

    distance_proximal: tuple[float, float] = (0, 80)
    distance_distal: tuple[float, float] = (100, 300)
    ap_amp_simple: tuple[float, float] = (0.1, 0.3)
    ap_amp_complex: tuple[float, float] = (0.0, 0.04)


@dataclass
class ElifeDataParams:
    """Parameters for ElifeData processing."""

    min_allowed: float = 0.001
    filter_depth: int = 5
    pk_sum_start: int = -2
    pk_sum_end: int = 8
    pk_window_start: float = 100
    pk_window_end: float = 150
    bs_win: tuple = (80, 98)
    pk_length: int = 20


class ElifeData:
    idx_ap = 0
    idx_glu = 1
    idx_gap = 2

    def __init__(self, params: ElifeDataParams | dict | None = None):
        """Initialize the ElifeData class with processing parameters.

        Parameters
        ----------
        params : ElifeDataParams | dict | None, optional
            Parameters for data processing
        """
        self.params = resolve_dataclass(params, ElifeDataParams)
        self._data = get_elife_data()
        self._register_core_data()

    def _register_core_data(self):
        """Extract and register core data as attributes."""
        data = self._data
        self.NR = len(data)
        self.distance = np.array([cell["distance"] for cell in data])

        # Extract time vectors and traces
        self.ptv = np.arange(0.1, 1000.1, 0.1)
        self.pglu = np.stack([cell["pglu"] for cell in data])
        self.pap = np.stack([cell["pap"] for cell in data])

        self.tvec = np.arange(1, 513)
        self.sap = np.stack([cell["sap"] for cell in data], axis=1)
        self.sglu = np.stack([cell["sglu"] for cell in data], axis=1)
        self.sgap = np.stack([cell["sgap"] for cell in data], axis=1)
        self.ssyn = np.stack([cell["ssyn"] for cell in data], axis=1)
        self.spk = np.stack([cell["spk"] for cell in data], axis=1)
        self.sbase = np.stack([cell["sbase"] for cell in data], axis=1)

        self.dap = np.stack([cell["dap"] for cell in data], axis=1)
        self.dglu = np.stack([cell["dglu"] for cell in data], axis=1)
        self.dgap = np.stack([cell["dgap"] for cell in data], axis=1)
        self.dpk = np.stack([cell["dpk"] for cell in data], axis=1)
        self.dbase = np.stack([cell["dbase"] for cell in data], axis=1)

        # Apply minimum value constraints
        self.spk[self.spk < self.params.min_allowed] = self.params.min_allowed
        self.dpk[self.dpk < self.params.min_allowed] = self.params.min_allowed

    @property
    def sapbase(self):
        """Spine AP trace minus baseline."""
        return self.sap - self.sbase[self.idx_ap]

    @property
    def sglubase(self):
        """Spine glutamate trace minus baseline."""
        return self.sglu - self.sbase[self.idx_glu]

    @property
    def sgapbase(self):
        """Spine glutamate+AP trace minus baseline."""
        return self.sgap - self.sbase[self.idx_gap]

    @property
    def dapbase(self):
        """Dendrite AP trace minus baseline."""
        return self.dap - self.dbase[self.idx_ap]

    @property
    def dglubase(self):
        """Dendrite glutamate trace minus baseline."""
        return self.dglu - self.dbase[self.idx_glu]

    @property
    def dgapbase(self):
        """Dendrite glutamate+AP trace minus baseline."""
        return self.dgap - self.dbase[self.idx_gap]

    @property
    def pk_window(self):
        """Get peak window indices."""
        start_idx = np.where(self.tvec >= self.params.pk_window_start)[0][0]
        end_idx = np.where(self.tvec >= self.params.pk_window_end)[0][0]
        return np.arange(start_idx, end_idx)

    @property
    def pk_sum_window(self):
        """Get peak sum range."""
        return np.arange(self.params.pk_sum_start, self.params.pk_sum_end)

    @property
    def spk_new(self):
        """Improved peak detection for spines."""
        if hasattr(self, "_spk_new"):
            return self._spk_new

        spk_new = np.zeros_like(self.spk)

        for nr in range(self.NR):
            # AP
            cidx = np.argmax(medfilt(self.sapbase[self.pk_window, nr], self.params.filter_depth))
            cidx = cidx + self.pk_window[0]
            spk_new[self.idx_ap, nr] = np.mean(self.sapbase[cidx + self.pk_sum_window, nr])

            # Glutamate
            cidx = np.argmax(medfilt(self.sglubase[self.pk_window, nr], self.params.filter_depth))
            cidx = cidx + self.pk_window[0]
            spk_new[self.idx_glu, nr] = np.mean(self.sglubase[cidx + self.pk_sum_window, nr])

            # Glutamate + AP
            cidx = np.argmax(medfilt(self.sgapbase[self.pk_window, nr], self.params.filter_depth))
            cidx = cidx + self.pk_window[0]
            spk_new[self.idx_gap, nr] = np.mean(self.sgapbase[cidx + self.pk_sum_window, nr])

        self._spk_new = spk_new
        return spk_new

    @property
    def dpk_new(self):
        """Improved peak detection for dendrites."""
        if hasattr(self, "_dpk_new"):
            return self._dpk_new

        dpk_new = np.zeros_like(self.dpk)

        for nr in range(self.NR):
            # AP
            cidx = np.argmax(medfilt(self.dapbase[self.pk_window, nr], self.params.filter_depth))
            cidx = cidx + self.pk_window[0]
            dpk_new[self.idx_ap, nr] = np.mean(self.dapbase[cidx + self.pk_sum_window, nr])

            # Glutamate
            cidx = np.argmax(medfilt(self.dglubase[self.pk_window, nr], self.params.filter_depth))
            cidx = cidx + self.pk_window[0]
            dpk_new[self.idx_glu, nr] = np.mean(self.dglubase[cidx + self.pk_sum_window, nr])

            # Glutamate + AP
            cidx = np.argmax(medfilt(self.dgapbase[self.pk_window, nr], self.params.filter_depth))
            cidx = cidx + self.pk_window[0]
            dpk_new[self.idx_gap, nr] = np.mean(self.dgapbase[cidx + self.pk_sum_window, nr])

        self._dpk_new = dpk_new
        return dpk_new

    @property
    def syn_peak_data(self):
        """Calculate peak for synthetic."""
        if hasattr(self, "_syn_peak_data"):
            return self._syn_peak_data

        bswin = self.params.bs_win
        pklength = self.params.pk_length

        synpk = np.zeros(self.NR)
        synpk_new = np.zeros(self.NR)
        synpk_idx = np.zeros(self.NR, dtype=int)

        for s in range(self.NR):
            synpk_idx[s] = np.argmax(medfilt(self.ssyn[self.pk_window, s], self.params.filter_depth))
            synpk_idx[s] = synpk_idx[s] + self.pk_window[0]
            synpk[s] = np.mean(self.ssyn[np.arange(pklength) + synpk_idx[s], s])
            synpk_new[s] = np.mean(self.ssyn[synpk_idx[s] + self.pk_sum_window, s])

        synbase = np.mean(self.ssyn[bswin[0] : bswin[1]], axis=0)
        synpk = synpk - synbase
        synpk_new = synpk_new - synbase

        self._syn_peak_data = {"synpk": synpk, "synpk_new": synpk_new, "synbase": synbase, "synpk_idx": synpk_idx}
        return self._syn_peak_data

    @property
    def synpk_new(self):
        """Get the improved synthetic peak."""
        return self.syn_peak_data["synpk_new"]

    @property
    def ampreglu(self):
        """Calculate amplification relative to glutamate."""
        result = self.nl_component / np.abs(self.spk_new[self.idx_glu])
        result[result < 0] = 0
        return result

    @property
    def nl_component(self):
        """Calculate nonlinear component."""
        result = self.spk_new[self.idx_gap] - self.synpk_new
        result[result < self.params.min_allowed] = self.params.min_allowed
        return result

    @property
    def syntrace(self):
        """Calculate synthetic trace."""
        sg = self.sglubase.copy()
        sg = correct_pmt(sg)
        return sg + nanshift(self.sapbase, n=5, token=0, axis=0)

    @property
    def amptrace(self):
        """Calculate amplification trace."""
        sg = self.sgapbase.copy()
        sg = correct_pmt(sg)
        return sg - self.syntrace

    @property
    def ampretrace(self):
        """Calculate amplification relative trace."""
        return self.amptrace / np.abs(self.spk[self.idx_glu])


def plot_amplification_demonstration(
    data: ElifeData,
    icell: int = 17,
    start_pos: int = 20,
    delta_pos: int = 10,
    show_fig: bool = True,
    save_fig: bool = False,
):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax = build_ax_amplification_demonstration(ax, data, icell, start_pos, delta_pos)

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("experimental_data") / "amplification_demonstration"
        save_figure(fig, fig_path)

    return fig


def plot_formatted_elife_data(
    data: ElifeData,
    show_error: bool = False,
    se: bool = True,
    show_fig: bool = True,
    save_fig: bool = False,
):
    """Plot formatted ELife data.

    Parameters
    ----------
    data : ElifeData
        The ELife data to plot.
    show_error : bool, optional
        Whether to show the error bars, by default False.
    se : bool, optional
        Whether to show the standard error, by default True.
    show_fig : bool, optional
        Whether to show the figure, by default True.
    save_fig : bool, optional
        Whether to save the figure, by default False.
    """
    fig = plt.figure(figsize=(6, 6), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.7])
    ax_ap_trace = fig.add_subplot(gs[0, 0])
    ax_amp_trace = fig.add_subplot(gs[1, 0])
    ax_ap_peaks = fig.add_subplot(gs[0, 1])
    ax_amp_peaks = fig.add_subplot(gs[1, 1])

    build_axes_formatted_elife_data(ax_ap_trace, ax_amp_trace, ax_ap_peaks, ax_amp_peaks, data, show_error, se)

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("experimental_data") / "formatted_elife_data"
        save_figure(fig, fig_path)

    return fig


def build_ax_amplification_demonstration(
    ax: plt.Axes,
    data: ElifeData,
    icell: int = 17,
    start_pos: int = 20,
    delta_pos: int = 10,
):
    """Build the amplification demonstration figure on a given axis.

    Parameters
    ----------
    ax : plt.Axes
        The axis to build the figure on.
    data : ElifeData
        The ELife data to plot.
    icell : int, optional
        The cell index to plot, by default 17.
    start_pos : int, optional
        The start position of the demonstration, by default 20.
    delta_pos : int, optional
        The delta position of the demonstration, by default 10.

    Returns
    -------
    ax : plt.Axes
    """
    c_ap = data.sapbase[:, icell]
    c_glu = correct_pmt(data.sglubase[:, icell])
    c_gap = correct_pmt(data.sgapbase[:, icell])
    c_syn = data.syntrace[:, icell]

    c_ap_peak = data.spk_new[data.idx_ap, icell]
    c_glu_peak = data.spk_new[data.idx_glu, icell]
    c_gap_peak = data.spk_new[data.idx_gap, icell]
    c_nl = c_gap_peak - c_ap_peak - c_glu_peak

    names = ["$\Delta Ca_{1 AP}$", "$\Delta Ca_{Glu}$", "$\Delta Ca_{pairing}$", "$\Delta Ca_{amp}$"]
    colors = ["black", "red", "darkorange", "darkviolet"]

    start_offset = start_pos - 100
    start_offset = 67
    ax.axhline(0, color="black", linewidth=0.7)
    ax.plot(data.tvec[start_offset:] - 100, c_ap[start_offset:], color=colors[0], linewidth=1.0)
    ax.plot(data.tvec[start_offset:] - 100, c_glu[start_offset:], color=colors[1], linewidth=1.0)
    ax.plot(data.tvec[start_offset:] - 100, c_gap[start_offset:], color=colors[2], linewidth=1.0)
    ax.plot(data.tvec[start_offset:] - 100, c_syn[start_offset:], color=colors[0], linewidth=1.0)
    ax.plot(data.tvec[start_offset:] - 100, c_syn[start_offset:], color=colors[1], linestyle="--", linewidth=1.0)
    ax.plot([0, 0], [-0.09, -0.04], color="black", linewidth=1.5, linestyle="-")

    def stem(x, y, ystart=0, color="black", linewidth=2.0):
        ax.plot([x, x], [ystart, y], color=color, linewidth=linewidth)
        ax.plot(x, y, color=color, marker="o", markersize=4, markeredgecolor=color, markerfacecolor=color)

    start_ap = start_pos - 100
    start_glu = start_ap + delta_pos
    start_gap = start_ap + 2 * delta_pos
    start_syn = start_ap + 3 * delta_pos
    stem(start_ap, c_ap_peak, color=colors[0])
    stem(start_glu, c_glu_peak, color=colors[1])
    stem(start_gap, c_gap_peak, color=colors[2])
    stem(start_syn, c_ap_peak + c_glu_peak + c_nl, ystart=c_ap_peak + c_glu_peak, color=colors[3])
    stem(start_syn, c_ap_peak + c_glu_peak, ystart=c_ap_peak, color=colors[1])
    stem(start_syn, c_ap_peak, color=colors[0])

    # Create legends and scale bars
    scale_x_root = 200
    scale_x_mag = 50
    scale_y_root = 0.3
    scale_y_mag = 0.07
    scale_y_offset = 0.01

    ax.plot([scale_x_root, scale_x_root], [scale_y_root, scale_y_root + scale_y_mag], color="black", linewidth=1.0)
    ax.plot([scale_x_root, scale_x_root + scale_x_mag], [scale_y_root, scale_y_root], color="black", linewidth=1.0)
    ax.text(scale_x_root, scale_y_root - scale_y_offset, f"{scale_x_mag}ms", ha="left", va="top", fontsize=12)
    ax.text(
        scale_x_root,
        scale_y_root,
        f"{int(scale_y_mag*100)}% $\Delta$ G/R",
        ha="right",
        va="bottom",
        rotation=90,
        fontsize=12,
    )

    legend_x_start = 260
    legend_y_start = 0.72
    legend_y_offset = -0.065

    for i, (name, color) in enumerate(zip(names, colors)):
        xpos = legend_x_start
        ypos = legend_y_start + i * legend_y_offset
        ax.text(xpos, ypos, name, ha="left", va="center", fontsize=16, color=color)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(-100, 400)
    ax.set_ylim(-0.11, 0.79)

    return ax


def build_axes_formatted_elife_data(
    ax_ap_trace: plt.Axes,
    ax_amp_trace: plt.Axes,
    ax_ap_peaks: plt.Axes,
    ax_amp_peaks: plt.Axes,
    data: ElifeData,
    show_error: bool = False,
    se: bool = True,
):
    """Plot formatted ELife data.

    Parameters
    ----------
    ax_ap_trace : plt.Axes
        The axis to plot the AP trace on.
    ax_amp_trace : plt.Axes
        The axis to plot the AMP trace on.
    ax_ap_peaks : plt.Axes
        The axis to plot the AP peaks on.
    ax_amp_peaks : plt.Axes
        The axis to plot the AMP peaks on.
    data : ElifeData
        The ELife data to plot.
    show_error : bool, optional
        Whether to show the error bars, by default False.
    se : bool, optional
        Whether to show the standard error, by default True.
    """
    # Classify dendritic sites into proximal, distal-simple, distal-complex
    # See params docstring for explanation
    site_params = DendriticSiteParams()
    idx_proximal = (data.distance >= site_params.distance_proximal[0]) & (
        data.distance <= site_params.distance_proximal[1]
    )
    idx_distal = (data.distance >= site_params.distance_distal[0]) & (data.distance <= site_params.distance_distal[1])
    idx_simple = (data.spk[data.idx_ap] >= site_params.ap_amp_simple[0]) & (
        data.spk[data.idx_ap] <= site_params.ap_amp_simple[1]
    )
    idx_complex = (data.spk[data.idx_ap] >= site_params.ap_amp_complex[0]) & (
        data.spk[data.idx_ap] <= site_params.ap_amp_complex[1]
    )

    idx_distal_simple = idx_distal & idx_simple
    idx_distal_complex = idx_distal & idx_complex

    # Get corrections
    correction_proximal = np.sqrt(np.sum(idx_proximal)) if se else 1
    correction_distal_simple = np.sqrt(np.sum(idx_distal_simple)) if se else 1
    correction_distal_complex = np.sqrt(np.sum(idx_distal_complex)) if se else 1

    # Get mean and sem of AP traces
    mn_ap_proximal = np.mean(data.sapbase[:, idx_proximal], axis=1)
    mn_ap_distal_simple = np.mean(data.sapbase[:, idx_distal_simple], axis=1)
    mn_ap_distal_complex = np.mean(data.sapbase[:, idx_distal_complex], axis=1)
    sem_ap_proximal = np.std(data.sapbase[:, idx_proximal], axis=1) / correction_proximal
    sem_ap_distal_simple = np.std(data.sapbase[:, idx_distal_simple], axis=1) / correction_distal_simple
    sem_ap_distal_complex = np.std(data.sapbase[:, idx_distal_complex], axis=1) / correction_distal_complex

    # Get mean and sem of AMP traces
    mn_amp_proximal = np.mean(data.ampretrace[:, idx_proximal], axis=1)
    mn_amp_distal_simple = np.mean(data.ampretrace[:, idx_distal_simple], axis=1)
    mn_amp_distal_complex = np.mean(data.ampretrace[:, idx_distal_complex], axis=1)
    sem_amp_proximal = np.std(data.ampretrace[:, idx_proximal], axis=1) / correction_proximal
    sem_amp_distal_simple = np.std(data.ampretrace[:, idx_distal_simple], axis=1) / correction_distal_simple
    sem_amp_distal_complex = np.std(data.ampretrace[:, idx_distal_complex], axis=1) / correction_distal_complex

    if show_error:
        ax_ap_trace.fill_between(
            data.tvec - 100,
            mn_ap_proximal - sem_ap_proximal,
            mn_ap_proximal + sem_ap_proximal,
            color=Proximal.color,
            alpha=0.2,
        )
        ax_ap_trace.fill_between(
            data.tvec - 100,
            mn_ap_distal_simple - sem_ap_distal_simple,
            mn_ap_distal_simple + sem_ap_distal_simple,
            color=DistalSimple.color,
            alpha=0.2,
        )
        ax_ap_trace.fill_between(
            data.tvec - 100,
            mn_ap_distal_complex - sem_ap_distal_complex,
            mn_ap_distal_complex + sem_ap_distal_complex,
            color=DistalComplex.color,
            alpha=0.2,
        )
    ax_ap_trace.plot(data.tvec - 100, mn_ap_proximal, color=Proximal.color)
    ax_ap_trace.plot(data.tvec - 100, mn_ap_distal_simple, color=DistalSimple.color)
    ax_ap_trace.plot(data.tvec - 100, mn_ap_distal_complex, color=DistalComplex.color)
    ax_ap_trace.set_xlabel("Time (ms)")
    ax_ap_trace.set_xlim(-50, 400)
    ax_ap_trace.set_ylim(-0.025, 0.3)

    if show_error:
        ax_amp_trace.fill_between(
            data.tvec - 100,
            mn_amp_proximal - sem_amp_proximal,
            mn_amp_proximal + sem_amp_proximal,
            color=Proximal.color,
            alpha=0.2,
        )
        ax_amp_trace.fill_between(
            data.tvec - 100,
            mn_amp_distal_simple - sem_amp_distal_simple,
            mn_amp_distal_simple + sem_amp_distal_simple,
            color=DistalSimple.color,
            alpha=0.2,
        )
        ax_amp_trace.fill_between(
            data.tvec - 100,
            mn_amp_distal_complex - sem_amp_distal_complex,
            mn_amp_distal_complex + sem_amp_distal_complex,
            color=DistalComplex.color,
            alpha=0.2,
        )
    ax_amp_trace.plot(data.tvec - 100, mn_amp_proximal, color=Proximal.color)
    ax_amp_trace.plot(data.tvec - 100, mn_amp_distal_simple, color=DistalSimple.color)
    ax_amp_trace.plot(data.tvec - 100, mn_amp_distal_complex, color=DistalComplex.color)
    ax_amp_trace.set_xlabel("Time (ms)")
    ax_amp_trace.set_xlim(-50, 400)
    ax_amp_trace.set_ylim(-0.25, 3.1)

    nbins = 10
    s = 25
    alpha = 0.85
    beewidth = 0.25

    ax_ap_peaks.scatter(
        0 + beewidth * beeswarm(data.spk[data.idx_ap, idx_proximal], nbins=nbins),
        data.spk[data.idx_ap, idx_proximal],
        color=Proximal.color,
        s=s,
        alpha=alpha,
    )
    ax_ap_peaks.scatter(
        1 + beewidth * beeswarm(data.spk[data.idx_ap, idx_distal_simple], nbins=nbins),
        data.spk[data.idx_ap, idx_distal_simple],
        color=DistalSimple.color,
        s=s,
        alpha=alpha,
    )
    ax_ap_peaks.scatter(
        2 + beewidth * beeswarm(data.spk[data.idx_ap, idx_distal_complex], nbins=nbins),
        data.spk[data.idx_ap, idx_distal_complex],
        color=DistalComplex.color,
        s=s,
        alpha=alpha,
    )
    ax_ap_peaks.set_xlim(-0.5, 2.5)
    ax_ap_peaks.set_ylim(-0.025, 0.3)

    ax_amp_peaks.scatter(
        0 + beewidth * beeswarm(data.ampreglu[idx_proximal], nbins=nbins),
        data.ampreglu[idx_proximal],
        color=Proximal.color,
        s=s,
        alpha=alpha,
    )
    ax_amp_peaks.scatter(
        1 + beewidth * beeswarm(data.ampreglu[idx_distal_simple], nbins=nbins),
        data.ampreglu[idx_distal_simple],
        color=DistalSimple.color,
        s=s,
        alpha=alpha,
    )
    ax_amp_peaks.scatter(
        2 + beewidth * beeswarm(data.ampreglu[idx_distal_complex], nbins=nbins),
        data.ampreglu[idx_distal_complex],
        color=DistalComplex.color,
        s=s,
        alpha=alpha,
    )
    ax_amp_peaks.set_xlim(-0.5, 2.5)
    ax_amp_peaks.set_ylim(-0.25, 3.1)

    format_spines(ax_ap_trace, x_pos=-65, y_pos=-0.027, xbounds=(-25, 400), ybounds=(0.0, 0.3))
    format_spines(ax_amp_trace, x_pos=-65, y_pos=-0.27, xbounds=(-25, 400), ybounds=(0.0, 3))
    format_spines(ax_ap_peaks, x_pos=-0.6, y_pos=-0.027, xbounds=(0, 2), ybounds=(0.0, 0.3))
    format_spines(ax_amp_peaks, x_pos=-0.6, y_pos=-0.27, xbounds=(0, 2), ybounds=(0.0, 3.0))

    ax_ap_trace.set_ylabel("$\Delta Ca_{AP}$")
    ax_amp_trace.set_ylabel("$\Delta Ca_{amp} \propto \Delta Ca_{glu}$")
    ax_ap_trace.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax_amp_trace.set_yticks([0.0, 1.0, 2.0, 3.0])
    ax_ap_peaks.set_xticks([0, 1, 2])
    ax_ap_peaks.set_xticklabels([])
    ax_amp_peaks.set_xticks(
        [0, 1, 2],
        labels=[Proximal.label, DistalSimple.labelnl, DistalComplex.labelnl],
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    ax_ap_peaks.spines["left"].set_visible(False)
    ax_amp_peaks.spines["left"].set_visible(False)
    ax_ap_peaks.set_yticks([])
    ax_amp_peaks.set_yticks([])

    return ax_ap_trace, ax_amp_trace, ax_ap_peaks, ax_amp_peaks
