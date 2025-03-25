from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from src.files import get_figure_dir
from src.plotting import FigParams, save_figure, Proximal, DistalSimple, DistalComplex, format_spines
from src.schematics import Neuron
from src.experimental import ElifeData, build_ax_amplification_demonstration, build_axes_formatted_elife_data
from src.conductance import build_axes_simulations


def add_group_legend(
    ax: plt.Axes,
    x: float,
    y_start: float,
    y_offset: float,
    y_extra: float = 0,
    ha: str = "center",
    va: str = "center",
    fontsize: float = FigParams.fontsize,
    nl_label: bool = False,
    extra_label_simple: str = "",
    extra_label_complex: str = "",
):
    y_proximal = y_start
    y_distal_simple = y_start + y_offset
    y_distal_complex = y_start + 2 * y_offset + y_extra

    label_simple = DistalSimple.labelnl if nl_label else DistalSimple.label
    label_complex = DistalComplex.labelnl if nl_label else DistalComplex.label
    label_simple += extra_label_simple
    label_complex += extra_label_complex

    ax.text(x, y_proximal, Proximal.label, color=Proximal.color, ha=ha, va=va, fontsize=fontsize)
    ax.text(x, y_distal_simple, label_simple, color=DistalSimple.color, ha=ha, va=va, fontsize=fontsize)
    ax.text(x, y_distal_complex, label_complex, color=DistalComplex.color, ha=ha, va=va, fontsize=fontsize)


@dataclass
class Figure1Params:
    icell: int = 17
    start_pos: int = 20
    delta_pos: int = 10
    width_ratios: list[float] = field(default_factory=lambda: [1, 1.2, 0.8])
    show_error: bool = True
    se: bool = True


def figure1(fig_params: Figure1Params, show_fig: bool = True, save_fig: bool = False):
    # Get ELife Data
    data = ElifeData()

    # Define parameters for 2 x 3 figure with 1.5 column width
    fig_width = FigParams.onepointfive_width
    fig_height = fig_width / 3 * 2

    # Build figure and axes
    fig = plt.figure(figsize=(fig_width, fig_height), **FigParams.all_fig_params())
    gs = fig.add_gridspec(2, 3, width_ratios=fig_params.width_ratios)
    ax_schematic = fig.add_subplot(gs[0, 0])
    ax_amp_demonstration = fig.add_subplot(gs[1, 0])
    ax_ap_trace = fig.add_subplot(gs[0, 1])
    ax_amp_trace = fig.add_subplot(gs[1, 1])
    ax_ap_peaks = fig.add_subplot(gs[0, 2])
    ax_amp_peaks = fig.add_subplot(gs[1, 2])

    # Add schematic figure
    neuron = Neuron()
    _ = neuron.plot(ax_schematic, origin=(0, 0), scale=1.0)

    # Set plot limits and aesthetics
    ax_schematic.set_xlim(-2.5, 2.5)
    ax_schematic.set_ylim(-1, 6)
    ax_schematic.set_aspect("equal")

    # Remove ticks and spines for cleaner visualization
    ax_schematic.set_xticks([])
    ax_schematic.set_yticks([])
    ax_schematic.spines["top"].set_visible(False)
    ax_schematic.spines["right"].set_visible(False)
    ax_schematic.spines["bottom"].set_visible(False)
    ax_schematic.spines["left"].set_visible(False)

    # Add amplification demonstration figure
    build_ax_amplification_demonstration(
        ax_amp_demonstration,
        data,
        icell=fig_params.icell,
        start_pos=fig_params.start_pos,
        delta_pos=fig_params.delta_pos,
    )

    # Build formatted ELife data figure
    build_axes_formatted_elife_data(
        ax_ap_trace,
        ax_amp_trace,
        ax_ap_peaks,
        ax_amp_peaks,
        data,
        show_error=fig_params.show_error,
        se=fig_params.se,
    )

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("core_figures") / "figure1"
        save_figure(fig, fig_path)

    return fig


@dataclass
class Figure2Params:
    dt = 0.01  # Time step for numerical integration (ms)
    t_start = 0  # Start time in ms
    t_end = 5  # End time in ms
    ap_peak_time = 1  # Time of AP peak in ms
    v_base = -70  # Base voltage in mV
    amplitude_proximal = 100  # AP amplitude in mV
    amplitude_distal_simple = 90  # AP amplitude in mV
    amplitude_distal_complex = 45  # AP amplitude in mV


def figure2(fig_params: Figure2Params, show_fig: bool = True, save_fig: bool = False):
    # Define parameters for 2 x 3 figure with 1.5 column width
    fig_width = FigParams.onepointfive_width
    fig_height = fig_width / 3 * 2
    fontsize = FigParams.fontsize

    # Build figure and axes
    fig = plt.figure(figsize=(fig_width, fig_height), **FigParams.all_fig_params())
    gs = fig.add_gridspec(2, 3)
    ax_voltage = fig.add_subplot(gs[0, 0])
    ax_nmdar = fig.add_subplot(gs[0, 1])
    ax_vgcc = fig.add_subplot(gs[0, 2])

    ap_amplitudes = [
        fig_params.amplitude_proximal,
        fig_params.amplitude_distal_simple,
        fig_params.amplitude_distal_complex,
    ]

    # Build axes for simulation
    build_axes_simulations(
        ax_voltage,
        ax_nmdar,
        ax_vgcc,
        t_start=fig_params.t_start,
        t_end=fig_params.t_end,
        dt=fig_params.dt,
        ap_peak_time=fig_params.ap_peak_time,
        ap_amplitudes=ap_amplitudes,
        v_base=fig_params.v_base,
        colors=[Proximal.color, DistalSimple.color, DistalComplex.color],
        labels=[Proximal.label, DistalSimple.label, DistalComplex.label],
        linewidth=FigParams.linewidth,
    )

    # Format AP response plots
    ax_voltage.set_xlabel("Time (ms)", fontsize=fontsize, labelpad=-6)
    ax_voltage.set_ylabel("AP Voltage (mV)", fontsize=fontsize, labelpad=-10)

    ax_nmdar.set_xlabel("Time (ms)", fontsize=fontsize, labelpad=-6)
    ax_nmdar.set_ylabel("P(open) (NMDAR)", fontsize=fontsize, labelpad=-10)

    ax_vgcc.set_xlabel("Time (ms)", fontsize=fontsize, labelpad=-6)
    ax_vgcc.set_ylabel("P(open) (VGCC)", fontsize=fontsize, labelpad=-10)

    nmdar_ylims = ax_nmdar.get_ylim()
    vgcc_ylims = ax_vgcc.get_ylim()
    min_ylim_prob = min(nmdar_ylims[0], vgcc_ylims[0])
    max_ylim_prob = np.round(max(nmdar_ylims[1], vgcc_ylims[1]) * 10) / 10
    ax_nmdar.set_ylim(min_ylim_prob, max_ylim_prob)
    ax_vgcc.set_ylim(min_ylim_prob, max_ylim_prob)

    format_spines(
        ax_voltage,
        x_pos=FigParams.spine_pos,
        y_pos=FigParams.spine_pos,
        xticks=[fig_params.t_start, fig_params.t_end],
        yticks=[fig_params.v_base, fig_params.v_base + max(ap_amplitudes)],
        xbounds=(fig_params.t_start, fig_params.t_end),
        ybounds=(fig_params.v_base, fig_params.v_base + max(ap_amplitudes)),
        spine_linewidth=FigParams.linewidth,
        tick_length=FigParams.tick_length,
        tick_width=FigParams.tick_width,
        tick_fontsize=FigParams.tick_fontsize,
    )

    format_spines(
        ax_nmdar,
        x_pos=FigParams.spine_pos,
        y_pos=FigParams.spine_pos,
        xticks=[fig_params.t_start, fig_params.t_end],
        yticks=[0, max_ylim_prob],
        xbounds=(fig_params.t_start, fig_params.t_end),
        ybounds=(0, max_ylim_prob),
        spine_linewidth=FigParams.linewidth,
        tick_length=FigParams.tick_length,
        tick_width=FigParams.tick_width,
        tick_fontsize=FigParams.tick_fontsize,
    )

    format_spines(
        ax_vgcc,
        x_pos=FigParams.spine_pos,
        y_pos=FigParams.spine_pos,
        xticks=[fig_params.t_start, fig_params.t_end],
        yticks=[0, max_ylim_prob],
        xbounds=(fig_params.t_start, fig_params.t_end),
        ybounds=(0, max_ylim_prob),
        spine_linewidth=FigParams.linewidth,
        tick_length=FigParams.tick_length,
        tick_width=FigParams.tick_width,
        tick_fontsize=FigParams.tick_fontsize,
    )

    add_group_legend(
        ax_vgcc,
        x=fig_params.t_end * 0.98,
        y_start=max_ylim_prob * 0.96,
        y_offset=-(max_ylim_prob / 10),
        ha="right",
        va="center",
        fontsize=fontsize,
    )

    ax_voltage.set_facecolor("none")
    ax_nmdar.set_facecolor("none")
    ax_vgcc.set_facecolor("none")

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("core_figures") / "figure2"
        save_figure(fig, fig_path)

    return fig


if __name__ == "__main__":
    # Set master parameters for showing / saving figures
    show_fig = True
    save_fig = True

    # Build Figure 1
    fig1params = Figure1Params()
    # figure1(fig1params, show_fig=show_fig, save_fig=save_fig)

    # Build Figure 2
    fig2params = Figure2Params()
    figure2(fig2params, show_fig=show_fig, save_fig=save_fig)
