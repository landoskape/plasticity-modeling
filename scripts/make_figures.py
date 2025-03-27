from typing import Literal
from dataclasses import dataclass, field
import joblib
import numpy as np
import matplotlib.pyplot as plt
from src.files import get_figure_dir, data_dir
from src.plotting import FigParams, save_figure, Proximal, DistalSimple, DistalComplex, format_spines, add_group_legend
from src.schematics import Neuron
from src.experimental import (
    ElifeData,
    build_axes_formatted_elife_data,
    build_ax_amplification_demonstration_with_spines,
)
from src.conductance import build_axes_simulations, build_axes_nevian_reconstruction, build_axes_transfer_functions


@dataclass
class Figure1Params:
    schematic_text_xpos: float = 3.25
    schematic_text_ystart: float = 0
    schematic_text_ydelta: float = 1.75
    schematic_text_yextra: float = 0.75
    xlim_schematic: tuple[float, float] = (-1.6, 3.5)
    ylim_schematic: tuple[float, float] = (-1, 6)
    icell: int = 17
    start_pos: int = -15
    start_offset: int = 67
    delta_pos: int = 15
    amp_legend_x_start: float = 260
    amp_legend_y_start: float = 0.72
    amp_legend_y_offset: float = -0.07
    amp_scale_x_root: float = 200
    amp_scale_x_mag: float = 50
    amp_scale_y_root: float = 0.3
    amp_scale_y_mag: float = 0.15
    amp_scale_y_offset: float = 0.01
    amp_xlim: tuple[float, float] = (-175, 400)
    amp_ylim: tuple[float, float] = (-0.04, 0.79)
    width_ratios: list[float] = field(default_factory=lambda: [1, 1.2, 0.4])
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
    neuron = Neuron(linewidth=FigParams.linewidth)
    _ = neuron.plot(ax_schematic, origin=(0, 0), scale=1.0)

    y_proximal = fig_params.schematic_text_ystart
    y_distal_simple = y_proximal + fig_params.schematic_text_ydelta
    y_distal_complex = y_distal_simple + fig_params.schematic_text_ydelta + fig_params.schematic_text_yextra
    ax_schematic.text(
        fig_params.schematic_text_xpos,
        y_proximal,
        Proximal.label,
        color=Proximal.color,
        ha="center",
        va="center",
        fontsize=FigParams.fontsize,
    )
    ax_schematic.text(
        fig_params.schematic_text_xpos,
        y_distal_simple,
        DistalSimple.labelnl + "\n(high $\Delta Ca_{AP}$)",
        color=DistalSimple.color,
        ha="center",
        va="center",
        fontsize=FigParams.fontsize,
    )
    ax_schematic.text(
        fig_params.schematic_text_xpos,
        y_distal_complex,
        DistalComplex.labelnl + "\n(low $\Delta Ca_{AP}$)",
        color=DistalComplex.color,
        ha="center",
        va="center",
        fontsize=FigParams.fontsize,
    )

    # Set plot limits and aesthetics
    ax_schematic.set_xlim(*fig_params.xlim_schematic)
    ax_schematic.set_ylim(*fig_params.ylim_schematic)
    ax_schematic.set_aspect("equal")

    # Remove ticks and spines for cleaner visualization
    ax_schematic.set_xticks([])
    ax_schematic.set_yticks([])
    ax_schematic.spines["top"].set_visible(False)
    ax_schematic.spines["right"].set_visible(False)
    ax_schematic.spines["bottom"].set_visible(False)
    ax_schematic.spines["left"].set_visible(False)

    # Add amplification demonstration figure
    build_ax_amplification_demonstration_with_spines(
        ax_amp_demonstration,
        data,
        icell=fig_params.icell,
        start_pos=fig_params.start_pos,
        start_offset=fig_params.start_offset,
        delta_pos=fig_params.delta_pos,
        legend_x_start=fig_params.amp_legend_x_start,
        legend_y_start=fig_params.amp_legend_y_start,
        legend_y_offset=fig_params.amp_legend_y_offset,
        scale_x_root=fig_params.amp_scale_x_root,
        scale_y_root=fig_params.amp_scale_y_root,
        scale_y_mag=fig_params.amp_scale_y_mag,
        xlim=fig_params.amp_xlim,
        ylim=fig_params.amp_ylim,
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

    format_spines(
        ax_schematic,
        **FigParams.kwargs_spines(),
    )

    format_spines(
        ax_ap_trace,
        xticks=[],
        yticks=[0.0, 0.3],
        xbounds=[0, 400],
        ybounds=[0, 0.3],
        **FigParams.kwargs_spines(),
    )

    format_spines(
        ax_amp_trace,
        xticks=[0, 400],
        yticks=[0.0, 3.0],
        xbounds=[0, 400],
        ybounds=[0, 3.0],
        **FigParams.kwargs_spines(),
    )

    ax_ap_trace.spines["bottom"].set_visible(False)
    ax_ap_peaks.spines["bottom"].set_visible(False)
    ax_amp_peaks.spines["bottom"].set_visible(False)

    ax_schematic.set_facecolor("none")
    ax_amp_demonstration.set_facecolor("none")
    ax_ap_trace.set_facecolor("none")
    ax_amp_trace.set_facecolor("none")
    ax_ap_peaks.set_facecolor("none")
    ax_amp_peaks.set_facecolor("none")

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("core_figures") / "figure1"
        save_figure(fig, fig_path)

    return fig


@dataclass
class Figure2Params:
    dt: float = 0.01  # Time step for numerical integration (ms)
    t_start: float = 0  # Start time in ms
    t_end: float = 5  # End time in ms
    ap_peak_time: float = 1  # Time of AP peak in ms
    v_base: float = -70  # Base voltage in mV
    amplitude_proximal: float = 100  # AP amplitude in mV
    amplitude_distal_simple: float = 90  # AP amplitude in mV
    amplitude_distal_complex: float = 45  # AP amplitude in mV
    nevian_buffer_type: Literal["BAPTA", "EGTA", "AVERAGE"] = "AVERAGE"
    x_legend: float = 0.6
    y_legend: float = 0.9
    y_offset: float = -0.15
    ha_legend: str = "left"
    va_legend: str = "center"


def figure2(fig_params: Figure2Params, show_fig: bool = True, save_fig: bool = False):
    # Define parameters for 2 x 3 figure with 1.5 column width
    fig_width = FigParams.double_width
    fig_height = fig_width / 5
    fontsize = FigParams.fontsize

    # Build figure and axes
    fig = plt.figure(figsize=(fig_width, fig_height), **FigParams.all_fig_params())
    gs = fig.add_gridspec(1, 4)
    ax_nevian = fig.add_subplot(gs[0])
    ax_voltage = fig.add_subplot(gs[1])
    ax_nmdar = fig.add_subplot(gs[2])
    ax_vgcc = fig.add_subplot(gs[3])

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
    ax_nmdar.set_ylabel("NMDAR P(open)", fontsize=fontsize, labelpad=-10)
    ax_vgcc.set_xlabel("Time (ms)", fontsize=fontsize, labelpad=-6)
    ax_vgcc.set_ylabel("VGCC P(open)", fontsize=fontsize, labelpad=-10)

    nmdar_ylims = ax_nmdar.get_ylim()
    vgcc_ylims = ax_vgcc.get_ylim()
    min_ylim_prob = min(nmdar_ylims[0], vgcc_ylims[0])
    max_ylim_prob = np.round(max(nmdar_ylims[1], vgcc_ylims[1]) * 10) / 10 * 1.05
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
        x=fig_params.t_end * 1.0,
        y_start=max_ylim_prob * 0.96,
        y_offset=-(max_ylim_prob / 10),
        ha="right",
        va="center",
        fontsize=fontsize,
    )

    # Build nevian axes
    build_axes_nevian_reconstruction(
        ax_nevian,
        buffer=fig_params.nevian_buffer_type,
        x_legend=fig_params.x_legend,
        y_legend=fig_params.y_legend,
        y_offset=fig_params.y_offset,
        ha_legend=fig_params.ha_legend,
        va_legend=fig_params.va_legend,
    )

    ax_nevian.set_xlabel("[buffer] (mM)", fontsize=FigParams.fontsize, labelpad=-5)
    ax_nevian.set_ylabel("plasticity", fontsize=FigParams.fontsize, labelpad=-1)

    format_spines(
        ax_nevian,
        x_pos=FigParams.spine_pos,
        y_pos=FigParams.spine_pos,
        xticks=[0.0, 2.0],
        yticks=[0.0, 1.0],
        xbounds=(0.0, 2.0),
        ybounds=(0.0, 1.0),
        spine_linewidth=FigParams.linewidth,
        tick_length=FigParams.tick_length,
        tick_width=FigParams.tick_width,
        tick_fontsize=FigParams.tick_fontsize,
    )

    # Remove white backgrounds
    ax_voltage.set_facecolor("none")
    ax_nmdar.set_facecolor("none")
    ax_vgcc.set_facecolor("none")
    ax_nevian.set_facecolor("none")

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("core_figures") / "figure2"
        save_figure(fig, fig_path)

    return fig


def figure2_option2(fig_params: Figure2Params, show_fig: bool = True, save_fig: bool = False):
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
    ax_nevian = fig.add_subplot(gs[1, 0])
    ax_transfer = fig.add_subplot(gs[1, 1])
    ax_ltpltd = fig.add_subplot(gs[1, 2])

    # Get data for transfer function plots
    conductance_data = joblib.load(data_dir() / "conductance_runs.joblib")

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
    ax_nmdar.set_ylabel("NMDAR P(open)", fontsize=fontsize, labelpad=-10)
    ax_vgcc.set_xlabel("Time (ms)", fontsize=fontsize, labelpad=-6)
    ax_vgcc.set_ylabel("VGCC P(open)", fontsize=fontsize, labelpad=-10)

    nmdar_ylims = ax_nmdar.get_ylim()
    vgcc_ylims = ax_vgcc.get_ylim()
    min_ylim_prob = min(nmdar_ylims[0], vgcc_ylims[0])
    max_ylim_prob = np.round(max(nmdar_ylims[1], vgcc_ylims[1]) * 10) / 10 * 1.05
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
        x=fig_params.t_end * 1.0,
        y_start=max_ylim_prob * 0.96,
        y_offset=-(max_ylim_prob / 10),
        ha="right",
        va="center",
        fontsize=fontsize,
    )

    # Build nevian axes
    build_axes_nevian_reconstruction(
        ax_nevian,
        buffer=fig_params.nevian_buffer_type,
        x_legend=fig_params.x_legend,
        y_legend=fig_params.y_legend,
        y_offset=fig_params.y_offset,
        ha_legend=fig_params.ha_legend,
        va_legend=fig_params.va_legend,
    )

    ax_nevian.set_xlabel("[buffer] (mM)", fontsize=FigParams.fontsize, labelpad=-5)
    ax_nevian.set_ylabel("plasticity", fontsize=FigParams.fontsize, labelpad=-1)

    format_spines(
        ax_nevian,
        x_pos=FigParams.spine_pos,
        y_pos=FigParams.spine_pos,
        xticks=[0.0, 2.0],
        yticks=[0.0, 1.0],
        xbounds=(0.0, 2.0),
        ybounds=(0.0, 1.0),
        spine_linewidth=FigParams.linewidth,
        tick_length=FigParams.tick_length,
        tick_width=FigParams.tick_width,
        tick_fontsize=FigParams.tick_fontsize,
    )

    # Build transfer axes
    build_axes_transfer_functions(ax_transfer, ax_ltpltd, conductance_data, ap_amplitudes=ap_amplitudes)

    format_spines(
        ax_transfer,
        x_pos=FigParams.spine_pos,
        y_pos=FigParams.spine_pos,
        xticks=[-70, 40],
        yticks=[0.0, 1.0],
        xbounds=(-70, 40),
        ybounds=(0.0, 1.0),
        spine_linewidth=FigParams.linewidth,
        tick_length=FigParams.tick_length,
        tick_width=FigParams.tick_width,
        tick_fontsize=FigParams.tick_fontsize,
    )

    format_spines(
        ax_ltpltd,
        x_pos=FigParams.spine_pos,
        y_pos=FigParams.spine_pos,
        xticks=[0, 1],
        yticks=[0.0, 1.0],
        xlabels=["LTP", "LTD"],
        xbounds=(0.0, 1.0),
        ybounds=(0.0, 1.0),
        spine_linewidth=FigParams.linewidth,
        tick_length=FigParams.tick_length,
        tick_width=FigParams.tick_width,
        tick_fontsize=FigParams.tick_fontsize,
    )

    # Remove white backgrounds
    ax_voltage.set_facecolor("none")
    ax_nmdar.set_facecolor("none")
    ax_vgcc.set_facecolor("none")
    ax_nevian.set_facecolor("none")
    ax_transfer.set_facecolor("none")
    ax_ltpltd.set_facecolor("none")

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("core_figures") / "figure2_option2"
        save_figure(fig, fig_path)

    return fig


if __name__ == "__main__":
    # Set master parameters for showing / saving figures
    show_fig = True
    save_fig = True

    # Build Figure 1
    # fig1params = Figure1Params()
    # figure1(fig1params, show_fig=show_fig, save_fig=save_fig)

    # Build Figure 2
    fig2params = Figure2Params()
    # figure2(fig2params, show_fig=show_fig, save_fig=save_fig)
    figure2_option2(fig2params, show_fig=show_fig, save_fig=save_fig)
