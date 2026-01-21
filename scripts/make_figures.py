from __future__ import annotations
from typing import Literal
from dataclasses import dataclass, field
import joblib
import numpy as np
import matplotlib.pyplot as plt
from src.files import get_figure_dir, data_dir, results_dir
from src.plotting import (
    FigParams,
    Proximal,
    DistalSimple,
    DistalComplex,
    save_figure,
    format_spines,
    add_group_legend,
    add_dpratio_legend,
)
from src.schematics import Neuron, build_integrated_schematic_axis, create_dpratio_colors
from src.experimental import (
    ElifeData,
    build_axes_formatted_elife_data,
    build_ax_amplification_demonstration_with_spines,
)
from src.conductance import (
    build_axes_simulations,
    build_axes_nevian_reconstruction,
    build_axes_transfer_functions,
    NMDAR,
    VGCC,
)
from src.iaf.plotting import (
    build_ax_latent_correlation_demonstration,
    build_ax_corrcoef,
    build_ax_trajectory,
    build_ax_weight_summary,
    build_ax_weight_fits,
    build_ax_sigmoid_example,
    build_plasticity_rule_axes,
    build_environment_compartment_mapping_ax,
    build_receptive_field_ax,
    build_tuning_representation_ax,
    build_stimulus_trajectory_ax,
    build_orientation_confusion_axes,
    build_weights_ax,
    build_tuning_type_axes,
    build_visual_tuning_summary_ax,
    build_tuning_group_trajectory_axes,
    build_relative_edge_weights_axes,
)
from src.iaf.analysis import (
    gather_metadata,
    gather_results,
    gather_rates,
    gather_weights,
    gather_num_connections,
    get_groupnames,
    sort_orientation_preference,
    summarize_weights,
)


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
    _ = neuron.plot(ax_schematic, origin=(0, 0))

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
    # Define parameters for 2 x 4 figure with 2 column width
    fig_width = FigParams.double_width
    fig_height = fig_width / 2
    fontsize = FigParams.fontsize

    # Build figure and axes
    fig = plt.figure(figsize=(fig_width, fig_height), **FigParams.all_fig_params())
    gs = fig.add_gridspec(2, 4)
    ax_voltage = fig.add_subplot(gs[0, 0])
    ax_nmdar = fig.add_subplot(gs[0, 1])
    ax_vgcc = fig.add_subplot(gs[0, 2])
    ax_integrated = fig.add_subplot(gs[0, 3])
    ax_nevian = fig.add_subplot(gs[1, 1])
    ax_transfer = fig.add_subplot(gs[1, 2])
    ax_ltpltd = fig.add_subplot(gs[1, 3])

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
        ax_integrated,
        t_start=fig_params.t_start,
        t_end=fig_params.t_end,
        dt=fig_params.dt,
        ap_peak_time=fig_params.ap_peak_time,
        ap_amplitudes=ap_amplitudes,
        v_base=fig_params.v_base,
        ca_in=conductance_data["ca_in"],
        ca_out=conductance_data["ca_out"],
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

    nmdar_ylims = (0, 1)
    vgcc_ylims = (0, 1)
    ax_nmdar.set_ylim(nmdar_ylims)
    ax_vgcc.set_ylim(vgcc_ylims)

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
        yticks=nmdar_ylims,
        xbounds=(fig_params.t_start, fig_params.t_end),
        ybounds=nmdar_ylims,
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
        yticks=vgcc_ylims,
        xbounds=(fig_params.t_start, fig_params.t_end),
        ybounds=vgcc_ylims,
        spine_linewidth=FigParams.linewidth,
        tick_length=FigParams.tick_length,
        tick_width=FigParams.tick_width,
        tick_fontsize=FigParams.tick_fontsize,
    )

    add_group_legend(
        ax_vgcc,
        x=fig_params.t_end * 1.0,
        y_start=0.96,
        y_offset=-(1 / 10),
        ha="right",
        va="center",
        fontsize=fontsize,
    )

    format_spines(
        ax_integrated,
        x_pos=FigParams.spine_pos,
        y_pos=FigParams.spine_pos,
        xticks=[0, 1, 2],
        yticks=[0.0, 1.0],
        xbounds=(0.0, 2.0),
        ybounds=(0.0, 1.0),
        spine_linewidth=FigParams.linewidth,
        tick_length=FigParams.tick_length,
        tick_width=FigParams.tick_width,
        tick_fontsize=FigParams.tick_fontsize,
    )
    ax_integrated.set_xticks([0, 1, 2])
    ax_integrated.set_ylabel("[Ca] Influx")

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
    build_axes_transfer_functions(
        ax_transfer,
        ax_ltpltd,
        conductance_data,
        ap_amplitudes=ap_amplitudes,
    )

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

    # labels=[Proximal.label, DistalSimple.labelnl, DistalComplex.labelnl],
    format_spines(
        ax_ltpltd,
        x_pos=FigParams.spine_pos,
        y_pos=FigParams.spine_pos,
        xticks=[0, 1, 2],
        yticks=[0.0, 1.0],
        xbounds=(0.0, 2.0),
        ybounds=(0.0, 1.0),
        spine_linewidth=FigParams.linewidth,
        tick_length=FigParams.tick_length,
        tick_width=FigParams.tick_width,
        tick_fontsize=FigParams.tick_fontsize,
    )
    ax_ltpltd.set_xticks(
        [0, 1, 2],
        labels=[Proximal.label, DistalSimple.labelnl, DistalComplex.labelnl],
        ha="right",
        rotation=45,
        rotation_mode="anchor",
        fontsize=FigParams.fontsize,
    )

    ax_transfer.set_xlabel("AP Peak (mV)", fontsize=FigParams.fontsize, labelpad=-5)
    ax_transfer.set_ylabel("plasticity", fontsize=FigParams.fontsize, labelpad=-1)
    ax_ltpltd.set_ylabel("plasticity", fontsize=FigParams.fontsize, labelpad=-1)

    # Remove white backgrounds
    ax_voltage.set_facecolor("none")
    ax_nmdar.set_facecolor("none")
    ax_vgcc.set_facecolor("none")
    ax_integrated.set_facecolor("none")
    ax_nevian.set_facecolor("none")
    ax_transfer.set_facecolor("none")
    ax_ltpltd.set_facecolor("none")

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("core_figures") / "figure2"
        save_figure(fig, fig_path)

    return fig


@dataclass
class Figure3Params:
    main_neuron_label_style: str = "label"
    main_neuron_label_offset: float = 0.27
    main_neuron_label_fontsize: float = FigParams.fontsize
    num_neurons: int = 5
    neurons_xoffset: float = 1
    neurons_yoffset: float = 0
    neurons_xshift: float = 0.6
    neurons_yshift: float = 0.85
    small_neuron_linewidth: float = FigParams.thinlinewidth
    small_neuron_soma_size: float = 0.5
    small_neuron_trunk_height: float = 0.75
    small_neuron_tuft_height: float = 0.75
    dp_max_ratio: float = 0.4
    dp_width: float = 2
    dp_pinch: float = 0.175
    dp_xoffset: float = -0.3
    dp_linewidth: float = FigParams.thicklinewidth * 2
    dp_markersize: float = 25
    dp_xlabel_yoffset: float = -0.15
    dp_ytitle_yoffset: float = 0.29
    dp_yschema_shift: float = 2.6
    dp_color_inset_position: tuple[float] = (0.535, 0.05, 0.0875, 0.4)
    dp_color_label_padding: float = 1
    dp_label_fontsize: float = FigParams.fontsize
    dp_title_fontsize: float = FigParams.fontsize
    xrange_buffer: float = 0.05
    yrange_buffer: float = 0.05
    plasticity_time_constant: float = 20
    plasticity_max_delay: float = 100
    plasticity_depression_ratio: float = 1.1
    plasticity_homeostasis_max_ratio: float = 5
    plasticity_ltp_color: str = NMDAR.color()
    plasticity_ltd_color: str = VGCC.color()
    plasticity_homeostasis_color: str = "k"
    plasticity_markersize: float = 7


def figure3(fig_params: Figure3Params, show_fig: bool = True, save_fig: bool = True):
    # Define parameters for 2 x 3 figure with 1.5 column width
    fig_width = FigParams.onepointfive_width
    fig_height = fig_width / 3 * 2 / 1.5 * 6 / 5
    fontsize = FigParams.fontsize

    # Build figure and axes
    fig = plt.figure(figsize=(fig_width, fig_height), **FigParams.all_fig_params())
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5])
    gs_plasticity = gs[0].subgridspec(4, 1, height_ratios=[0.1, 1, 0.175, 1])
    gs_schematic = gs[1].subgridspec(2, 1, height_ratios=[5, 1])
    ax_stdp = fig.add_subplot(gs_plasticity[1])
    ax_homeostasis = fig.add_subplot(gs_plasticity[3])
    ax_schematic = fig.add_subplot(gs_schematic[0])
    ax_table = fig.add_subplot(gs_schematic[1])

    build_integrated_schematic_axis(
        ax_schematic,
        ax_table,
        main_neuron_label_style=fig_params.main_neuron_label_style,
        main_neuron_label_offset=fig_params.main_neuron_label_offset,
        main_neuron_label_fontsize=fig_params.main_neuron_label_fontsize,
        num_neurons=fig_params.num_neurons,
        neurons_xoffset=fig_params.neurons_xoffset,
        neurons_yoffset=fig_params.neurons_yoffset,
        neurons_xshift=fig_params.neurons_xshift,
        neurons_yshift=fig_params.neurons_yshift,
        small_neuron_linewidth=fig_params.small_neuron_linewidth,
        small_neuron_soma_size=fig_params.small_neuron_soma_size,
        small_neuron_trunk_height=fig_params.small_neuron_trunk_height,
        small_neuron_tuft_height=fig_params.small_neuron_tuft_height,
        dp_max_ratio=fig_params.dp_max_ratio,
        dp_width=fig_params.dp_width,
        dp_pinch=fig_params.dp_pinch,
        dp_xoffset=fig_params.dp_xoffset,
        dp_markersize=fig_params.dp_markersize,
        dp_linewidth=fig_params.dp_linewidth,
        dp_xlabel_yoffset=fig_params.dp_xlabel_yoffset,
        dp_ytitle_yoffset=fig_params.dp_ytitle_yoffset,
        dp_yschema_shift=fig_params.dp_yschema_shift,
        dp_color_inset_position=fig_params.dp_color_inset_position,
        dp_color_label_padding=fig_params.dp_color_label_padding,
        dp_label_fontsize=fig_params.dp_label_fontsize,
        dp_title_fontsize=fig_params.dp_title_fontsize,
        xrange_buffer=fig_params.xrange_buffer,
        yrange_buffer=fig_params.yrange_buffer,
    )

    build_plasticity_rule_axes(
        ax_stdp,
        ax_homeostasis,
        time_constant=fig_params.plasticity_time_constant,
        max_delay=fig_params.plasticity_max_delay,
        depression_ratio=fig_params.plasticity_depression_ratio,
        homeostasis_max_ratio=fig_params.plasticity_homeostasis_max_ratio,
        ltp_color=fig_params.plasticity_ltp_color,
        ltd_color=fig_params.plasticity_ltd_color,
        homeostasis_color=fig_params.plasticity_homeostasis_color,
        markersize=fig_params.plasticity_markersize,
        fontsize=fontsize,
    )

    xlims = ax_stdp.get_xlim()
    ylims = ax_stdp.get_ylim()
    format_spines(
        ax_stdp,
        x_pos=0.5,
        y_pos=0.5,
        xticks=[-xlims[1], xlims[1]],
        yticks=[-1, 1],
        ylabels=[],
        xbounds=(-xlims[1], xlims[1]),
        ybounds=(-ylims[1], ylims[1]),
        spine_linewidth=FigParams.linewidth,
        tick_length=FigParams.tick_length * 2.25,
        tick_width=FigParams.tick_width,
        tick_direction="inout",
        tick_fontsize=FigParams.tick_fontsize,
    )

    xlims = ax_homeostasis.get_xlim()
    ylims = ax_homeostasis.get_ylim()
    format_spines(
        ax_homeostasis,
        x_pos=0.5,
        y_pos=0.5,
        xticks=[-xlims[1], xlims[1]],
        yticks=[],
        xbounds=(-xlims[1], xlims[1]),
        ybounds=ylims,
        spine_linewidth=FigParams.linewidth,
        tick_length=FigParams.tick_length,
        tick_width=FigParams.tick_width,
        tick_direction="inout",
        tick_fontsize=FigParams.tick_fontsize,
    )

    # Maybe an example of inputs and spiking?

    # Remove white backgrounds
    ax_stdp.set_facecolor("none")
    ax_homeostasis.set_facecolor("none")
    ax_schematic.set_facecolor("none")
    ax_table.set_facecolor("none")

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("core_figures") / "figure3"
        save_figure(fig, fig_path)

    return fig


@dataclass
class Figure4Params:
    example_simulations: str = "20250324"
    full_simulations: str = "20250509"
    schematic_T: int = 300
    schematic_N: int = 6
    schematic_sigma_latent: int = 15
    schematic_sigma_noise: int = 10
    schematic_max_correlation: float = 0.9
    schematic_y_offsets: float = -1.0
    schematic_y_latent: float = 1.0
    schematic_colormap: str = "Reds"
    schematic_latent_color: str = "black"
    schematic_text_offset_fraction: float = 0.025
    corrcoef_source_name: str = "excitatory"
    example_idpratio: int = 2
    example_irepeat: int = 0
    example_ineuron: int = 0
    summary_idpratio: int = 2
    summary_irepeat: int = 6
    summary_ineuron: int = 0
    example_cmap: str = "gray_r"
    example_include_psth: bool = False
    feature_colormap: str = "Reds"
    feature_label_max_value: float = 0.8
    feature_label_min_value: float = 0.1
    feature_offset_fraction: float = 0.02
    feature_width_fraction: float = 0.025
    color_bar: bool = True
    color_bar_offset_fraction: float = 0.02
    color_bar_width_fraction: float = 0.04
    summary_average_method: str = "fraction"
    summary_average_window: float = 0.2
    summary_cmap: str = "plasma_r"
    summary_cmap_pinch: float = 0.25
    summary_legend_xpos: float = 0.96
    summary_legend_ypos: float = 0.12
    fits_beeswarm_width: float = 0.3
    fits_share_ylim: bool = True


def figure4(fig_params: Figure4Params, show_fig: bool = True, save_fig: bool = False):
    # Define parameters for 2 x 3 figure with 1.5 column width
    fig_width = FigParams.double_width
    fig_height = fig_width / 4 * 2
    fontsize = FigParams.fontsize

    # Build feature colormap
    feature_cmap = plt.get_cmap(fig_params.feature_colormap)
    feature_colors = feature_cmap(
        np.linspace(fig_params.feature_label_max_value, fig_params.feature_label_min_value, 40),
    )

    # Build figure and axes
    fig = plt.figure(figsize=(fig_width, fig_height), **FigParams.all_fig_params())
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 3])
    gs_inputs = gs[0].subgridspec(2, 1)
    gs_corrcoef = gs_inputs[1].subgridspec(2, 2, width_ratios=[0.05, 1], height_ratios=[0.05, 1])
    ax_schematic = fig.add_subplot(gs_inputs[0])
    ax_cc_left = fig.add_subplot(gs_corrcoef[1, 0])
    ax_cc_top = fig.add_subplot(gs_corrcoef[0, 1])
    ax_corrcoef = fig.add_subplot(gs_corrcoef[1, 1], sharex=ax_cc_top, sharey=ax_cc_left)

    gs_results = gs[1].subgridspec(3, 3, width_ratios=[1, 1, 0.75])
    ax_basal = fig.add_subplot(gs_results[0, 0])
    ax_dsimple = fig.add_subplot(gs_results[1, 0])
    ax_dcomplex = fig.add_subplot(gs_results[2, 0])
    ax_summary_basal = fig.add_subplot(gs_results[0, 1])
    ax_summary_dsimple = fig.add_subplot(gs_results[1, 1])
    ax_summary_dcomplex = fig.add_subplot(gs_results[2, 1])
    ax_fits_basal = fig.add_subplot(gs_results[0, 2])
    ax_fits_dsimple = fig.add_subplot(gs_results[1, 2])
    ax_fits_dcomplex = fig.add_subplot(gs_results[2, 2])
    ax_inset_sigmoid = ax_fits_dsimple.inset_axes((0.4, 0.1, 0.55, 0.3))
    ax_inset_features = ax_fits_dsimple.inset_axes((0.4, 0.05, 0.55, 0.03))

    ax_schematic = build_ax_latent_correlation_demonstration(
        ax_schematic,
        T=fig_params.schematic_T,
        N=fig_params.schematic_N,
        sigma_latent=fig_params.schematic_sigma_latent,
        sigma_noise=fig_params.schematic_sigma_noise,
        max_correlation=fig_params.schematic_max_correlation,
        y_latent=fig_params.schematic_y_latent,
        y_offsets=fig_params.schematic_y_offsets,
        colormap=fig_params.schematic_colormap,
        latent_color=fig_params.schematic_latent_color,
        text_offset_fraction=fig_params.schematic_text_offset_fraction,
    )

    # Just get rid of the entire axis
    ax_schematic.set_xticks([])
    ax_schematic.set_yticks([])
    for spine in ax_schematic.spines.values():
        spine.set_visible(False)

    # Get example results
    example_folder = results_dir("iaf_runs") / "correlated" / fig_params.example_simulations
    example_metadata = gather_metadata(example_folder, experiment_type="correlation")
    example_results = gather_results(example_metadata)
    max_correlation = example_metadata["base_config"].sources["excitatory"].max_correlation

    # Get results for summary data (and trajectory example)
    summary_folder = results_dir("iaf_runs") / "correlated" / fig_params.full_simulations
    summary_metadata = gather_metadata(summary_folder, experiment_type="correlation")
    summary_firing_rates = gather_rates(summary_metadata, experiment_type="correlation")
    summary_num_connections = gather_num_connections(summary_metadata, experiment_type="correlation")
    summary_weights = gather_weights(
        summary_metadata,
        experiment_type="correlation",
        average_method=fig_params.summary_average_method,
        average_window=fig_params.summary_average_window,
        norm_by_max_weight=True,
        norm_by_num_synapses=True,
        num_connections=summary_num_connections,
    )
    summary_results = gather_results(summary_metadata)

    idx_to_example_ratio = np.array(example_metadata["ratios"]) == fig_params.example_idpratio
    idx_to_example_repeat = np.array(example_metadata["repeats"]) == fig_params.example_irepeat
    idx_to_example = np.where(idx_to_example_ratio & idx_to_example_repeat)[0]
    if len(idx_to_example) != 1:
        raise ValueError(
            f"No example found for idpratio {fig_params.example_idpratio} and irepeat {fig_params.example_irepeat}"
        )
    idx_to_example = idx_to_example[0]
    num_ratios = len(example_metadata["dp_ratios"])
    colors = create_dpratio_colors(
        num_ratios,
        cmap=fig_params.summary_cmap,
        cmap_pinch=fig_params.summary_cmap_pinch,
    )[0]

    idx_to_summary_example_ratio = np.array(summary_metadata["ratios"]) == fig_params.summary_idpratio
    idx_to_summary_example_repeat = np.array(summary_metadata["repeats"]) == fig_params.summary_irepeat
    idx_to_summary_example = np.where(idx_to_summary_example_ratio & idx_to_summary_example_repeat)[0]
    if len(idx_to_summary_example) != 1:
        raise ValueError(
            f"No example found for idpratio {fig_params.summary_idpratio} and irepeat {fig_params.summary_irepeat}"
        )
    idx_to_summary_example = idx_to_summary_example[0]
    num_ratios = len(summary_metadata["dp_ratios"])
    colors = create_dpratio_colors(
        num_ratios,
        cmap=fig_params.summary_cmap,
        cmap_pinch=fig_params.summary_cmap_pinch,
    )[0]
    summary_example_color = colors[fig_params.summary_idpratio]

    _ = build_ax_corrcoef(
        ax_corrcoef,
        ax_cc_left,
        ax_cc_top,
        example_results[idx_to_example],
        feature_colors,
        source_name=fig_params.corrcoef_source_name,
        rho_max=max_correlation,
    )
    for ax in (ax_cc_left, ax_cc_top, ax_corrcoef):
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plot_firing_rates = summary_firing_rates[
        fig_params.summary_idpratio, fig_params.summary_irepeat, fig_params.summary_ineuron
    ]
    _ = build_ax_trajectory(
        None,
        ax_basal,
        ax_dsimple,
        ax_dcomplex,
        summary_results[idx_to_summary_example],
        plot_firing_rates,
        ineuron=fig_params.summary_ineuron,
        feature_colors=feature_colors,
        cmap=fig_params.example_cmap,
        feature_offset_fraction=fig_params.feature_offset_fraction,
        feature_width_fraction=fig_params.feature_width_fraction,
        color_bar=fig_params.color_bar,
        color_bar_offset_fraction=fig_params.color_bar_offset_fraction,
        color_bar_width_fraction=fig_params.color_bar_width_fraction,
    )

    xoffset = -fig_params.feature_offset_fraction - fig_params.feature_width_fraction
    num_seconds = summary_firing_rates.shape[-1]
    xticks = [0, 1]
    xlabels = [0, num_seconds]
    for iax, (ax, wg) in enumerate(
        zip((ax_basal, ax_dsimple, ax_dcomplex, None), [Proximal, DistalSimple, DistalComplex, None]),
    ):
        if ax is None:
            continue
        requires_xaxis = iax == 3 or (iax == 2 and not fig_params.example_include_psth)
        ax.set_xlim(
            xoffset,
            1 + fig_params.color_bar * (fig_params.color_bar_width_fraction + fig_params.color_bar_offset_fraction),
        )
        if iax == 3:
            ax.set_ylim(0, np.max(plot_firing_rates) * 1.02)
            ax.set_ylabel("Firing\nRate", fontsize=fontsize, labelpad=-9)
            yticks = [0, np.max(plot_firing_rates)]
            ybounds = (0, np.max(plot_firing_rates))
        else:
            ax.set_ylabel(wg.labelnl, fontsize=fontsize, labelpad=0, color=wg.color)
            yticks = []
            ybounds = ax.get_ylim()
        format_spines(
            ax,
            x_pos=FigParams.spine_pos,
            y_pos=FigParams.spine_pos,
            xticks=xticks if requires_xaxis else [],
            yticks=yticks,
            xbounds=[0, 1],
            ybounds=ybounds,
            spine_linewidth=FigParams.linewidth,
            tick_length=FigParams.tick_length,
            tick_width=FigParams.tick_width,
            tick_fontsize=FigParams.tick_fontsize,
        )
        if iax != 3:
            ax.spines["left"].set_visible(False)
        if requires_xaxis:
            ax.set_xlabel("Time (s)", fontsize=fontsize, labelpad=-6)
            ax.set_xticks(xticks, labels=xlabels)
        else:
            ax.spines["bottom"].set_visible(False)

    build_ax_weight_summary(
        ax_summary_basal,
        ax_summary_dsimple,
        ax_summary_dcomplex,
        summary_metadata,
        summary_weights,
        cmap=fig_params.summary_cmap,
        cmap_pinch=fig_params.summary_cmap_pinch,
    )
    yticks = [0, max_correlation]
    for iax, ax in enumerate([ax_summary_basal, ax_summary_dsimple, ax_summary_dcomplex]):
        requires_xaxis = iax == 2
        ax.set_xlim(0, 1)
        format_spines(
            ax,
            x_pos=FigParams.spine_pos,
            y_pos=FigParams.spine_pos,
            xticks=[0, 1] if requires_xaxis else [],
            yticks=yticks,
            xbounds=[0, 1],
            ybounds=yticks,
            spine_linewidth=FigParams.linewidth,
            tick_length=FigParams.tick_length,
            tick_width=FigParams.tick_width,
            tick_fontsize=FigParams.tick_fontsize,
        )
        ax.set_ylabel("Input\nCorr." + r" ($\rho$)", fontsize=fontsize, labelpad=-9)
        if requires_xaxis:
            ax.set_xlabel("Relative Weight", fontsize=fontsize, labelpad=-6)
        else:
            ax.spines["bottom"].set_visible(False)

    build_ax_weight_fits(
        ax_fits_basal,
        ax_fits_dsimple,
        ax_fits_dcomplex,
        summary_metadata,
        summary_weights,
        cmap=fig_params.summary_cmap,
        cmap_pinch=fig_params.summary_cmap_pinch,
        beeswarm_width=fig_params.fits_beeswarm_width,
        share_ylim=fig_params.fits_share_ylim,
    )
    ratios = summary_metadata["dp_ratios"]
    num_ratios = len(ratios)
    xticks = range(num_ratios)
    xlabels = [f"{(ratio-1.0)*100:1g}" for ratio in ratios]
    yticks = [-1, 0, 0.4]
    for iax, ax in enumerate([ax_fits_basal, ax_fits_dsimple, ax_fits_dcomplex]):
        requires_xaxis = iax == 2
        ax.set_xlim(-0.5, num_ratios - 0.5)
        format_spines(
            ax,
            x_pos=FigParams.spine_pos,
            y_pos=FigParams.spine_pos,
            xticks=xticks if requires_xaxis else [],
            yticks=yticks,
            xbounds=[0, num_ratios - 1],
            ybounds=ax.get_ylim(),
            spine_linewidth=FigParams.linewidth,
            tick_length=FigParams.tick_length,
            tick_width=FigParams.tick_width,
            tick_fontsize=FigParams.tick_fontsize,
        )
        ax.set_ylabel(r"$\sigma$ half-point ($\rho$)", fontsize=fontsize, labelpad=0)
        if requires_xaxis:
            ax.set_xlabel("Extra Depression (%)", fontsize=fontsize, labelpad=0)
            ax.set_xticklabels(xlabels)
        else:
            ax.spines["bottom"].set_visible(False)

    add_dpratio_legend(
        ax_summary_basal,
        ratios,
        x=fig_params.summary_legend_xpos,
        y=fig_params.summary_legend_ypos,
        fontsize=FigParams.fontsize,
        cmap=fig_params.summary_cmap,
        cmap_pinch=fig_params.summary_cmap_pinch,
    )

    xlim_sum = ax_summary_basal.get_xlim()
    ylim_sum = ax_summary_basal.get_ylim()
    xratio = (fig_params.summary_legend_xpos - xlim_sum[0]) / (xlim_sum[1] - xlim_sum[0])
    yratio = (fig_params.summary_legend_ypos - ylim_sum[0]) / (ylim_sum[1] - ylim_sum[0])
    xlim = ax_fits_basal.get_xlim()
    ylim = ax_fits_basal.get_ylim()
    xpos = xlim[0] + xratio * (xlim[1] - xlim[0])
    ypos = ylim[0] + yratio * (ylim[1] - ylim[0])
    add_dpratio_legend(
        ax_fits_basal,
        ratios,
        x=xpos,
        y=ypos,
        fontsize=FigParams.fontsize,
        cmap=fig_params.summary_cmap,
        cmap_pinch=fig_params.summary_cmap_pinch,
    )

    build_ax_sigmoid_example(
        ax_inset_sigmoid,
        summary_results[idx_to_summary_example],
        ineuron=fig_params.summary_ineuron,
        color=summary_example_color,
    )
    ax_inset_sigmoid.set_xlim(0, 0.4)
    format_spines(
        ax_inset_sigmoid,
        x_pos=FigParams.spine_pos,
        y_pos=FigParams.spine_pos,
        xbounds=(0, 0.4),
        ybounds=(0, 1),
        xticks=[],
        yticks=[0, 1],
        ylabels=["0", "1"],
        spine_linewidth=FigParams.thinlinewidth,
        tick_length=FigParams.tick_length / 2,
        tick_width=FigParams.tick_width / 2,
        tick_fontsize=FigParams.tick_fontsize * 0.7,
    )
    ax_inset_sigmoid.spines["bottom"].set_visible(False)
    ax_inset_features.imshow(feature_colors[::-1][None], aspect="auto", interpolation="none", extent=[0, 0.4, 0, 1])
    ax_inset_features.set_xlim(0, 0.4)
    ax_inset_features.set_xticks([])
    ax_inset_features.set_yticks([])
    for spine in ax_inset_features.spines.values():
        spine.set_visible(False)

    # Remove white backgrounds
    ax_schematic.set_facecolor("none")
    ax_corrcoef.set_facecolor("none")
    ax_cc_left.set_facecolor("none")
    ax_cc_top.set_facecolor("none")
    ax_basal.set_facecolor("none")
    ax_dsimple.set_facecolor("none")
    ax_dcomplex.set_facecolor("none")
    ax_summary_basal.set_facecolor("none")
    ax_summary_dsimple.set_facecolor("none")
    ax_summary_dcomplex.set_facecolor("none")
    ax_fits_basal.set_facecolor("none")
    ax_fits_dsimple.set_facecolor("none")
    ax_fits_dcomplex.set_facecolor("none")
    ax_inset_sigmoid.set_facecolor("none")
    ax_inset_features.set_facecolor("none")

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("core_figures") / "figure4"
        save_figure(fig, fig_path)

    return fig


@dataclass
class Figure5Params:
    full_config: str = "hofer"
    full_simulations: str = "20250320"
    mapping_simple_tuft_inset_xoffset: float = 0.14
    mapping_complex_tuft_inset_xoffset: float = -0.14
    mapping_tuft_yoffset: float = -0.25
    field_width: float = 0.95
    field_scale: float = 1.25
    field_inset_yoffset_fraction: float = 0.08
    input_inset_yoffset_extra: float = 0.0
    vonmises_concentration: float = 1.0
    baseline_rate: float = 5.0
    driven_rate: float = 45.0
    gabor_width: float = 0.6
    gabor_envelope: float = 0.4
    gabor_gamma: float = 1.5
    gabor_halfsize: float = 25
    gabor_phase: float = 0
    gabor_highlight_magnitude: float = 1  # 4 if not using edge_ticks!!!
    gabor_vmax_scale: float = 1.5
    stimulus_stims_per_row: int = 5
    stimulus_num_edges: int = 7
    stimulus_hspacing: float = 0.15
    stimulus_vspacing: float = 0.05
    stimulus_arrow_width: float = 0.75
    stimulus_arrow_mutation: float = 6
    stimulus_use_edge_ticks: bool = True
    stimulus_edge_tick_fraction: float = 0.55
    stimulus_edge_tick_color: str = "k"
    stimulus_edge_tick_lw: float = 0.8
    stimulus_edge_tick_alpha: float = 0.8
    x_offset_input_label: float = -0.05
    x_offset_rate_label: float = -0.9
    x_offset_field_label: float = -0.55
    ylabel_fontsize: float = FigParams.smallfontsize
    include_arrows: bool = True
    tuning_hspacing: float = 6
    tuning_vspacing: float = 21
    tuning_fontsize_label: float = FigParams.smallfontsize
    tuning_fontsize_title: float = FigParams.smallfontsize
    confusion_fontsize: float = FigParams.smallfontsize
    confusion_tickfontsize: float = FigParams.tinyfontsize
    example_dpratio: tuple[int, int] = (0, 4)
    example_edge: tuple[int, int] = (1, 1)
    example_simulation: tuple[int, int] = (0, 0)
    example_neuron: tuple[int, int] = (0, 0)


def figure5(fig_params: Figure5Params, show_fig: bool = True, save_fig: bool = False):
    fig_width = FigParams.double_width * 3 / 4
    fig_height = fig_width / 3 * 1.5

    if fig_params.full_config == "hofer_replacement":
        norm_by_max_weight = True
        norm_by_num_synapses = False
        norm_by_total_synapses = True
    else:
        norm_by_max_weight = True
        norm_by_num_synapses = True
        norm_by_total_synapses = False

    fig = plt.figure(figsize=(fig_width, fig_height), **FigParams.all_fig_params())
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    gs_stim = gs[0].subgridspec(2, 1, height_ratios=[1, 2.75])
    ax_trajectory = fig.add_subplot(gs_stim[0])
    ax_receptive_field = fig.add_subplot(gs_stim[1])
    gs_inputs = gs[1].subgridspec(2, 1, height_ratios=[2.75, 1])
    ax_tuning = fig.add_subplot(gs_inputs[1])
    ax_inputs = fig.add_subplot(gs_inputs[0])
    gs_confusion = gs[2].subgridspec(2, 1)
    ax_distal_simple = fig.add_subplot(gs_confusion[0])
    ax_distal_complex = fig.add_subplot(gs_confusion[1])

    build_environment_compartment_mapping_ax(
        ax_inputs,
        simple_tuft_inset_xoffset=fig_params.mapping_simple_tuft_inset_xoffset,
        complex_tuft_inset_xoffset=fig_params.mapping_complex_tuft_inset_xoffset,
        tuft_yoffset=fig_params.mapping_tuft_yoffset,
        gabor_highlight_magnitude=fig_params.gabor_highlight_magnitude,
    )
    build_receptive_field_ax(
        ax_receptive_field,
        field_width=fig_params.field_width,
        field_scale=fig_params.field_scale,
        field_inset_yoffset_fraction=fig_params.field_inset_yoffset_fraction,
        input_inset_yoffset_extra=fig_params.input_inset_yoffset_extra,
        vonmises_concentration=fig_params.vonmises_concentration,
        baseline_rate=fig_params.baseline_rate,
        driven_rate=fig_params.driven_rate,
        gabor_width=fig_params.gabor_width,
        gabor_envelope=fig_params.gabor_envelope,
        gabor_gamma=fig_params.gabor_gamma,
        gabor_halfsize=fig_params.gabor_halfsize,
        gabor_phase=fig_params.gabor_phase,
        x_offset_input_label=fig_params.x_offset_input_label,
        x_offset_rate_label=fig_params.x_offset_rate_label,
        x_offset_field_label=fig_params.x_offset_field_label,
        fontsize=fig_params.ylabel_fontsize,
        include_arrows=fig_params.include_arrows,
    )
    build_tuning_representation_ax(
        ax_tuning,
        hspacing=fig_params.tuning_hspacing,
        vspacing=fig_params.tuning_vspacing,
        fontsize_label=fig_params.tuning_fontsize_label,
        fontsize_title=fig_params.tuning_fontsize_title,
    )
    build_stimulus_trajectory_ax(
        ax_trajectory,
        stims_per_row=fig_params.stimulus_stims_per_row,
        num_edges=fig_params.stimulus_num_edges,
        highlight_magnitude=fig_params.gabor_highlight_magnitude,
        vmax_scale=fig_params.gabor_vmax_scale,
        hspacing=fig_params.stimulus_hspacing,
        vspacing=fig_params.stimulus_vspacing,
        arrow_width=fig_params.stimulus_arrow_width,
        arrow_mutation=fig_params.stimulus_arrow_mutation,
        use_edge_ticks=fig_params.stimulus_use_edge_ticks,
        edge_tick_fraction=fig_params.stimulus_edge_tick_fraction,
        edge_tick_color=fig_params.stimulus_edge_tick_color,
        edge_tick_lw=fig_params.stimulus_edge_tick_lw,
        edge_tick_alpha=fig_params.stimulus_edge_tick_alpha,
    )

    # Analyze main run
    experiment_folder = results_dir("iaf_runs") / fig_params.full_config / fig_params.full_simulations
    metadata = gather_metadata(experiment_folder, experiment_type="hofer")
    num_connections = gather_num_connections(metadata, experiment_type="hofer")
    weights = gather_weights(
        metadata,
        experiment_type="hofer",
        average_method="fraction",
        average_window=0.2,
        norm_by_max_weight=norm_by_max_weight,
        norm_by_num_synapses=norm_by_num_synapses,
        norm_by_total_synapses=norm_by_total_synapses,
        num_connections=num_connections,
    )
    orientation_preference = {sg: np.argmax(weights[sg], axis=-1) % 4 for sg in get_groupnames()}
    build_orientation_confusion_axes(
        ax_distal_simple,
        ax_distal_complex,
        orientation_preference,
        fontsize=fig_params.confusion_fontsize,
        tickfontsize=fig_params.confusion_tickfontsize,
    )

    # Remove white backgrounds
    ax_inputs.set_facecolor("none")
    ax_receptive_field.set_facecolor("none")
    ax_tuning.set_facecolor("none")
    ax_trajectory.set_facecolor("none")
    ax_distal_simple.set_facecolor("none")
    ax_distal_complex.set_facecolor("none")

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("core_figures") / "figure5"
        save_figure(fig, fig_path)

    return fig


def figure5_supplemental(fig_params: Figure5Params, show_fig: bool = True, save_fig: bool = False):
    fig_width = FigParams.single_width
    fig_height = fig_width / 3 * 2

    if fig_params.full_config == "hofer_replacement":
        norm_by_max_weight = True
        norm_by_num_synapses = False
        norm_by_total_synapses = True
    else:
        norm_by_max_weight = True
        norm_by_num_synapses = True
        norm_by_total_synapses = False

    fig = plt.figure(figsize=(fig_width, fig_height), **FigParams.all_fig_params())
    gs = fig.add_gridspec(2, 3)
    ax_proximal_weights_ex0 = fig.add_subplot(gs[0, 0])
    ax_simple_weights_ex0 = fig.add_subplot(gs[0, 1])
    ax_complex_weights_ex0 = fig.add_subplot(gs[0, 2])
    ax_proximal_weights_ex1 = fig.add_subplot(gs[1, 0])
    ax_simple_weights_ex1 = fig.add_subplot(gs[1, 1])
    ax_complex_weights_ex1 = fig.add_subplot(gs[1, 2])

    # Analyze main run
    experiment_folder = results_dir("iaf_runs") / fig_params.full_config / fig_params.full_simulations
    metadata = gather_metadata(experiment_folder, experiment_type="hofer")
    num_connections = gather_num_connections(metadata, experiment_type="hofer")
    weights = gather_weights(
        metadata,
        experiment_type="hofer",
        average_method="fraction",
        average_window=0.2,
        norm_by_max_weight=norm_by_max_weight,
        norm_by_num_synapses=norm_by_num_synapses,
        norm_by_total_synapses=norm_by_total_synapses,
        num_connections=num_connections,
    )

    build_weights_ax(
        ax_proximal_weights_ex0,
        ax_simple_weights_ex0,
        ax_complex_weights_ex0,
        weights,
        vmax=fig_params.gabor_vmax_scale,
        dpratio=fig_params.example_dpratio[0],
        edge=fig_params.example_edge[0],
        simulation=fig_params.example_simulation[0],
        neuron=fig_params.example_neuron[0],
        gabor_width=fig_params.gabor_width,
        gabor_envelope=fig_params.gabor_envelope,
        gabor_gamma=fig_params.gabor_gamma,
        gabor_halfsize=fig_params.gabor_halfsize,
        gabor_phase=fig_params.gabor_phase,
        fontsize=fig_params.ylabel_fontsize,
    )
    build_weights_ax(
        ax_proximal_weights_ex1,
        ax_simple_weights_ex1,
        ax_complex_weights_ex1,
        weights,
        vmax=fig_params.gabor_vmax_scale,
        dpratio=fig_params.example_dpratio[1],
        edge=fig_params.example_edge[1],
        simulation=fig_params.example_simulation[1],
        neuron=fig_params.example_neuron[1],
        gabor_width=fig_params.gabor_width,
        gabor_envelope=fig_params.gabor_envelope,
        gabor_gamma=fig_params.gabor_gamma,
        gabor_halfsize=fig_params.gabor_halfsize,
        gabor_phase=fig_params.gabor_phase,
        fontsize=fig_params.ylabel_fontsize,
        show_titles=False,
    )

    # Remove white backgrounds
    ax_proximal_weights_ex0.set_facecolor("none")
    ax_simple_weights_ex0.set_facecolor("none")
    ax_complex_weights_ex0.set_facecolor("none")
    ax_proximal_weights_ex1.set_facecolor("none")
    ax_simple_weights_ex1.set_facecolor("none")
    ax_complex_weights_ex1.set_facecolor("none")

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("core_figures") / "figure5_supplemental"
        save_figure(fig, fig_path)

    return fig


@dataclass
class Figure6Params:
    full_config: str = "hofer_replacement"
    full_simulations: str = "20250424"
    trajectory_example_ratio: int = 0
    trajectory_example_edge: int = 2
    trajectory_linewidth: float = 1.0
    trajectory_alpha: float = 0.3
    labeltype: str = "label"
    fontsize: float = FigParams.fontsize
    label_fontsize: float = FigParams.fontsize


def figure6(fig_params: Figure6Params, show_fig: bool = True, save_fig: bool = False):
    fig_width = FigParams.double_width
    fig_height = fig_width / 4 * 2

    if fig_params.full_config == "hofer_replacement":
        norm_by_max_weight = True
        norm_by_num_synapses = False
        norm_by_total_synapses = True
    else:
        norm_by_max_weight = True
        norm_by_num_synapses = True
        norm_by_total_synapses = False

    experiment_folder = results_dir("iaf_runs") / fig_params.full_config / fig_params.full_simulations
    metadata = gather_metadata(experiment_folder, experiment_type="hofer")
    num_connections = gather_num_connections(metadata, experiment_type="hofer")
    weights = gather_weights(
        metadata,
        experiment_type="hofer",
        average_method="fraction",
        average_window=0.2,
        norm_by_max_weight=norm_by_max_weight,
        norm_by_num_synapses=norm_by_num_synapses,
        norm_by_total_synapses=norm_by_total_synapses,
        num_connections=num_connections,
    )
    results = gather_results(metadata)
    orientation_preference = {sg: np.argmax(weights[sg], axis=-1) % 4 for sg in get_groupnames()}
    summary, trajectory = summarize_weights(
        weights,
        results,
        metadata,
        orientation_preference["proximal"],
        consolidate_other=True,
        norm_by_max_weight=norm_by_max_weight,
        norm_by_num_synapses=norm_by_num_synapses,
        norm_by_total_synapses=norm_by_total_synapses,
        num_connections=num_connections,
    )

    fig = plt.figure(figsize=(fig_width, fig_height), **FigParams.all_fig_params())
    gs_leftright = fig.add_gridspec(1, 2)
    gs_left_topbottom = gs_leftright[0].subgridspec(2, 1)
    gs_left_top = gs_left_topbottom[0].subgridspec(1, 2)
    gs_left_bottom = gs_left_topbottom[1].subgridspec(1, 2)
    gs_traj = gs_left_top[1].subgridspec(3, 1)
    ax_tuning_type = fig.add_subplot(gs_left_top[0])
    ax_proximal_traj = fig.add_subplot(gs_traj[0])
    ax_simple_traj = fig.add_subplot(gs_traj[1])
    ax_complex_traj = fig.add_subplot(gs_traj[2])
    ax_simple_relative = fig.add_subplot(gs_left_bottom[0])
    ax_complex_relative = fig.add_subplot(gs_left_bottom[1])
    ax_complex_relative.sharey(ax_simple_relative)

    build_tuning_type_axes(ax_tuning_type, fontsize=fig_params.label_fontsize)

    build_tuning_group_trajectory_axes(
        ax_proximal=ax_proximal_traj,
        ax_simple=ax_simple_traj,
        ax_complex=ax_complex_traj,
        trajectory=trajectory,
        example_ratio=fig_params.trajectory_example_ratio,
        example_edge=fig_params.trajectory_example_edge,
        linewidth=fig_params.trajectory_linewidth,
        alpha=fig_params.trajectory_alpha,
        fontsize=fig_params.fontsize,
        labeltype=fig_params.labeltype,
    )

    build_relative_edge_weights_axes(
        ax_simple=ax_simple_relative,
        ax_complex=ax_complex_relative,
        metadata=metadata,
        summary=summary,
        cmap="plasma_r",
        cmap_pinch=0.25,
        fontsize=fig_params.fontsize,
        labeltype=fig_params.labeltype,
    )

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("core_figures") / "figure6"
        save_figure(fig, fig_path)

    return fig


if __name__ == "__main__":
    # Set master parameters for showing / saving figures
    show_fig = False
    save_fig = True

    # Build Figure 1
    # fig1params = Figure1Params()
    # figure1(fig1params, show_fig=show_fig, save_fig=save_fig)

    # Build Figure 2
    # fig2params = Figure2Params()
    # figure2(fig2params, show_fig=show_fig, save_fig=save_fig)

    # Build Figure 3
    # fig3params = Figure3Params()
    # figure3(fig3params, show_fig=show_fig, save_fig=save_fig)

    # Build Figure 4
    # fig4params = Figure4Params()
    # figure4(fig4params, show_fig=show_fig, save_fig=save_fig)

    # Build Figure 5
    fig5params = Figure5Params()
    figure5(fig5params, show_fig=show_fig, save_fig=save_fig)

    # Build Figure 5 Supplemental
    # figure5_supplemental(fig5params, show_fig=show_fig, save_fig=save_fig)

    # Build Figure 6
    # fig6params = Figure6Params()
    # figure6(fig6params, show_fig=show_fig, save_fig=save_fig)
