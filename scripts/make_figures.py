from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from src.files import get_figure_dir
from src.plotting import FigParams, save_figure
from src.schematics import Neuron
from src.experimental import ElifeData, build_ax_amplification_demonstration, build_axes_formatted_elife_data


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


if __name__ == "__main__":
    # Set master parameters for showing / saving figures
    show_fig = False
    save_fig = True

    # Build Figure 1
    fig1params = Figure1Params()
    figure1(fig1params, show_fig=show_fig, save_fig=save_fig)
