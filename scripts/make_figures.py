import matplotlib.pyplot as plt
from src.files import get_figure_dir
from src.plotting import FigParams, save_figure
from src.schematics import Neuron
from src.experimental import ElifeData, build_ax_amplification_demonstration, build_axes_formatted_elife_data


def figure1(show_error: bool = True, se: bool = True, show_fig: bool = True, save_fig: bool = False):
    # Get ELife Data
    data = ElifeData()

    fig_width = FigParams.onepointfive_width
    fig_height = fig_width / 3 * 2
    fig = plt.figure(figsize=(fig_width, fig_height), **FigParams.all_fig_params())
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.2, 0.8])
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
    build_ax_amplification_demonstration(ax_amp_demonstration, data, icell=17, start_pos=20, delta_pos=10)

    # Build formatted ELife data figure
    build_axes_formatted_elife_data(
        ax_ap_trace, ax_amp_trace, ax_ap_peaks, ax_amp_peaks, data, show_error=show_error, se=se
    )

    if show_fig:
        plt.show(block=True)

    if save_fig:
        fig_path = get_figure_dir("core_figures") / "figure1"
        save_figure(fig, fig_path)

    return fig


if __name__ == "__main__":
    show_error = True
    se = True
    show_fig = False
    save_fig = True

    figure1(show_fig=show_fig, save_fig=save_fig)
