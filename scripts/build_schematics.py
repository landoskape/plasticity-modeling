from __future__ import annotations
import matplotlib.pyplot as plt
from src.schematics import Neuron
from src.files import get_figure_dir
from src.plotting import save_figure


def base_schematic(show_fig: bool = True, save_fig: bool = False):
    """
    Generate a base neuron schematic visualization.

    Similar to initial_schematic but with slightly different configuration.
    Creates a visualization of a neuron and optionally displays and/or saves the figure.

    Args:
        show_fig: Whether to display the figure. Defaults to True.
        save_fig: Whether to save the figure to disk. Defaults to False.
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create neuron object and plot at origin with default scale
    neuron = Neuron()
    elements = neuron.plot(ax, origin=(0, 0), scale=1.0)

    # Set plot limits and aesthetics
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1, 6)
    ax.set_aspect("equal")

    # Remove ticks and spines for cleaner visualization
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    if show_fig:
        plt.show(block=True)

    if save_fig:
        figure_path = get_figure_dir("schematic") / "dendrite_branch_schematic"
        save_figure(fig, figure_path)


if __name__ == "__main__":
    # When script is run directly, show and save both types of schematics
    show_fig = True
    save_fig = True
    base_schematic(show_fig=show_fig, save_fig=save_fig)
