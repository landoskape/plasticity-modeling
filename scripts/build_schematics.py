import matplotlib.pyplot as plt
from src.schematics import Neuron
from src.files import get_figure_dir
from src.plotting import save_figure


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 8))
    neuron = Neuron()
    elements = neuron.plot(ax, origin=(0, 0), scale=1.0)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1, 6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    figure_path = get_figure_dir("schematic") / "dendrite_branch_schematic"
    save_figure(fig, figure_path)
