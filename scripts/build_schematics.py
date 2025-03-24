import matplotlib.pyplot as plt
from src.schematics import Neuron


if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 8))
    neuron = Neuron()
    elements = neuron.plot(ax, origin=(0, 0), scale=1.0)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1, 6)
    ax.set_aspect("equal")
    ax.set_axisbelow(True)
    plt.grid(True)
    plt.show()
