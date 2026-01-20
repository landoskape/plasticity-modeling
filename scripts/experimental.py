from __future__ import annotations
from src.experimental import ElifeData, plot_amplification_demonstration, plot_formatted_elife_data


if __name__ == "__main__":
    show_error = True
    se = True
    show_fig = False
    save_fig = True
    data = ElifeData()
    plot_amplification_demonstration(data, show_fig=show_fig, save_fig=save_fig)
    plot_formatted_elife_data(data, show_error=show_error, se=se, show_fig=show_fig, save_fig=save_fig)
