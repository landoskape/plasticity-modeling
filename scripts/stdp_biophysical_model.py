from __future__ import annotations
import numpy as np
from src.conductance import plot_channel_properties, plot_simulations

if __name__ == "__main__":
    min_voltage = -80
    max_voltage = 40
    num_points = 200

    dt = 0.01  # Time step for numerical integration (ms)
    t_start = 0
    t_end = 3
    ap_peak_time = 1  # Time of AP peak in ms
    ap_amplitudes = np.linspace(20, 100, 5)  # AP amplitudes to test

    show_fig = True
    save_fig = True

    fig_properties = plot_channel_properties(
        min_voltage,
        max_voltage,
        num_points,
        show_fig=show_fig,
        save_fig=save_fig,
    )
    fig_simulations = plot_simulations(
        t_start,
        t_end,
        dt,
        ap_peak_time,
        ap_amplitudes,
        show_fig=show_fig,
        save_fig=save_fig,
    )
