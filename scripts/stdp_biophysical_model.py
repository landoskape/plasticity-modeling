import numpy as np
import matplotlib.pyplot as plt

from src.conductance import NMDAR, VGCC, AP

if __name__ == "__main__":
    # Simulation parameters
    v_range = np.linspace(-80, 40, 200)  # Voltage range for steady-state plots
    dt = 0.01  # Time step for numerical integration (ms)
    t_start = 0
    t_end = 3
    t_range = np.linspace(t_start, t_end, int((t_end - t_start) / dt))
    ap_peak_time = 1  # Time of AP peak in ms
    ap_amplitudes = np.linspace(10, 100, 10)  # AP amplitudes to test

    # Initialize channels
    nmdar = NMDAR()
    vgcc = VGCC()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)

    # Subplot 1: Open probabilities
    ax1.plot(v_range, nmdar.open_probability(v_range), "k", label="NMDAR")
    ax1.plot(v_range, vgcc.open_probability_activation(v_range), "b", label="VGCC")
    ax1.set_xlabel("Membrane Potential (mV)")
    ax1.set_ylabel("Open Probability")
    ax1.set_title("Channel Open Probabilities")
    ax1.legend()

    # Subplot 2: Time constants
    ax2.plot(v_range, nmdar.time_constant(v_range), "k", label="NMDAR")
    tau_m, tau_h = vgcc.time_constant(v_range)
    ax2.plot(v_range, tau_m, "b", label="VGCC")
    ax2.set_xlabel("Membrane Potential (mV)")
    ax2.set_ylabel("Time Constant (ms)")
    ax2.set_title("Channel Time Constants")
    ax2.legend()

    plt.tight_layout()
    plt.show(block=True)

    # Create figure: Response to action potentials
    plt.figure(figsize=(12, 4))

    # Generate voltage traces and responses for different AP amplitudes
    for amp in ap_amplitudes:
        # Create AP waveform
        ap = AP(v_amp=amp, v_dur=1, v_base=-70)
        v_trace = ap.voltage(t_range, ap_peak_time)

        # Initialize state variables to steady state at baseline voltage
        m = vgcc.open_probability_activation(v_trace[0])
        h = vgcc.open_probability_inactivation(v_trace[0])
        n = nmdar.open_probability(v_trace[0])

        # Arrays to store results
        m_trace = np.zeros_like(t_range)
        h_trace = np.zeros_like(t_range)
        n_trace = np.zeros_like(t_range)
        m_trace[0] = m
        h_trace[0] = h
        n_trace[0] = n

        # Numerical integration using Euler's method
        for i in range(1, len(t_range)):
            m += dt * vgcc.dmdt(v_trace[i - 1], m)
            h += dt * vgcc.dhdt(v_trace[i - 1], h)
            n += dt * nmdar.dndt(v_trace[i - 1], n)

            m_trace[i] = m
            h_trace[i] = h
            n_trace[i] = n

        # Calculate open probabilities
        vgcc_p = m_trace**2 * h_trace  # VGCC open probability
        nmdar_p = n_trace  # NMDAR open probability

        # Plot results
        plt.subplot(131)
        plt.plot(t_range, v_trace, color="k", label=f"{amp:.0f} mV")

        plt.subplot(132)
        plt.plot(t_range, nmdar_p, color="k", label=f"{amp:.0f} mV")

        plt.subplot(133)
        plt.plot(t_range, vgcc_p, color="k", label=f"{amp:.0f} mV")

    # Format AP response plots
    plt.subplot(131)
    plt.xlabel("Time (ms)")
    plt.ylabel("Voltage (mV)")
    plt.title("Action Potentials")
    plt.legend()

    plt.subplot(132)
    plt.xlabel("Time (ms)")
    plt.ylabel("Open Probability")
    plt.title("NMDAR Response")
    plt.legend()

    plt.subplot(133)
    plt.xlabel("Time (ms)")
    plt.ylabel("Open Probability")
    plt.title("VGCC Response")
    plt.legend()
    plt.tight_layout()

    plt.show(block=True)
