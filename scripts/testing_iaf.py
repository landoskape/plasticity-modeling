from src.simulation import run_simulation

if __name__ == "__main__":
    spike_times, basal_weights, apical_weights = run_simulation(
        duration=20.0,
        apical_depression_ratio=1.0,
        num_signals=3,
    )
    print("hi")
