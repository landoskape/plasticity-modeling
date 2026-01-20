from __future__ import annotations
import joblib
from src.conductance import run_simulations
from src.files import data_dir

if __name__ == "__main__":
    # Run simulations with many ap amplitudes
    # (Will run from 0 to 100 mV amplitude from rest)
    num_ap_amplitudes = 400
    data = run_simulations(num_ap_amplitudes=num_ap_amplitudes)
    # joblib.dump(data, data_dir() / "conductance_runs.joblib")
