from __future__ import annotations
from argparse import ArgumentParser
from pathlib import Path
import joblib
from src.conductance import run_simulations
from src.files import results_dir

if __name__ == "__main__":
    parser = ArgumentParser(description="Run conductance simulations with varying AP amplitudes.")
    parser.add_argument(
        "--num_ap_amplitudes",
        type=int,
        default=400,
        help="Number of AP amplitudes to simulate (0 to 100 mV from rest).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path. Default: results/conductance_runs.joblib",
    )
    args = parser.parse_args()

    # Run simulations with many ap amplitudes
    # (Will run from 0 to 100 mV amplitude from rest)
    data = run_simulations(num_ap_amplitudes=args.num_ap_amplitudes)

    output_path = Path(args.output) if args.output else results_dir() / "conductance_runs.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, output_path)
