from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime

from src.files import results_dir

from src.iaf.optuna_proximal import ProximalSearchSpace, run_optuna_proximal


def get_args():
    parser = ArgumentParser(description="Run Optuna optimization for proximal tuning.")
    parser.add_argument("--study-name", type=str, required=True, help="Base name for the Optuna study.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional run directory. If omitted, a timestamped directory is created.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hofer_all_proximal",
        help="Config name (without .yaml) to use as a base.",
    )
    parser.add_argument("--duration", type=int, default=2400, help="Simulation duration in seconds.")
    parser.add_argument("--num-neurons", type=int, default=3, help="Number of neurons per trial.")
    parser.add_argument("--n-trials", type=int, default=100, help="Number of Optuna trials to run.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout for optimize() in seconds.")
    parser.add_argument(
        "--storage-timeout",
        type=int,
        default=60,
        help="SQLite connection timeout in seconds.",
    )
    parser.add_argument(
        "--average-window",
        type=float,
        default=0.1,
        help="Fraction of the final window to average for entropy.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for Optuna and RNG.")

    return parser.parse_args()


def main() -> None:
    args = get_args()
    average_window: float = args.average_window
    space = ProximalSearchSpace()
    run_dir = args.run_dir
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = results_dir("optuna_runs") / f"{args.study_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_optuna_proximal(
        study_name=args.study_name,
        run_dir=run_dir,
        config_name=args.config,
        duration=args.duration,
        num_neurons=args.num_neurons,
        n_trials=args.n_trials,
        timeout=args.timeout,
        storage_timeout=args.storage_timeout,
        average_window=average_window,
        seed=args.seed,
        space=space,
    )


if __name__ == "__main__":
    main()
