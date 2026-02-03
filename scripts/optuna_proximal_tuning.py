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
    parser.add_argument("--n-trials", type=int, default=10, help="Number of Optuna trials to run.")
    parser.add_argument("--timeout", type=int, default=None, help="Optional timeout for optimize() in seconds.")
    parser.add_argument(
        "--storage-timeout",
        type=int,
        default=20,
        help="SQLite connection timeout in seconds.",
    )
    parser.add_argument(
        "--average-window",
        type=float,
        default=0.1,
        help="Fraction of the final window to average for entropy.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for Optuna and RNG.")

    parser.add_argument("--num-synapses-min", type=int, default=36)
    parser.add_argument("--num-synapses-max", type=int, default=3600)
    parser.add_argument("--num-synapses-step", type=int, default=36)

    parser.add_argument("--max-weight-min", type=float, default=1e-13)
    parser.add_argument("--max-weight-max", type=float, default=1e-8)
    parser.add_argument("--max-weight-log", action="store_true", default=True)

    parser.add_argument("--conductance-threshold-min", type=float, default=0.0)
    parser.add_argument("--conductance-threshold-max", type=float, default=0.5)

    parser.add_argument("--independent-noise-rate-min", type=float, default=0.0)
    parser.add_argument("--independent-noise-rate-max", type=float, default=1.0)

    parser.add_argument("--stdp-rate-min", type=float, default=1e-4)
    parser.add_argument("--stdp-rate-max", type=float, default=0.1)
    parser.add_argument("--stdp-rate-log", action="store_true")

    parser.add_argument("--dp-ratio-min", type=float, default=0.99)
    parser.add_argument("--dp-ratio-max", type=float, default=1.2)

    return parser.parse_args()


def main() -> None:
    args = get_args()
    average_window: float = args.average_window
    space = ProximalSearchSpace(
        num_synapses_min=args.num_synapses_min,
        num_synapses_max=args.num_synapses_max,
        num_synapses_step=args.num_synapses_step,
        max_weight_min=args.max_weight_min,
        max_weight_max=args.max_weight_max,
        max_weight_log=args.max_weight_log,
        conductance_threshold_min=args.conductance_threshold_min,
        conductance_threshold_max=args.conductance_threshold_max,
        independent_noise_rate_min=args.independent_noise_rate_min,
        independent_noise_rate_max=args.independent_noise_rate_max,
        stdp_rate_min=args.stdp_rate_min,
        stdp_rate_max=args.stdp_rate_max,
        stdp_rate_log=args.stdp_rate_log,
        dp_ratio_min=args.dp_ratio_min,
        dp_ratio_max=args.dp_ratio_max,
    )
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
