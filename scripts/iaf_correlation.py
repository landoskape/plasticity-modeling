from argparse import ArgumentParser
from datetime import datetime
import joblib
from tqdm import tqdm
from src.files import results_dir, save_repo_snapshot
from src.iaf.experiments import get_experiment


def get_args():
    parser = ArgumentParser(description="Run an experiment varying the distal depression-potentiation ratio.")
    parser.add_argument(
        "--config",
        type=str,
        default="correlated",
        choices=["correlated", "ica"],
        help="Which configuration to use for the experiment",
    )
    parser.add_argument(
        "--distal_dp_ratios",
        type=float,
        nargs="+",
        default=[1.0, 1.025, 1.05, 1.075, 1.1],
        help="The distal depression-potentiation ratios to simulate.",
    )
    parser.add_argument(
        "--num_neurons",
        type=int,
        default=3,
        help=(
            "The number of neurons to simulate for each simulation instance.\n"
            "Each simulation will use the same source populations but "
            "randomized neurons and synapses (all with the same metaparameters)."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="The number of times to repeat the experiment for each ratio.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=2400,
        help="The duration of the simulation in seconds.",
    )
    parser.add_argument(
        "--no_distal",
        action="store_true",
        help=(
            "Whether to run the experiment without distal dendrites.\n"
            "If used, then the distal_dp_ratios will be used to set the "
            "dp_ratio of proximal synapses."
        ),
    )
    parser.add_argument(
        "--save_source_rates",
        action="store_true",
        help="Whether to save the source rates of the simulation.",
    )
    parser.add_argument(
        "--dp_ratio_index",
        type=int,
        default=None,
        help="Index into distal_dp_ratios for this job (for array jobs).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Repeat number for this job (for array jobs).",
    )
    parser.add_argument(
        "--exp_folder",
        type=str,
        default=None,
        help="Name for the experiment folder. If provided, uses {name}_{timestamp} format. If not provided, defaults to timestamp with increment.",
    )
    return parser.parse_args()


def get_experiment_folder(args):
    timestamp = datetime.now().strftime("%Y%m%d")

    if args.exp_folder is not None:
        # Use {name}_{timestamp} format without increment
        folder_name = f"{args.exp_folder}_{timestamp}"
        exp_folder = results_dir("iaf_runs") / args.config / folder_name
    else:
        # Default behavior: use timestamp with increment if folder exists
        exp_folder = results_dir("iaf_runs") / args.config / timestamp
        if exp_folder.exists():
            # Add a _# to the folder name until we find a unique name
            i = 1
            while exp_folder.exists():
                exp_folder = results_dir("iaf_runs") / args.config / f"{timestamp}_{i}"
                i += 1

    if not exp_folder.exists():
        exp_folder.mkdir(parents=True, exist_ok=True)
    return exp_folder


def run_experiment(args):
    experiment_folder = get_experiment_folder(args)

    # Get parameters for easier access
    config = args.config
    distal_dp_ratios = args.distal_dp_ratios
    num_neurons = args.num_neurons
    repeats = args.repeats
    duration = args.duration
    no_distal = args.no_distal
    save_source_rates = args.save_source_rates
    dp_ratio_index = args.dp_ratio_index
    repeat = args.repeat

    # Save the parameters (only on first job to avoid overwriting)
    args_path = experiment_folder / "args.joblib"
    if not args_path.exists():
        joblib.dump(args, args_path)
        # Save the state of the repo (only on first job)
        save_repo_snapshot(experiment_folder / "repo.zip", verbose=False)

    # If dp_ratio_index and repeat are specified, run only that combination
    if dp_ratio_index is not None and repeat is not None:
        # Validate indices
        if dp_ratio_index < 0 or dp_ratio_index >= len(distal_dp_ratios):
            raise ValueError(f"dp_ratio_index {dp_ratio_index} out of range [0, {len(distal_dp_ratios)})")
        if repeat < 0 or repeat >= repeats:
            raise ValueError(f"repeat {repeat} out of range [0, {repeats})")

        distal_dp_ratio = distal_dp_ratios[dp_ratio_index]

        if not no_distal:
            sim, cfg = get_experiment(
                config,
                distal_dp_ratio=distal_dp_ratio,
                num_simulations=num_neurons,
                no_distal=no_distal,
            )
        else:
            sim, cfg = get_experiment(
                config,
                base_dp_ratio=distal_dp_ratio,
                num_simulations=num_neurons,
                no_distal=no_distal,
            )

        results = sim.run(duration=duration, save_source_rates=save_source_rates)
        results["sim"] = sim
        results["cfg"] = cfg

        # Save the results
        results_path = experiment_folder / f"ratio_{dp_ratio_index}_repeat_{repeat}.joblib"
        joblib.dump(results, results_path)
    else:
        # Run all the requested experiments (backward compatibility)
        for iratio, distal_dp_ratio in enumerate(tqdm(distal_dp_ratios, desc="Distal DP Ratios")):
            for repeat in tqdm(range(repeats), desc="Repeats"):
                if not no_distal:
                    sim, cfg = get_experiment(
                        config,
                        distal_dp_ratio=distal_dp_ratio,
                        num_simulations=num_neurons,
                        no_distal=no_distal,
                    )
                else:
                    sim, cfg = get_experiment(
                        config,
                        base_dp_ratio=distal_dp_ratio,
                        num_simulations=num_neurons,
                        no_distal=no_distal,
                    )

                results = sim.run(duration=duration, save_source_rates=save_source_rates)
                results["sim"] = sim
                results["cfg"] = cfg

                # Save the results
                results_path = experiment_folder / f"ratio_{iratio}_repeat_{repeat}.joblib"
                joblib.dump(results, results_path)


if __name__ == "__main__":
    args = get_args()
    run_experiment(args)
