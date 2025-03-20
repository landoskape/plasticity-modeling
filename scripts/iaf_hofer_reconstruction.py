from argparse import ArgumentParser
from datetime import datetime
import joblib
from src.files import results_dir, save_repo_snapshot
from src.iaf.experiments import get_experiment


def get_args():
    parser = ArgumentParser(description="Run an experiment varying the apical depression-potentiation ratio.")
    parser.add_argument(
        "--config",
        type=str,
        default="hofer",
        choices=["hofer"],
        help="Which configuration to use for the experiment",
    )
    parser.add_argument(
        "--apical_dp_ratios",
        type=float,
        nargs="+",
        default=[1.0, 1.025, 1.05, 1.075, 1.1],
        help="The apical depression-potentiation ratios to simulate.",
    )
    parser.add_argument(
        "--edge_probabilities",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0],
        help="The edge probabilities to simulate with.",
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
        "--no_apical",
        action="store_true",
        help=(
            "Whether to run the experiment without apical dendrites.\n"
            "If used, then the apical_dp_ratios will be used to set the "
            "dp_ratio of basal synapses."
        ),
    )
    return parser.parse_args()


def get_experiment_folder(args):
    timestamp = datetime.now().strftime("%Y%m%d")
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
    apical_dp_ratios = args.apical_dp_ratios
    edge_probabilities = args.edge_probabilities
    num_neurons = args.num_neurons
    repeats = args.repeats
    duration = args.duration
    no_apical = args.no_apical

    # Save the parameters
    joblib.dump(args, experiment_folder / "args.joblib")

    # Save the state of the repo
    save_repo_snapshot(experiment_folder / "repo.zip", verbose=False)

    # Run all the requested experiments
    for iratio, apical_dp_ratio in enumerate(apical_dp_ratios):
        for iedge, edge_probability in enumerate(edge_probabilities):
            for repeat in range(repeats):
                if not no_apical:
                    sim, cfg = get_experiment(
                        config,
                        apical_dp_ratio=apical_dp_ratio,
                        num_simulations=num_neurons,
                        no_apical=no_apical,
                        edge_probability=edge_probability,
                    )
                else:
                    sim, cfg = get_experiment(
                        config,
                        base_dp_ratio=apical_dp_ratio,
                        num_simulations=num_neurons,
                        no_apical=no_apical,
                        edge_probability=edge_probability,
                    )

                results = sim.run(duration=duration)
                results["sim"] = sim
                results["cfg"] = cfg

                # Save the results
                results_path = experiment_folder / f"ratio_{iratio}_edge_{iedge}_repeat_{repeat}.joblib"
                joblib.dump(results, results_path)


if __name__ == "__main__":
    args = get_args()
    run_experiment(args)
