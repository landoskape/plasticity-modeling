#!/usr/bin/env python3
import argparse
import os
import json
from pathlib import Path
import time
import joblib
from src.iaf.experiments import get_correlated_experiment, get_ica_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Run simulations with specific apical_dp_ratio")
    parser.add_argument(
        "--array_index",
        type=int,
        required=True,
        help="SLURM array index (maps to a specific apical_dp_ratio)",
    )
    parser.add_argument(
        "--experiment_type",
        type=str,
        choices=["correlated", "ica"],
        default="correlated",
        help="Type of experiment to run",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=5,
        help="Number of simulations to run for each apical_dp_ratio",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10000,
        help="Duration of the simulation in seconds",
    )
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument(
        "--dp_ratios",
        type=str,
        default="1.0, 1.025, 1.05, 1.075, 1.1",
        help="Comma-separated list of apical_dp_ratio values",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse the dp_ratios string into a list of floats
    dp_ratios = [float(x) for x in args.dp_ratios.split(",")]

    # Map array index to a specific dp_ratio
    if args.array_index < 0 or args.array_index >= len(dp_ratios):
        raise ValueError(f"Array index {args.array_index} out of range (0-{len(dp_ratios)-1})")

    dp_ratio = dp_ratios[args.array_index]
    print(f"Running simulations with apical_dp_ratio = {dp_ratio}")

    # Create output directory
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", args.array_index)
    output_path = Path(args.output_dir) / f"dp_ratio_{dp_ratio}_{job_id}_{array_task_id}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = {
        "dp_ratio": dp_ratio,
        "experiment_type": args.experiment_type,
        "num_simulations": args.num_simulations,
        "job_id": job_id,
        "array_task_id": array_task_id,
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
    }

    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run simulations
    results = []
    for i in range(args.num_simulations):
        print(f"Running simulation {i+1}/{args.num_simulations}")

        # Get the appropriate experiment based on type
        if args.experiment_type == "correlated":
            simulation = get_correlated_experiment(apical_dp_ratio=dp_ratio)
        else:  # ica
            simulation = get_ica_experiment(apical_dp_ratio=dp_ratio)

        # Run the simulation
        start_time = time.time()
        results = simulation.run(duration=args.duration)
        end_time = time.time()

        # Extract and save results
        sim_results = {
            "simulation_index": i,
            "runtime_seconds": end_time - start_time,
            "weights": simulation.get_weights().tolist() if hasattr(simulation, "get_weights") else None,
            # Add other relevant metrics from your simulation
        }
        results.append(sim_results)

        results["simulation_index"] = i
        results["runtime_seconds"] = end_time - start_time
        results["simulation"] = simulation
        joblib.dump(results, output_path / f"simulation_{i}.joblib")

    print(f"Completed {args.num_simulations} simulations with apical_dp_ratio = {dp_ratio}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
