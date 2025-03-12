from src.iaf.experiments import parameter_grid_search, run_grid_search
from src.iaf.simulation import Simulation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os


def run_simulation_experiment(config):
    """
    Run a simulation with the given configuration and return results.

    Args:
        config: SimulationConfig object

    Returns:
        Dictionary containing simulation results
    """
    # Create and run simulation
    simulation = Simulation.from_config(config)
    simulation.run(duration=config.duration)

    # Collect results (customize this based on what metrics you want to track)
    results = {
        "firing_rate": simulation.neuron.output_spike_count / config.duration,
        "mean_weight_basal": np.mean(simulation.synapses["basal"].weights),
        "mean_weight_apical": np.mean(simulation.synapses["apical"].weights),
        # Add more metrics as needed
    }

    return results


def save_results(params, results, results_list):
    """
    Save the results from a single parameter combination.

    Args:
        params: Dictionary of parameter values for this run
        results: Dictionary of results from this run
        results_list: List to append the combined results to
    """
    # Combine parameters and results into a single dictionary
    combined = {**params, **results}
    results_list.append(combined)

    # Print progress update
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    print(f"Completed simulation with {param_str}")
    print(f"  Results: firing_rate={results['firing_rate']:.2f} Hz")


def run_correlation_grid_search():
    """
    Example grid search over correlation parameters.
    """
    # Parameters to explore
    parameter_ranges = {
        "sources.excitatory.max_correlation": [0.2, 0.4, 0.6],
        "sources.excitatory.rate_mean": [10.0, 20.0, 30.0],
        "synapses.basal.plasticity.depression_potentiation_ratio": [1.0, 1.1, 1.2],
        "synapses.apical.plasticity.depression_potentiation_ratio": [0.9, 1.0, 1.1],
    }

    # Storage for results
    all_results = []

    # Run the grid search
    run_grid_search(
        base_config_file="correlated.yaml",
        parameter_ranges=parameter_ranges,
        experiment_fn=run_simulation_experiment,
        result_handler=lambda params, results: save_results(params, results, all_results),
    )

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(all_results)

    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/correlation_grid_search.csv", index=False)

    # Generate some basic visualizations
    plot_grid_search_results(results_df)

    return results_df


def run_ica_grid_search():
    """
    Example grid search over ICA parameters.
    """
    # Parameters to explore
    parameter_ranges = {
        "sources.excitatory.source_strength": [2.0, 3.0, 4.0],
        "sources.excitatory.num_signals": [2, 3, 4],
        "synapses.basal.plasticity.depression_potentiation_ratio": [1.0, 1.1, 1.2],
        "synapses.apical.plasticity.depression_potentiation_ratio": [0.9, 1.0, 1.1],
    }

    # Storage for results
    all_results = []

    # Run the grid search
    run_grid_search(
        base_config_file="ica.yaml",
        parameter_ranges=parameter_ranges,
        experiment_fn=run_simulation_experiment,
        result_handler=lambda params, results: save_results(params, results, all_results),
    )

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(all_results)

    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/ica_grid_search.csv", index=False)

    # Generate some basic visualizations
    plot_grid_search_results(results_df)

    return results_df


def plot_grid_search_results(results_df):
    """
    Generate visualizations from grid search results.

    Args:
        results_df: DataFrame containing grid search results
    """
    # Create output directory
    os.makedirs("results/plots", exist_ok=True)

    # Get parameter columns (those that vary)
    param_cols = [
        col for col in results_df.columns if col not in ["firing_rate", "mean_weight_basal", "mean_weight_apical"]
    ]

    # For each parameter, create plots showing its effect on firing rate and weights
    for param in param_cols:
        if len(results_df[param].unique()) > 1:  # Only plot if parameter actually varies
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Group by this parameter and plot means with error bars
            grouped = results_df.groupby(param)
            means = grouped.mean()
            std = grouped.std()

            # Plot firing rate
            axes[0].errorbar(means.index, means["firing_rate"], yerr=std["firing_rate"], marker="o")
            axes[0].set_xlabel(param)
            axes[0].set_ylabel("Firing Rate (Hz)")
            axes[0].set_title(f"Effect of {param} on Firing Rate")

            # Plot weights
            axes[1].errorbar(
                means.index, means["mean_weight_basal"], yerr=std["mean_weight_basal"], marker="o", label="Basal"
            )
            axes[1].errorbar(
                means.index, means["mean_weight_apical"], yerr=std["mean_weight_apical"], marker="s", label="Apical"
            )
            axes[1].set_xlabel(param)
            axes[1].set_ylabel("Mean Weight")
            axes[1].set_title(f"Effect of {param} on Synaptic Weights")
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(f"results/plots/effect_of_{param.replace('.', '_')}.png")
            plt.close()

    # Create heatmap for interactions between pairs of parameters
    # For simplicity, we'll just do this for the first two parameters if there are multiple
    if len(param_cols) >= 2:
        param1, param2 = param_cols[0], param_cols[1]

        # Check if we have enough distinct values for a meaningful heatmap
        if len(results_df[param1].unique()) > 1 and len(results_df[param2].unique()) > 1:
            # Create pivot tables
            pivot_firing = results_df.pivot_table(values="firing_rate", index=param1, columns=param2, aggfunc="mean")

            # Plot heatmap for firing rate
            plt.figure(figsize=(8, 6))
            plt.pcolormesh(pivot_firing)
            plt.colorbar(label="Firing Rate (Hz)")
            plt.xlabel(param2)
            plt.ylabel(param1)
            plt.title(f"Interaction between {param1} and {param2} on Firing Rate")
            plt.tight_layout()
            plt.savefig(f"results/plots/heatmap_{param1.replace('.', '_')}_{param2.replace('.', '_')}.png")
            plt.close()


if __name__ == "__main__":
    # Run the grid searches
    print("Running grid search for correlation model...")
    corr_results = run_correlation_grid_search()

    print("\nRunning grid search for ICA model...")
    ica_results = run_ica_grid_search()

    print("\nGrid searches complete. Results saved to results/ directory.")
