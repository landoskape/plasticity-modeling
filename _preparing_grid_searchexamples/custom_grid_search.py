from src.iaf.experiments import parameter_grid_search, run_grid_search
from src.iaf.simulation import Simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from pathlib import Path


class GridSearchExperiment:
    """
    A configurable grid search experiment class that can be easily customized.
    """

    def __init__(
        self,
        base_config_file,
        parameter_ranges,
        result_metrics=None,
        output_dir=None,
        experiment_name=None,
        random_seeds=None,
    ):
        """
        Initialize a grid search experiment.

        Args:
            base_config_file: Name of the base YAML configuration file (e.g., "correlated.yaml")
            parameter_ranges: Dictionary mapping parameter paths to lists of values to test
            result_metrics: List of metrics to track (function names to call on simulation results)
            output_dir: Directory to save results to
            experiment_name: Name for this experiment (used in file naming)
            random_seeds: List of random seeds to use for each parameter combination
        """
        self.base_config_file = base_config_file
        self.parameter_ranges = parameter_ranges

        # Set up metrics to track
        self.result_metrics = result_metrics or ["firing_rate", "mean_weights"]

        # Set up output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"grid_search_{timestamp}"
        self.output_dir = Path(output_dir or "results") / self.experiment_name
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "plots", exist_ok=True)

        # Random seeds for reproducibility and robustness testing
        self.random_seeds = random_seeds or [None]  # Default to using config seed

        # Storage for results
        self.results = []

    def run_simulation(self, config):
        """
        Run a simulation with the given configuration and compute metrics.

        Args:
            config: SimulationConfig object

        Returns:
            Dictionary of computed metrics
        """
        # Create and run simulation
        simulation = Simulation.from_config(config)
        simulation.run(duration=config.duration)

        # Compute standard metrics
        results = {}

        # Firing rate
        if "firing_rate" in self.result_metrics:
            results["firing_rate"] = simulation.neuron.output_spike_count / config.duration

        # Mean weights for each synapse group
        if "mean_weights" in self.result_metrics:
            for synapse_name, synapse in simulation.synapses.items():
                results[f"mean_weight_{synapse_name}"] = np.mean(synapse.weights)

        # Weight distributions
        if "weight_distributions" in self.result_metrics:
            for synapse_name, synapse in simulation.synapses.items():
                # Store histograms or basic stats about distribution
                hist, edges = np.histogram(synapse.weights, bins=20)
                results[f"weight_hist_{synapse_name}"] = {
                    "hist": hist,
                    "edges": edges,
                    "std": np.std(synapse.weights),
                    "median": np.median(synapse.weights),
                }

        # Add custom metrics here
        # ...

        return results

    def save_result(self, params, results):
        """
        Save a single simulation result.

        Args:
            params: Dictionary of parameter values
            results: Dictionary of metric results
        """
        # Special handling for histogram data
        simple_results = {}
        for key, value in results.items():
            if isinstance(value, dict) and "hist" in value:
                # For weight distributions, just store summary statistics
                simple_results[f"{key}_std"] = value["std"]
                simple_results[f"{key}_median"] = value["median"]
            else:
                simple_results[key] = value

        # Combine parameters and results
        combined = {**params, **simple_results}
        self.results.append(combined)

        # Print progress update
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"Completed simulation with {param_str}")
        if "firing_rate" in results:
            print(f"  Results: firing_rate={results['firing_rate']:.2f} Hz")

    def run(self):
        """
        Run the complete grid search experiment.
        """
        # Add seeds to parameter ranges if multiple
        if len(self.random_seeds) > 1:
            self.parameter_ranges["seed"] = self.random_seeds

        # Run grid search
        run_grid_search(
            base_config_file=self.base_config_file,
            parameter_ranges=self.parameter_ranges,
            experiment_fn=self.run_simulation,
            result_handler=self.save_result,
        )

        # Convert results to DataFrame for analysis
        self.results_df = pd.DataFrame(self.results)

        # Save results to CSV
        csv_path = self.output_dir / f"{self.experiment_name}.csv"
        self.results_df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        return self.results_df

    def analyze(self):
        """
        Analyze the results and generate visualizations.
        """
        if not hasattr(self, "results_df"):
            raise ValueError("No results found. Run the experiment first.")

        # Get parameter columns (those that vary)
        metric_cols = [col for col in self.results_df.columns if any(col.startswith(m) for m in self.result_metrics)]
        param_cols = [col for col in self.results_df.columns if col not in metric_cols and col != "seed"]

        # Only analyze parameters that actually vary
        param_cols = [col for col in param_cols if len(self.results_df[col].unique()) > 1]

        # Generate plots for each parameter's effect on metrics
        self._generate_parameter_effect_plots(param_cols, metric_cols)

        # Generate heatmaps for parameter interactions
        if len(param_cols) >= 2:
            self._generate_interaction_heatmaps(param_cols, metric_cols)

        # If we used multiple seeds, analyze variance
        if "seed" in self.results_df.columns and len(self.results_df["seed"].unique()) > 1:
            self._analyze_seed_variance(param_cols, metric_cols)

    def _generate_parameter_effect_plots(self, param_cols, metric_cols):
        """Generate plots showing each parameter's effect on metrics."""
        for param in param_cols:
            # Filter basic metrics (not histograms)
            basic_metrics = [m for m in metric_cols if not m.startswith("weight_hist")]

            # Create plot with subplots for each metric
            n_metrics = len(basic_metrics)
            fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 5))
            if n_metrics == 1:
                axes = [axes]  # Make iterable for single metric case

            # Group by this parameter
            grouped = self.results_df.groupby(param)
            means = grouped.mean(numeric_only=True)
            std = grouped.std(numeric_only=True)

            # Plot each metric
            for i, metric in enumerate(basic_metrics):
                if metric in means.columns:
                    axes[i].errorbar(means.index, means[metric], yerr=std[metric], marker="o")
                    axes[i].set_xlabel(param)
                    axes[i].set_ylabel(metric)
                    axes[i].set_title(f"Effect of {param} on {metric}")

            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / f"effect_of_{param.replace('.', '_')}.png")
            plt.close()

    def _generate_interaction_heatmaps(self, param_cols, metric_cols):
        """Generate heatmaps showing interactions between pairs of parameters."""
        for i, param1 in enumerate(param_cols[:-1]):
            for param2 in param_cols[i + 1 :]:
                # Filter to basic metrics
                basic_metrics = [m for m in metric_cols if not m.startswith("weight_hist")]

                for metric in basic_metrics:
                    if metric in self.results_df.columns:
                        # Create pivot table for this pair of parameters
                        pivot = self.results_df.pivot_table(values=metric, index=param1, columns=param2, aggfunc="mean")

                        # Plot heatmap
                        plt.figure(figsize=(8, 6))
                        sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f")
                        plt.title(f"Interaction: {param1} vs {param2} on {metric}")
                        plt.tight_layout()
                        plt.savefig(
                            self.output_dir
                            / "plots"
                            / f"heatmap_{param1.replace('.', '_')}_{param2.replace('.', '_')}_{metric}.png"
                        )
                        plt.close()

    def _analyze_seed_variance(self, param_cols, metric_cols):
        """Analyze variance across different random seeds."""
        # Group by all parameters except seed
        if param_cols:
            grouped = self.results_df.groupby(param_cols)

            # Calculate coefficient of variation for each metric
            cv_results = grouped[metric_cols].apply(lambda x: x.std() / x.mean() if x.mean().any() != 0 else x.std())

            # Save results
            cv_results.to_csv(self.output_dir / "seed_variance_analysis.csv")

            # Plot the coefficient of variation
            plt.figure(figsize=(10, 6))
            cv_results.mean().plot(kind="bar")
            plt.title("Average Coefficient of Variation Across Seeds")
            plt.ylabel("Coefficient of Variation")
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / "seed_variance.png")
            plt.close()


# Example usage
if __name__ == "__main__":
    # Define parameters to explore for correlated input experiment
    correlated_params = {
        # Input parameters
        "sources.excitatory.max_correlation": [0.2, 0.4, 0.6],
        "sources.excitatory.rate_mean": [10.0, 20.0, 30.0],
        # Plasticity parameters
        "synapses.basal.plasticity.depression_potentiation_ratio": [1.0, 1.1, 1.2],
        "synapses.apical.plasticity.depression_potentiation_ratio": [0.9, 1.0, 1.1],
        # Neuron parameters
        "neuron.time_constant": [15e-3, 20e-3, 25e-3],
    }

    # Create and run the experiment
    experiment = GridSearchExperiment(
        base_config_file="correlated.yaml",
        parameter_ranges=correlated_params,
        result_metrics=["firing_rate", "mean_weights", "weight_distributions"],
        experiment_name="correlated_grid_search",
        random_seeds=[42, 43, 44, 45, 46],  # Multiple seeds to test robustness
    )

    # Run the experiment
    results = experiment.run()

    # Analyze the results
    experiment.analyze()

    print("Grid search complete. Results and analysis saved to", experiment.output_dir)
