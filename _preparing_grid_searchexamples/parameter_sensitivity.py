from src.iaf.experiments import parameter_grid_search
from src.iaf.simulation import Simulation
from src.iaf.config import SimulationConfig
from src.files import config_dir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from datetime import datetime


def run_parameter_sensitivity_analysis(base_config_file, parameter_ranges, metrics=None, n_seeds=5, output_dir=None):
    """
    Run a parameter sensitivity analysis by varying each parameter individually
    from baseline.

    Args:
        base_config_file: Base YAML configuration file (e.g., "correlated.yaml")
        parameter_ranges: Dict of parameters to test with ranges as [min, max, n_steps]
        metrics: List of metrics to track
        n_seeds: Number of random seeds to test for each parameter setting
        output_dir: Output directory for results

    Returns:
        DataFrame with sensitivity analysis results
    """
    # Initialize outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_dir or "results") / f"sensitivity_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "plots", exist_ok=True)

    # Set up metrics
    metrics = metrics or ["firing_rate", "mean_weights"]

    # Load base configuration
    base_config_path = config_dir() / base_config_file
    base_config = SimulationConfig.from_yaml(base_config_path)

    # Generate random seeds
    seeds = np.random.randint(0, 10000, size=n_seeds)

    # Collect all results
    all_results = []

    # Baseline run (with all default parameters)
    baseline_results = run_baseline_simulation(base_config, metrics, seeds)
    baseline_results["parameter"] = "baseline"
    baseline_results["relative_value"] = 1.0
    all_results.append(baseline_results)

    # Run sensitivity analysis for each parameter
    for param_name, param_range in parameter_ranges.items():
        # Get baseline value for this parameter
        baseline_value = get_parameter_value(base_config, param_name)

        # Generate parameter values to test
        min_val, max_val, n_steps = param_range
        if n_steps == 1:
            values = [min_val]  # Just one test value
        else:
            values = np.linspace(min_val, max_val, n_steps)

        # Skip baseline value if it's in the range
        values = [v for v in values if abs(v - baseline_value) > 1e-10]

        print(f"Testing parameter: {param_name}")
        print(f"  Baseline value: {baseline_value}")
        print(f"  Test values: {values}")

        for value in values:
            # Run simulations with this parameter value
            relative_value = value / baseline_value if baseline_value != 0 else float("inf")
            param_results = run_parameter_variation(base_config, param_name, value, metrics, seeds)
            param_results["parameter"] = param_name
            param_results["value"] = value
            param_results["relative_value"] = relative_value
            all_results.append(param_results)

    # Combine all results
    results_df = pd.concat(all_results, ignore_index=True)

    # Save to CSV
    results_df.to_csv(output_dir / "sensitivity_results.csv", index=False)

    # Generate analysis plots
    plot_sensitivity_results(results_df, output_dir)

    return results_df


def run_baseline_simulation(config, metrics, seeds):
    """Run baseline simulation with default parameters."""
    results = {metric: [] for metric in metrics}
    results["seed"] = []

    for seed in seeds:
        # Create a copy of the config with this seed
        sim_config = config.model_copy(deep=True)
        sim_config.seed = int(seed)

        # Run simulation
        simulation = Simulation.from_config(sim_config)
        simulation.run(duration=sim_config.duration)

        # Calculate metrics
        results["seed"].append(seed)
        for metric in metrics:
            if metric == "firing_rate":
                results[metric].append(simulation.neuron.output_spike_count / sim_config.duration)
            elif metric == "mean_weights":
                # Add mean weight for each synapse group
                for synapse_name, synapse in simulation.synapses.items():
                    metric_name = f"mean_weight_{synapse_name}"
                    if metric_name not in results:
                        results[metric_name] = []
                    results[metric_name].append(np.mean(synapse.weights))

    # Convert to DataFrame and calculate statistics
    df = pd.DataFrame(results)

    # Calculate statistics across seeds
    stats = {}
    for col in df.columns:
        if col != "seed":
            stats[f"{col}_mean"] = df[col].mean()
            stats[f"{col}_std"] = df[col].std()
            stats[f"{col}_cv"] = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0

    return stats


def run_parameter_variation(config, param_name, param_value, metrics, seeds):
    """Run simulations with a specific parameter value."""
    results = {metric: [] for metric in metrics}
    results["seed"] = []

    for seed in seeds:
        # Create a copy of the config with this seed
        sim_config = config.model_copy(deep=True)
        sim_config.seed = int(seed)

        # Set the parameter value
        set_parameter_value(sim_config, param_name, param_value)

        # Run simulation
        simulation = Simulation.from_config(sim_config)
        simulation.run(duration=sim_config.duration)

        # Calculate metrics
        results["seed"].append(seed)
        for metric in metrics:
            if metric == "firing_rate":
                results[metric].append(simulation.neuron.output_spike_count / sim_config.duration)
            elif metric == "mean_weights":
                # Add mean weight for each synapse group
                for synapse_name, synapse in simulation.synapses.items():
                    metric_name = f"mean_weight_{synapse_name}"
                    if metric_name not in results:
                        results[metric_name] = []
                    results[metric_name].append(np.mean(synapse.weights))

    # Convert to DataFrame and calculate statistics
    df = pd.DataFrame(results)

    # Calculate statistics across seeds
    stats = {}
    for col in df.columns:
        if col != "seed":
            stats[f"{col}_mean"] = df[col].mean()
            stats[f"{col}_std"] = df[col].std()
            stats[f"{col}_cv"] = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0

    return stats


def get_parameter_value(config, param_path):
    """Get the current value of a parameter from its path."""
    current = config
    parts = param_path.split(".")

    for part in parts[:-1]:
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ValueError(f"Cannot find {part} in {param_path}")

    last_part = parts[-1]
    if hasattr(current, last_part):
        return getattr(current, last_part)
    elif isinstance(current, dict) and last_part in current:
        return current[last_part]
    else:
        raise ValueError(f"Cannot find {last_part} in {param_path}")


def set_parameter_value(config, param_path, value):
    """Set a parameter value from its path."""
    current = config
    parts = param_path.split(".")

    for part in parts[:-1]:
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ValueError(f"Cannot find {part} in {param_path}")

    last_part = parts[-1]
    if hasattr(current, last_part):
        setattr(current, last_part, value)
    elif isinstance(current, dict) and last_part in current:
        current[last_part] = value
    else:
        raise ValueError(f"Cannot find {last_part} in {param_path}")


def plot_sensitivity_results(results_df, output_dir):
    """Generate plots analyzing parameter sensitivity."""
    # Get all metrics (filtering out seed, parameter, value columns)
    metric_cols = [
        col
        for col in results_df.columns
        if col not in ["seed", "parameter", "value", "relative_value"]
        and not col.endswith("_std")
        and not col.endswith("_cv")
    ]

    # 1. Parameter sensitivity barchart
    for metric in metric_cols:
        # Get baseline value
        baseline = results_df[results_df["parameter"] == "baseline"][metric].values[0]

        # Filter out baseline row for plotting
        df_plot = results_df[results_df["parameter"] != "baseline"].copy()

        # Calculate change from baseline
        df_plot[f"{metric}_change"] = (df_plot[metric] - baseline) / baseline * 100

        # Create a colormap based on relative values
        norm = plt.Normalize(df_plot["relative_value"].min(), df_plot["relative_value"].max())
        cmap = plt.cm.coolwarm

        # Plot
        plt.figure(figsize=(12, 8))

        # Group by parameter and calculate the max absolute change for sorting
        param_max_change = (
            df_plot.groupby("parameter")[f"{metric}_change"].apply(lambda x: max(abs(x))).sort_values(ascending=False)
        )

        # Get the top parameters by impact
        top_params = param_max_change.index[:10]  # Limit to top 10 for clarity

        # Filter to top parameters
        df_top = df_plot[df_plot["parameter"].isin(top_params)]

        # Create plot
        g = sns.barplot(
            data=df_top,
            x="parameter",
            y=f"{metric}_change",
            hue="relative_value",
            palette=sns.color_palette("coolwarm", n_colors=len(df_top)),
        )
        plt.title(f"Sensitivity Analysis: % Change in {metric}")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(f"% Change from Baseline")
        plt.xlabel("Parameter")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / f"sensitivity_{metric}.png")
        plt.close()

        # 2. Create tornado chart for this metric (showing range of changes)
        df_range = df_plot.groupby("parameter").agg({f"{metric}_change": ["min", "max"]}).reset_index()
        df_range.columns = ["parameter", "min_change", "max_change"]

        # Sort by maximum absolute change
        df_range["abs_max"] = df_range[["min_change", "max_change"]].abs().max(axis=1)
        df_range = df_range.sort_values("abs_max", ascending=True)

        # Filter to top parameters
        df_range = df_range.tail(10)  # Top 10

        # Plot tornado chart
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(df_range))

        # Plot bars
        plt.barh(y_pos, df_range["max_change"], height=0.8, color="red", alpha=0.6, label="Increase")
        plt.barh(y_pos, df_range["min_change"], height=0.8, color="blue", alpha=0.6, label="Decrease")

        # Add parameter names
        plt.yticks(y_pos, df_range["parameter"])
        plt.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

        plt.title(f"Tornado Chart: Parameter Sensitivity for {metric}")
        plt.xlabel(f"% Change from Baseline")
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / f"tornado_{metric}.png")
        plt.close()


if __name__ == "__main__":
    # Define parameters to test with their ranges
    # Format: parameter_name: [min_value, max_value, n_steps]
    correlated_params = {
        # Input parameters
        "sources.excitatory.max_correlation": [0.2, 0.6, 3],
        "sources.excitatory.rate_mean": [10.0, 30.0, 3],
        # Plasticity parameters
        "synapses.basal.plasticity.depression_potentiation_ratio": [0.9, 1.3, 3],
        "synapses.apical.plasticity.depression_potentiation_ratio": [0.8, 1.2, 3],
        "synapses.basal.plasticity.stdp_rate": [0.005, 0.02, 3],
        "synapses.apical.plasticity.stdp_rate": [0.005, 0.02, 3],
        # Neuron parameters
        "neuron.time_constant": [15e-3, 25e-3, 3],
        "neuron.resistance": [80e6, 120e6, 3],
        "neuron.homeostasis_set_point": [15.0, 25.0, 3],
    }

    # Run the sensitivity analysis
    results = run_parameter_sensitivity_analysis(
        base_config_file="correlated.yaml",
        parameter_ranges=correlated_params,
        metrics=["firing_rate", "mean_weights"],
        n_seeds=5,
    )

    print("Sensitivity analysis complete. Results saved to results/sensitivity_* directory.")
