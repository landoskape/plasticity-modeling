from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import textwrap

import numpy as np
import optuna
import pandas as pd


def get_args():
    parser = ArgumentParser(description="Review an Optuna study in a run directory.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to the Optuna run directory.")
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Override the study name if it differs from run_metadata.json.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="How many top trials to display.",
    )
    parser.add_argument(
        "--trial-number",
        type=int,
        default=None,
        help="If set, print details for a specific trial number.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Print compact one-line summaries instead of a wide table.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=180,
        help="Max line width for compact output.",
    )
    return parser.parse_args()


def _load_study_name(run_dir: Path, override: str | None) -> str:
    if override:
        return override
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return metadata["study_name"]
    raise FileNotFoundError("run_metadata.json not found and --study-name not provided")


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _format_literal(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def main() -> None:
    args = get_args()
    run_dir = args.run_dir
    study_name = _load_study_name(run_dir, args.study_name)
    db_path = (run_dir / "optuna.db").resolve()
    storage_url = f"sqlite:///{db_path.as_posix()}"

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    print(f"Study: {study.study_name}")
    print(f"Storage: {storage_url}")
    print(f"Trials: {len(study.trials)}")
    trials_df = study.trials_dataframe()
    if trials_df.empty:
        print("No trials found.")
        return

    if args.trial_number is not None:
        trial = study.trials[args.trial_number]
        print("\nTrial details:")
        print(f"number: {trial.number}")
        print(f"state: {trial.state}")
        print(f"value: {_format_value(trial.value)}")
        print(f"datetime_start: {trial.datetime_start}")
        print(f"datetime_complete: {trial.datetime_complete}")
        formatted_params = {key: _format_value(value) for key, value in trial.params.items()}
        print(f"params: {formatted_params}")
        param_literals = {key: _format_literal(value) for key, value in trial.params.items()}
        get_experiment_args = (
            "get_proximal_experiment(\"hofer_all_proximal\", "
            f"num_synapses={param_literals.get('num_synapses')}, "
            f"max_weight={param_literals.get('max_weight')}, "
            f"conductance_threshold={param_literals.get('conductance_threshold')}, "
            f"independent_noise_rate={param_literals.get('independent_noise_rate')}, "
            f"stdp_rate={param_literals.get('stdp_rate')}, "
            f"depression_potentiation_ratio={param_literals.get('depression_potentiation_ratio')}, "
            "num_simulations=1)"
        )
        print(f"get_proximal_experiment call: {get_experiment_args}")
        print("\n")
        if trial.user_attrs:
            formatted_user_attrs = {key: _format_value(value) for key, value in trial.user_attrs.items()}
            formatted_user_attrs.pop("avg_proximal_weights", None)
            formatted_user_attrs.pop("avg_spike_rate_hz", None)
            print(f"user_attrs: {formatted_user_attrs}")
        if trial.system_attrs:
            print(f"\n system_attrs: {trial.system_attrs}")
        if trial.user_attrs.get("avg_proximal_weights") is not None:
            print("\n avg_proximal_weights:")
            for ineuron, weights in enumerate(trial.user_attrs["avg_proximal_weights"]):
                weights = np.array(weights)
                weights = weights / np.max(weights)
                formatted_weights = [_format_value(value) for value in weights]
                print(f"\n neuron {ineuron}: {formatted_weights}")
        if trial.user_attrs.get("avg_spike_rate_hz") is not None:
            formatted_rates = [_format_value(value) for value in trial.user_attrs["avg_spike_rate_hz"]]
            print(f"\n avg_spike_rate_hz: {formatted_rates}")
        return

    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")

    sorted_df = trials_df.sort_values("value", ascending=True)
    top_df = sorted_df.head(args.top_n)
    print("\nTop trials (lowest value first):")
    if args.compact:
        for _, row in top_df.iterrows():
            params = {
                k.replace("params_", ""): _format_value(row[k]) for k in top_df.columns if k.startswith("params_")
            }
            value = _format_value(row.get("value", None))
            number = row.get("number", None)
            params_text = ", ".join([f"{key}={value}" for key, value in params.items()])
            line = f"trial={number} value={value} params: {params_text}"
            print(textwrap.shorten(line, width=args.max_width, placeholder="..."))
    else:
        columns = [
            "number",
            "state",
            "value",
            "datetime_start",
            "datetime_complete",
        ]
        param_cols = [col for col in trials_df.columns if col.startswith("params_")]
        user_cols = [col for col in trials_df.columns if col.startswith("user_attrs_")]
        display_cols = [col for col in columns if col in trials_df.columns] + param_cols + user_cols
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        formatted = top_df[display_cols].copy()
        for col in formatted.columns:
            if col.startswith("params_") or col.startswith("user_attrs_") or col == "value":
                formatted[col] = formatted[col].map(_format_value)
        print(formatted.to_string(index=False))

    summary = trials_df["value"].describe()
    print("\nValue summary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
