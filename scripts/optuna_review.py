from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path
import textwrap

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
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_params}")

    trials_df = study.trials_dataframe()
    if trials_df.empty:
        print("No trials found.")
        return

    sorted_df = trials_df.sort_values("value", ascending=True)
    top_df = sorted_df.head(args.top_n)
    print("\nTop trials (lowest value first):")
    if args.compact:
        for _, row in top_df.iterrows():
            params = {k.replace("params_", ""): row[k] for k in top_df.columns if k.startswith("params_")}
            value = row.get("value", None)
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
        print(top_df[display_cols].to_string(index=False))

    summary = trials_df["value"].describe()
    print("\nValue summary:")
    print(summary.to_string())


if __name__ == "__main__":
    main()
