"""
Reproducibility pipeline for plasticity-modeling.

Runs all steps needed to reproduce manuscript results from scratch:
  python run_pipeline.py                            # everything
  python run_pipeline.py --steps figures            # just figures
  python run_pipeline.py --steps conductance        # just conductance
  python run_pipeline.py --steps correlation hofer  # just IAF simulations
  python run_pipeline.py --force                    # re-run even if outputs exist
  python run_pipeline.py --config manuscript.yaml --steps figures  # use manuscript data
"""

from __future__ import annotations

import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import yaml

ALL_STEPS = ["conductance", "correlation", "hofer", "figures"]

ROOT = Path(__file__).resolve().parent


def load_config(config_path: str = "pipeline.yaml") -> dict:
    """Load config from path."""
    if not config_path.endswith(".yaml"):
        config_path = config_path + ".yaml"
    config_path = ROOT / config_path
    config_path = ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config


def run_cmd(cmd: list[str], description: str) -> None:
    """Run a command, printing it and streaming output."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"  $ {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nERROR: Step failed with return code {result.returncode}")
        sys.exit(result.returncode)


def do_step_conductance(config: dict, force: bool, peek: bool = False) -> None:
    """Run conductance simulations."""
    cfg = config.get("conductance", {})
    output_path = ROOT / "results" / "conductance_runs.joblib"

    if output_path.exists() and not force:
        print(f"[conductance] Skipping — {output_path} already exists")
        return
    elif peek:
        print(f"[conductance] Would run — {output_path} does not exist")
        return

    cmd = [
        sys.executable,
        "scripts/conductance_data.py",
        "--num_ap_amplitudes",
        str(cfg.get("num_ap_amplitudes", 400)),
    ]
    run_cmd(cmd, "Running conductance simulations")


def do_step_correlation(config: dict, force: bool, peek: bool = False) -> None:
    """Run correlated IAF simulations (example + full)."""
    cfg = config.get("correlation", {})
    corr_config = cfg.get("config", "correlated")

    # Example run
    example_name = cfg.get("example_run_name", "correlated_example")
    example_dir = ROOT / "results" / "iaf_runs" / corr_config / example_name
    if example_dir.exists() and not force:
        print(f"[correlation/example] Skipping — {example_dir} already exists")
    elif peek:
        print(f"[correlation/example] Would run — {example_dir} does not exist")
    else:
        cmd = [
            sys.executable,
            "scripts/iaf_correlation.py",
            "--config",
            corr_config,
            "--run_name",
            example_name,
            "--duration",
            str(cfg.get("example_duration", 2400)),
            "--repeats",
            str(cfg.get("example_repeats", 1)),
        ]
        if cfg.get("example_save_source_rates", False):
            cmd.append("--save_source_rates")
        run_cmd(cmd, f"Running correlation example simulation ({example_name})")

    # Full run
    full_name = cfg.get("full_run_name", "correlated")
    full_dir = ROOT / "results" / "iaf_runs" / corr_config / full_name
    if full_dir.exists() and not force:
        print(f"[correlation/full] Skipping — {full_dir} already exists")
    elif peek:
        print(f"[correlation/full] Would run — {full_dir} does not exist")
    else:
        cmd = [
            sys.executable,
            "scripts/iaf_correlation.py",
            "--config",
            corr_config,
            "--run_name",
            full_name,
            "--duration",
            str(cfg.get("full_duration", 9600)),
            "--repeats",
            str(cfg.get("full_repeats", 10)),
        ]
        run_cmd(cmd, f"Running correlation full simulation ({full_name})")


def do_step_hofer(config: dict, force: bool, peek: bool = False) -> None:
    """Run Hofer reconstruction simulations."""
    cfg = config.get("hofer", {})
    hofer_config = cfg.get("config", "hofer")
    run_name = cfg.get("run_name", "hofer")
    run_dir = ROOT / "results" / "iaf_runs" / hofer_config / run_name

    if run_dir.exists() and not force:
        print(f"[hofer] Skipping — {run_dir} already exists")
        return
    elif peek:
        print(f"[hofer] Would run — {run_dir} does not exist")
        return

    cmd = [
        sys.executable,
        "scripts/iaf_hofer_reconstruction.py",
        "--config",
        hofer_config,
        "--run_name",
        run_name,
        "--duration",
        str(cfg.get("duration", 9600)),
        "--repeats",
        str(cfg.get("repeats", 10)),
    ]
    run_cmd(cmd, f"Running Hofer reconstruction simulation ({run_name})")


def do_step_figures(config: dict, force: bool, peek: bool = False) -> None:
    """Generate manuscript figures."""
    cfg = config.get("figures", {})
    corr_cfg = config.get("correlation", {})
    hofer_cfg = config.get("hofer", {})

    cmd = [
        sys.executable,
        "scripts/make_figures.py",
        "--mode",
        cfg.get("mode", "save"),
        "--correlated-example-run",
        corr_cfg.get("example_run_name", "correlated_example"),
        "--correlated-full-run",
        corr_cfg.get("full_run_name", "correlated"),
        "--hofer-run",
        hofer_cfg.get("run_name", "hofer"),
    ]

    if peek:
        print("[figures] Would run — generating manuscript figures")
        return

    run_cmd(cmd, "Generating manuscript figures")


STEP_FUNCTIONS = {
    "conductance": do_step_conductance,
    "correlation": do_step_correlation,
    "hofer": do_step_hofer,
    "figures": do_step_figures,
}


def main():
    parser = ArgumentParser(description="Run the full reproducibility pipeline.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        help="Which pipeline steps to run. Default: all.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pipeline.yaml",
        help="Path to a config YAML (default: pipeline.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if outputs already exist.",
    )
    parser.add_argument(
        "--peek",
        action="store_true",
        help="Show what each selected step would run/skip, then exit without running.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("Plasticity Modeling — Reproducibility Pipeline")
    print(f"Steps: {', '.join(args.steps)}")
    print(f"Force: {args.force}")
    print(f"Peek: {args.peek}")

    for step_name in args.steps:
        STEP_FUNCTIONS[step_name](config, args.force, args.peek)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
