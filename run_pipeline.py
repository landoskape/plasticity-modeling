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


def load_config(config_override: str | None = None) -> dict:
    """Load pipeline.yaml, optionally overlay a --config file, then pipeline_local.yaml."""
    config_path = ROOT / "pipeline.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    # Overlay --config file if provided
    if config_override is not None:
        override_path = Path(config_override)
        if not override_path.is_absolute():
            override_path = ROOT / override_path
        if not override_path.exists():
            raise FileNotFoundError(f"Config override not found: {override_path}")
        with open(override_path) as f:
            override = yaml.safe_load(f) or {}
        for key, val in override.items():
            if isinstance(val, dict) and key in config and isinstance(config[key], dict):
                config[key].update(val)
            else:
                config[key] = val

    # Overlay pipeline_local.yaml if present (highest priority)
    local_path = ROOT / "pipeline_local.yaml"
    if local_path.exists():
        with open(local_path) as f:
            local = yaml.safe_load(f) or {}
        # Shallow merge per top-level key
        for key, val in local.items():
            if isinstance(val, dict) and key in config and isinstance(config[key], dict):
                config[key].update(val)
            else:
                config[key] = val

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


def step_conductance(config: dict, force: bool) -> None:
    """Run conductance simulations."""
    cfg = config.get("conductance", {})
    output_path = ROOT / "data" / "conductance_runs.joblib"

    if output_path.exists() and not force:
        print(f"[conductance] Skipping — {output_path} already exists")
        return

    cmd = [
        sys.executable, "scripts/conductance_data.py",
        "--num_ap_amplitudes", str(cfg.get("num_ap_amplitudes", 400)),
    ]
    run_cmd(cmd, "Running conductance simulations")


def step_correlation(config: dict, force: bool) -> None:
    """Run correlated IAF simulations (example + full)."""
    cfg = config.get("correlation", {})
    corr_config = cfg.get("config", "correlated")

    # Example run
    example_name = cfg.get("example_run_name", "correlated_example")
    example_dir = ROOT / "results" / "iaf_runs" / corr_config / example_name
    if example_dir.exists() and not force:
        print(f"[correlation/example] Skipping — {example_dir} already exists")
    else:
        cmd = [
            sys.executable, "scripts/iaf_correlation.py",
            "--config", corr_config,
            "--run_name", example_name,
            "--duration", str(cfg.get("example_duration", 2400)),
            "--repeats", str(cfg.get("example_repeats", 1)),
        ]
        if cfg.get("example_save_source_rates", False):
            cmd.append("--save_source_rates")
        run_cmd(cmd, f"Running correlation example simulation ({example_name})")

    # Full run
    full_name = cfg.get("full_run_name", "correlated")
    full_dir = ROOT / "results" / "iaf_runs" / corr_config / full_name
    if full_dir.exists() and not force:
        print(f"[correlation/full] Skipping — {full_dir} already exists")
    else:
        cmd = [
            sys.executable, "scripts/iaf_correlation.py",
            "--config", corr_config,
            "--run_name", full_name,
            "--duration", str(cfg.get("full_duration", 9600)),
            "--repeats", str(cfg.get("full_repeats", 10)),
        ]
        run_cmd(cmd, f"Running correlation full simulation ({full_name})")


def step_hofer(config: dict, force: bool) -> None:
    """Run Hofer reconstruction simulations."""
    cfg = config.get("hofer", {})
    hofer_config = cfg.get("config", "hofer")
    run_name = cfg.get("run_name", "hofer")
    run_dir = ROOT / "results" / "iaf_runs" / hofer_config / run_name

    if run_dir.exists() and not force:
        print(f"[hofer] Skipping — {run_dir} already exists")
        return

    cmd = [
        sys.executable, "scripts/iaf_hofer_reconstruction.py",
        "--config", hofer_config,
        "--run_name", run_name,
        "--duration", str(cfg.get("duration", 9600)),
        "--repeats", str(cfg.get("repeats", 10)),
    ]
    run_cmd(cmd, f"Running Hofer reconstruction simulation ({run_name})")


def step_figures(config: dict, force: bool) -> None:
    """Generate manuscript figures."""
    cfg = config.get("figures", {})
    corr_cfg = config.get("correlation", {})
    hofer_cfg = config.get("hofer", {})

    cmd = [
        sys.executable, "scripts/make_figures.py",
        "--mode", cfg.get("mode", "save"),
        "--correlated-example-run", corr_cfg.get("example_run_name", "correlated_example"),
        "--correlated-full-run", corr_cfg.get("full_run_name", "correlated"),
        "--hofer-run", hofer_cfg.get("run_name", "hofer"),
    ]
    run_cmd(cmd, "Generating manuscript figures")


STEP_FUNCTIONS = {
    "conductance": step_conductance,
    "correlation": step_correlation,
    "hofer": step_hofer,
    "figures": step_figures,
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
        default=None,
        help="Path to a config YAML that overrides pipeline.yaml (e.g. manuscript.yaml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run even if outputs already exist.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    print("Plasticity Modeling — Reproducibility Pipeline")
    print(f"Steps: {', '.join(args.steps)}")
    print(f"Force: {args.force}")

    for step_name in args.steps:
        STEP_FUNCTIONS[step_name](config, args.force)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
