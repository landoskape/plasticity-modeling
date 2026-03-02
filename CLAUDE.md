# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Computational neuroscience project modeling synaptic plasticity. It simulates how synaptic weights evolve under STDP (spike-timing-dependent plasticity) rules in integrate-and-fire neurons, then generates manuscript figures from the simulation results. The three main modeling components are:

1. **Conductance model** — biophysical VGCC/calcium model predicting how AP amplitude affects plasticity (`src/conductance.py`)
2. **Correlated input simulations** — IAF neuron with correlated Poisson inputs and STDP learning across proximal/distal compartments (`configs/correlated.yaml`)
3. **Hofer reconstruction** — IAF neuron with Gabor-based orientation-tuned inputs modeling co-axial connectivity (`configs/hofer.yaml`)

## Build & Run

```bash
conda activate plasticity-modeling  # activate environment first
pip install -e .                    # editable install (required)
pip install -e .[cpu]               # includes PyTorch (optional, for classical Hebbian models)
python run_pipeline.py              # full pipeline: conductance → correlation → hofer → figures
python run_pipeline.py --steps figures
python run_pipeline.py --config manuscript.yaml --steps figures  # use published run names
```

## Formatting

Black with line length 120: `python -m black path/to/file.py`

No linter or test framework is configured.

## Architecture

### Pipeline system (`run_pipeline.py`)

Orchestrates all steps. Config is loaded with 3-layer merge: `pipeline.yaml` (base) → `--config` override → `pipeline_local.yaml` (gitignored, highest priority). Each step checks if output exists before running (skip with `--force`).

### Simulation core (`src/iaf/`)

- `config.py` — Pydantic v2 models (`SimulationConfig`, `NeuronConfig`, `SynapseGroupConfig`, etc.) loaded from YAML via `BaseConfig.from_yaml`. All simulation parameters are validated here.
- `simulation.py` — `Simulation` class wires together source populations, an IAF neuron, and synapse groups. Created from config via `Simulation.from_config()`.
- `experiments.py` — `get_experiment()` is the main entry point for creating a configured simulation from a named config with parameter overrides (d/p ratio, edge probability, etc.).
- `iaf_neuron.py` — The integrate-and-fire neuron model.
- `source_population.py` — Input spike generators (Poisson, correlated, Gabor-based).
- `synapse_group.py` — Synapse groups with STDP plasticity, homeostasis, and weight dynamics.
- `analysis.py` — Post-simulation analysis (weight trajectories, tuning curves, etc.).

### Simulation scripts (`scripts/`)

- `iaf_correlation.py` / `iaf_hofer_reconstruction.py` — Run simulations with `--config`, `--run_name`, `--duration`, `--repeats`. Results go to `results/iaf_runs/{config_name}/{run_name}/`.
- `conductance_data.py` — Runs biophysical conductance simulations, outputs to `data/conductance_runs.joblib`.
- `make_figures.py` — Generates Figures 1–6. Each figure has a `FigureNParams` dataclass and a `figureN()` function. Run names can be overridden via CLI flags (`--correlated-full-run`, `--hofer-run`).

### Shared utilities

- `src/files.py` — Path helpers (`root_dir()`, `config_dir()`, `data_dir()`, `results_dir()`, `get_figure_dir()`). All paths are relative to repo root.
- `src/plotting.py` — `FigParams` (global style constants), `save_figure()`, synapse group color schemes (`Proximal`, `DistalSimple`, `DistalComplex`), formatting helpers.
- `src/schematics.py` — Programmatic neuron/schematic diagrams for figures.
- `src/experimental.py` — Loads experimental data (eLife paper results).

### Configuration flow

YAML configs in `configs/` define full simulation parameters (neuron, sources, synapses, plasticity rules). The typical flow is:
1. `get_experiment("correlated")` loads `configs/correlated.yaml` → `SimulationConfig`
2. `Simulation.from_config(config)` builds the simulation objects
3. `simulation.run(duration)` executes and returns results
4. Results are saved per-repeat as joblib files in `results/iaf_runs/`

## Conventions

- Use `from __future__ import annotations` in all modules.
- Modern typing: `list[...]`, `dict[...]`, `X | None` (not `Optional[X]`).
- Pydantic v2 style: `.model_validate()`, `.model_dump()`.
- Numpy-style docstrings for public functions.
- Use `pathlib.Path` for all file paths; use `src/files.py` helpers for standard directories.
