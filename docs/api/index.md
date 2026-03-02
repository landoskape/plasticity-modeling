# API Reference

Auto-generated documentation from source code docstrings.

## Core Modules

- **[Conductance](conductance.md)** — Biophysical VGCC/calcium model (`src.conductance`)

## IAF Simulation Package

The `src.iaf` package contains the integrate-and-fire neuron simulation framework:

- **[Configuration](iaf/config.md)** — Pydantic v2 config models for simulations
- **[Simulation](iaf/simulation.md)** — Main simulation orchestrator
- **[Experiments](iaf/experiments.md)** — Factory functions for creating configured simulations
- **[Neuron](iaf/neuron.md)** — Integrate-and-fire neuron model
- **[Source Populations](iaf/source-population.md)** — Input spike generators
- **[Synapse Groups](iaf/synapse-group.md)** — Synapse models with STDP
- **[Analysis](iaf/analysis.md)** — Post-simulation analysis utilities

## Utilities

- **[File Helpers](files.md)** — Path utilities for the repository (`src.files`)
- **[Plotting](plotting.md)** — Figure styling and save utilities (`src.plotting`)
- **[Experimental Data](experimental.md)** — Experimental data loading (`src.experimental`)
- **[Schematics](schematics.md)** — Programmatic neuron diagrams (`src.schematics`)
