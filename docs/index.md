# Plasticity Modeling

Computational neuroscience project modeling how **action potential amplitude** shapes **compartment-dependent synaptic plasticity**. This codebase accompanies our manuscript and reproduces all simulation results and figures.

## What This Project Does

The project combines three modeling components to investigate how dendritic location and AP back-propagation interact with spike-timing-dependent plasticity (STDP):

1. **Conductance model** — A biophysical VGCC/calcium model predicting how AP amplitude affects plasticity magnitude through voltage-gated calcium channels and NMDA receptors.

2. **Correlated input simulations** — An integrate-and-fire neuron receiving correlated Poisson inputs, with STDP learning at proximal and distal synaptic compartments, demonstrating weight divergence driven by AP attenuation.

3. **Hofer reconstruction** — An IAF neuron with Gabor-based orientation-tuned inputs, modeling co-axial connectivity to reproduce experimental findings on orientation selectivity and compartment-specific plasticity.

## Quick Start

```bash
git clone https://github.com/landoskape/plasticity-modeling
cd plasticity-modeling
conda create -n plasticity-modeling python=3.11
conda activate plasticity-modeling
pip install -e .
conda install numba
```

Run the full pipeline to reproduce all results and figures:

```bash
python run_pipeline.py
```

Or generate just the figures (if data already exists):

```bash
python run_pipeline.py --steps figures
```

See the [Installation](getting-started/installation.md) guide for detailed setup instructions and the [Pipeline Overview](getting-started/pipeline.md) for all available options.

## Project Structure

```
plasticity-modeling/
├── src/                    # Core library
│   ├── iaf/                # IAF neuron simulation package
│   ├── conductance.py      # Biophysical conductance model
│   ├── plotting.py         # Figure styling and utilities
│   └── ...
├── configs/                # Simulation YAML configs
├── scripts/                # Pipeline step scripts
├── pipeline.yaml           # Default pipeline configuration
└── run_pipeline.py         # Main entry point
```
