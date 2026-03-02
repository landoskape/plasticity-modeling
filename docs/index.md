# Plasticity Modeling

This is a computational neuroscience project modeling how **action potential amplitude** shapes **compartment-dependent synaptic plasticity** via it's asymmetric effect on potentiation and depression in **spike-timing dependent plasticity**. This codebase accompanies our manuscript and reproduces all simulation results and figures.

## What This Project Does

The project combines three modeling components to investigate how dendritic location and AP back-propagation interact with spike-timing-dependent plasticity (STDP):

1. **Conductance model** — A biophysical VGCC/calcium model predicting how AP amplitude affects plasticity magnitude through voltage-gated calcium channels and NMDA receptors. This shows how the differences in NMDARs and VGCCs voltage-dependence leads to a divergence in AP-evoked calcium release depending on the context. 

2. **Correlated input simulations** — An integrate-and-fire neuron receiving correlated Poisson inputs, with STDP learning at proximal and distal synaptic compartments. This demonstrates that the depression/potentiation ratio affects how correlated inputs must be to achieve stable tuning (with each other, or with the dominant input source driving postsynaptic spikes). We introduce our multicompartment STDP model and show local consequences of D/P ratio.

3. **Hofer reconstruction** — An IAF neuron with Gabor-based orientation-tuned inputs, modeling visual input tuning to reproduce compartment-specific dendritic spine tuning properties observed in vivo in visual cortex. This models the findings of [Iacaruso et al., 2017](https://www.nature.com/articles/nature23019). 

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
