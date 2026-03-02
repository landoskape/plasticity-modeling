# Hofer Reconstruction

The Hofer reconstruction simulation uses orientation-tuned inputs to model co-axial connectivity patterns observed experimentally, investigating how compartment-dependent plasticity affects orientation selectivity.

## Motivation

Iacaruso et al. (2017) showed that co-axial connections (aligned with a neuron's preferred orientation) are selectively located in distal dendrites. Our model asks: can compartment-dependent STDP, driven by AP attenuation, account for the observed patterns of spine tuning in vivo? 

## Gabor-Based Inputs

Input neurons have orientation preferences defined by Gabor-like tuning curves using the von Mises distribution:

\[
r(\theta) = r_\text{base} + r_\text{driven} \cdot \exp\left(\kappa \cos(\theta - \theta_\text{pref})\right)
\]

where:

- \(\theta\) is the stimulus orientation
- \(\theta_\text{pref}\) is the neuron's preferred orientation
- \(\kappa\) is the concentration parameter (tuning width)
- \(r_\text{base}\) and \(r_\text{driven}\) are the baseline and driven firing rates

### Edge Probability

The `edge_probability` parameter controls the likelihood of visual edges appearing, which shapes how correlated these triplets of inputs are across the 3x3 pixel grid. 

## Orientation Selectivity Analysis

Post-simulation analysis measures how well the postsynaptic neuron develops orientation selectivity through STDP:

1. **Weight-orientation profiles** — How synaptic weights distribute across input orientations
2. **Selectivity indices** — Quantifying the degree of orientation tuning in the learned weights
3. **Co-axial bias** — Whether strengthened connections align with the preferred orientation axis

## Configuration

The simulation is configured via `configs/hofer.yaml`:

```bash
python scripts/iaf_hofer_reconstruction.py --config hofer --run_name test --duration 2400 --repeats 1
```

Variants include:

- `hofer_replacement.yaml` — Adds synapse replacement (structural plasticity)
- `hofer_all_proximal.yaml` — All-proximal control condition

## Implementation

- **Source population**: [`create_source_population()`](../api/iaf/source-population.md) with `SourceGaborConfig`
- **Experiment factory**: [`get_experiment("hofer")`](../api/iaf/experiments.md)
- **Analysis**: [`gather_results()`](../api/iaf/analysis.md) for orientation selectivity metrics
