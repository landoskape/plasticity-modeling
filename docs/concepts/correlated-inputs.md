# Correlated Inputs

The correlated input simulation demonstrates how compartment-specific AP attenuation drives divergent weight evolution at proximal and distal synapses.

## Input Structure

Excitatory inputs are generated as correlated Poisson processes. Each source neuron fires with a rate drawn from a normal distribution, and spike trains share pairwise correlations that decay with source neuron distance:

- **Number of source neurons**: 40
- **Maximum correlation**: 0.4
- **Decay function**: linear with source distance
- **Rate distribution**: mean 20 Hz, std 10 Hz

Inhibitory inputs are independent Poisson processes (200 neurons at 10 Hz) providing a stabilizing background.

## Synapse Compartments

The simulation defines three excitatory synapse groups connected to the same source population:

| Group | Location | Synapses | D/P Ratio | Description |
|-------|----------|----------|-----------|-------------|
| **Proximal** | Near soma | 1000 | 1.1 | Large AP → mild net depression |
| **Distal-simple** | Far from soma | 40 | 1.1 | Same D/P as proximal (control) |
| **Distal-complex** | Far from soma | 40 | 1.0 | Attenuated AP → balanced plasticity |

The key manipulation is the D/P ratio: proximal and distal-simple synapses share the same ratio (1.1), while distal-complex synapses have a balanced ratio (1.0) reflecting greater AP attenuation.

## Weight Divergence

Over thousands of seconds of simulated time, the different D/P ratios cause weight distributions to diverge:

- **Proximal** synapses: competitive dynamics under mild net depression, weights become sparse and selective
- **Distal-simple** synapses: similar to proximal but with fewer synapses
- **Distal-complex** synapses: balanced plasticity leads to different weight distributions, demonstrating the effect of AP attenuation

## Configuration

The simulation is configured via `configs/correlated.yaml` and run by:

```bash
python scripts/iaf_correlation.py --config correlated --run_name test --duration 2400 --repeats 1
```

See the [Configuration](../getting-started/configuration.md) page for details on the YAML structure.

## Implementation

- **Source population**: [`create_source_population()`](../api/iaf/source-population.md) with `SourceCorrelationConfig`
- **Experiment factory**: [`get_experiment("correlated")`](../api/iaf/experiments.md)
- **Analysis**: [`gather_results()`](../api/iaf/analysis.md) for post-simulation analysis
