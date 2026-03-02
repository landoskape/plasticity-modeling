# Configuration

## Pipeline Configuration

The base `pipeline.yaml` defines all pipeline parameters for re-running simulations and making figures.

```yaml
conductance:
  num_ap_amplitudes: 400

correlation:
  config: "correlated"
  example_run_name: "correlated_example"
  example_duration: 2400
  example_repeats: 1
  full_run_name: "correlated"
  full_duration: 9600
  full_repeats: 10

hofer:
  config: "hofer"
  run_name: "hofer"
  duration: 9600
  repeats: 10

figures:
  mode: "save"
```

### Manuscript configuration

The config called `manusript.yaml` contains the `*_run_name`s for the data used in the manuscript. Make sure to download the data from online (it will be available upon submission). This ignores the other pipeline details since they aren't used when simulation data already exists.  

## Simulation Configs

Simulation parameters are defined in YAML files under `configs/`. These control the neuron model, input sources, synapse groups, and plasticity rules.

### Main configs

| Config | File | Description |
|--------|------|-------------|
| `correlated` | `configs/correlated.yaml` | Correlated Poisson inputs, proximal/distal compartments |
| `hofer` | `configs/hofer.yaml` | Gabor-tuned inputs, orientation selectivity |

There are more configs, which were used for development and testing of alternate models. The two listed above were the ones used in the manuscript.

### Config structure

Each simulation config defines:

- **`neuron`** — IAF neuron parameters (time constant, resistance, thresholds, homeostasis)
- **`sources`** — Input population definitions (type, rates, correlation structure)
- **`synapses`** — Synapse group parameters (weights, plasticity rules, STDP settings)

Configs are loaded via Pydantic validation:

```python
from src.iaf.experiments import get_experiment

sim, config = get_experiment("correlated")
```

See the [API Reference](../api/iaf/config.md) for full details on all configuration fields.
