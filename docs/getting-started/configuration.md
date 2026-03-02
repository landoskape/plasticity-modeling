# Configuration

## Pipeline Configuration

The pipeline uses a **3-layer configuration merge**, with each layer overriding the previous:

1. **`pipeline.yaml`** (base) — Default settings checked into the repository
2. **`--config` override** — Optional YAML passed via CLI (e.g., `manuscript.yaml`)
3. **`pipeline_local.yaml`** (highest priority) — Local overrides, gitignored

### Default pipeline config

The base `pipeline.yaml` defines all pipeline parameters:

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

### Local overrides

Create a `pipeline_local.yaml` in the repo root to point the pipeline at your own simulation runs:

```yaml
correlation:
  full_run_name: "my_full_run"
hofer:
  run_name: "my_hofer_run"
```

This file is gitignored. The pipeline skips simulation steps if the named output directories already exist under `results/iaf_runs/`.

## Simulation Configs

Simulation parameters are defined in YAML files under `configs/`. These control the neuron model, input sources, synapse groups, and plasticity rules.

### Available configs

| Config | File | Description |
|--------|------|-------------|
| `correlated` | `configs/correlated.yaml` | Correlated Poisson inputs, proximal/distal compartments |
| `hofer` | `configs/hofer.yaml` | Gabor-tuned inputs, orientation selectivity |
| `hofer_replacement` | `configs/hofer_replacement.yaml` | Hofer variant with synapse replacement |
| `hofer_all_proximal` | `configs/hofer_all_proximal.yaml` | Hofer variant, all-proximal control |

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
