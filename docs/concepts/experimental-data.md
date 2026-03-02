# Experimental Data

The model is calibrated and validated against experimental datasets measuring synaptic plasticity at different dendritic locations.

## eLife Dataset

The primary experimental reference is an eLife publication providing measurements of calcium influx at identified dendritic sites. The data includes:

- **Dendritic location** (distance from soma)
- **Calcium signals** (AP-evoked calcium influx, synaptic calcium influx via glutamate uncaging, pairing of both like in LTP protocols)
- **Site classification** — proximal vs distal, simple vs complex dendrite morphology

### Site Classification

Dendritic sites are classified using the `DendriticSiteParams` dataclass, which defines thresholds for:

- **Distance from soma** — separating proximal and distal compartments
- **AP-evoked calcium** — distinguishing high vs low calcium influx branches based on experimental measurements.
- **Branch complexity** — distinguishing simple (single branch) from complex (branching) dendrites
    Note: branch complexity was measured in some sites, but is inferred from AP-evoked calcium in most recordings. 

### Data Loading

```python
from src.experimental import get_elife_data

data = get_elife_data()
```

This loads the MATLAB data from the eLife paper's Figure 2.

## Implementation

- **Data loading**: [`get_elife_data()`](../api/experimental.md) in `src/experimental.py`
- **Site classification**: [`DendriticSiteParams`](../api/experimental.md) dataclass
