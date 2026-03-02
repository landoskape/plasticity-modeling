# Experimental Data

The model is calibrated and validated against experimental datasets measuring synaptic plasticity at different dendritic locations.

## eLife Dataset

The primary experimental reference is an eLife publication providing measurements of plasticity at identified dendritic sites. The data includes:

- **Plasticity magnitude** at individual synapses
- **Dendritic location** (distance from soma)
- **Site classification** — proximal vs distal, simple vs complex dendrite morphology

### Site Classification

Dendritic sites are classified using the `DendriticSiteParams` dataclass, which defines thresholds for:

- **Distance from soma** — separating proximal and distal compartments
- **Branch complexity** — distinguishing simple (single branch) from complex (branching) dendrites

### Data Loading

```python
from src.experimental import get_elife_data

data = get_elife_data()
```

This loads the MATLAB data from the eLife paper's Figure 2, performing necessary corrections (e.g., PMT closure correction via `correct_pmt()`).

## Nevian-Sakmann Calibration

AP amplitude measurements from Nevian & Sakmann provide the calibration data linking dendritic distance to AP attenuation. These measurements establish the relationship between:

- **Soma-to-dendrite distance** → **AP amplitude attenuation**
- **AP amplitude** → **D/P ratio** (via the conductance model)

This calibration chain connects the biophysical conductance model to the abstract STDP parameters used in the IAF simulations.

## Implementation

- **Data loading**: [`get_elife_data()`](../api/experimental.md) in `src/experimental.py`
- **Site classification**: [`DendriticSiteParams`](../api/experimental.md) dataclass
- **PMT correction**: [`correct_pmt()`](../api/experimental.md) function
