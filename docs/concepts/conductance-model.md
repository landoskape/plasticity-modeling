# Conductance Model

The conductance model predicts how action potential (AP) amplitude affects calcium influx through voltage-gated calcium channels (VGCCs) and NMDA receptors, providing the biophysical basis for compartment-dependent plasticity.

## Overview

When an AP back-propagates into dendrites, it activates VGCCs and relieves the Mg²⁺ block of NMDA receptors. The resulting calcium influx drives synaptic plasticity. Because APs attenuate along dendrites, distal synapses experience smaller voltage transients and therefore different calcium dynamics than proximal synapses.

## VGCC Model

The voltage-gated calcium channel model uses Hodgkin-Huxley-style gating:

\[
I_\text{VGCC} = m^2 \cdot h \cdot \text{GHK}(V, [\text{Ca}^{2+}]_i, [\text{Ca}^{2+}]_o)
\]

where:

- \(m\) is the activation gate (voltage-dependent, fast)
- \(h\) is the inactivation gate (voltage-dependent, slow)
- GHK is the Goldman-Hodgkin-Katz current equation accounting for calcium concentration gradients

### GHK Equation

The Goldman-Hodgkin-Katz equation for calcium current:

\[
I = z^2 \cdot \frac{F^2 V}{RT} \cdot \frac{[\text{Ca}^{2+}]_i - [\text{Ca}^{2+}]_o \exp(-zFV/RT)}{1 - \exp(-zFV/RT)}
\]

## NMDA Receptor Component

The NMDA receptor conductance includes voltage-dependent Mg²⁺ block:

\[
g_\text{NMDA}(V) = \frac{1}{1 + [\text{Mg}^{2+}] \cdot \exp(-\gamma V) / \eta}
\]

The Mg²⁺ block is relieved by depolarization, making the NMDA receptor a coincidence detector for pre- and postsynaptic activity.

## Transfer Functions

The conductance model generates **transfer functions** mapping AP amplitude to:

1. **Total calcium influx** — integrated calcium entry during an AP
2. **Depression/potentiation (D/P) ratio** — the relative strength of LTD vs LTP, used as a parameter in the STDP rule

These transfer functions are computed by `scripts/conductance_data.py` and saved to `results/conductance_runs.joblib`.

## Implementation

The channel models are implemented in the [`VGCC`](../api/conductance.md), and [`NMDAR`](../api/conductance.md) class in `src/conductance.py`, along with the other components of the transfer function model. 
