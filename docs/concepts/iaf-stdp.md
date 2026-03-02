# IAF Neuron & STDP

The simulation core is an integrate-and-fire (IAF) neuron with spike-timing-dependent plasticity (STDP) at excitatory synapses.

## Leaky Integrate-and-Fire Neuron

The membrane potential \(V_m\) evolves according to:

\[
\tau_m \frac{dV_m}{dt} = -(V_m - V_\text{rest}) + R_m \cdot I_\text{syn}(t)
\]

where:

- \(\tau_m\) is the membrane time constant (default: 20 ms)
- \(R_m\) is the membrane resistance (default: 100 MΩ)
- \(V_\text{rest}\) is the resting/reset potential (default: −70 mV)

When \(V_m\) crosses the spike threshold (default: −50 mV), the neuron fires and \(V_m\) is reset to \(V_\text{rest}\).

## Synaptic Input

Synaptic current is the sum of conductance-based excitatory and inhibitory inputs:

\[
I_\text{syn}(t) = \sum_j g_j(t) \cdot (E_j - V_m)
\]

Each synapse group has its own reversal potential \(E_j\), time constant \(\tau_j\), and weight dynamics.

## STDP Rule

The STDP learning rule modifies synaptic weights based on the relative timing of pre- and postsynaptic spikes:

\[
\Delta w = \begin{cases}
A_+ \exp(-\Delta t / \tau_+) & \text{if } \Delta t > 0 \text{ (pre before post → LTP)} \\
-A_- \exp(\Delta t / \tau_-) & \text{if } \Delta t < 0 \text{ (post before pre → LTD)}
\end{cases}
\]

where \(\Delta t = t_\text{post} - t_\text{pre}\).

The **depression/potentiation (D/P) ratio** controls the balance:

\[
\frac{A_-}{A_+} = \text{D/P ratio}
\]

- D/P ratio > 1 → net depression (distal synapses with attenuated APs)
- D/P ratio = 1 → balanced plasticity
- D/P ratio < 1 → net potentiation

## Homeostasis

A slow homeostatic mechanism adjusts the STDP learning rate to maintain a target firing rate:

\[
\tau_h \frac{d\hat{r}}{dt} = -\hat{r} + r(t)
\]

The homeostatic scaling modulates the STDP rate based on the difference between the estimated rate \(\hat{r}\) and the set point \(r_\text{target}\).

## Synapse Replacement

An optional mechanism replaces weak synapses with new random connections, modeling structural plasticity. This is used in some Hofer reconstruction variants.

## Implementation

- **Neuron**: [`IaF`](../api/iaf/neuron.md) class in `src/iaf/iaf_neuron.py`
- **Synapse groups**: [`SourcedSynapseGroup`, `DirectSynapseGroup`](../api/iaf/synapse-group.md) in `src/iaf/synapse_group.py`
- **Configuration**: [`NeuronConfig`, `SynapseGroupConfig`](../api/iaf/config.md) in `src/iaf/config.py`
