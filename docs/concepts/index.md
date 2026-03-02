# Concepts Overview

This section provides the scientific background behind the modeling work in this project.

## The Central Question

How does **action potential (AP) amplitude** interact with **dendritic location** to shape compartment-specific synaptic plasticity? Our experimental work shows branch-specific reductions in AP amplitude that cause a selective deficit of AP-evoked calcium influx, while preserving AP-dependent amplification of synaptic calcium influx. Since depression and potentiation in STDP depend on these two calcium signals, respectively, these data predict compartment specific plasticity. 

We inferred that a reduction in depression would permit synapses with weaker correlations to somatic firing to maintain stability, which provides an explanation for why retinotopically-displaced inputs are specifically observed on distal dendrites in visual cortical layer 2/3 cells. In this work, we demonstrate that our biophysically, experimentally grounded model indeed produces this synaptic input distribution. 

## Modeling Approach

We address this question through three complementary modeling components:

### 1. Experimental Data

The model is directly inspired and aligned with experimental data from our earlier eLife publication. 

[Read more →](experimental-data.md)

### 2. Conductance Model

A biophysical model of voltage-gated calcium channels (VGCCs) and NMDA receptors predicts how AP amplitude maps to calcium influx and, ultimately, to plasticity magnitude. This provides the **transfer functions** that link AP attenuation to the depression/potentiation ratio used in the STDP rule.

[Read more →](conductance-model.md)

### 3. IAF Neuron with STDP

An integrate-and-fire neuron model with spike-timing-dependent plasticity forms the simulation core. Synapses are grouped into compartments (proximal and distal) with different depression/potentiation ratios reflecting AP attenuation.

[Read more →](iaf-stdp.md)

### 4. Correlated Input Simulations

Correlated Poisson inputs drive the IAF neuron, and we observe how proximal and distal synaptic weights diverge over time due to compartment-specific plasticity rules.

[Read more →](correlated-inputs.md)

### 5. Hofer Reconstruction

Gabor-based orientation-tuned inputs model the co-axial connectivity patterns observed experimentally. This simulation reconstructs the orientation selectivity findings from the Hofer et al. dataset.

[Read more →](hofer-reconstruction.md)
