---
layout: default
title: Home
---

# Compartment-specific plasticity produces observed distribution of spine tuning in primary visual cortex

<p class="lede">A biophysically grounded model linking dendritic calcium signaling, compartment-specific STDP, and the distribution of synaptic tuning in layer 2/3 visual cortex.</p>

## Why this model

Neurons segregate synaptic inputs across dendritic compartments. Anatomical constraints set where synapses can form, and plasticity sculpts their strength and selectivity. In visual cortex, retinotopically displaced inputs that support edge tuning preferentially form on distal dendritic branches, yet most computational models treat neurons as single-point compartments. We ask whether compartment-specific plasticity rules can explain this spatial organization.

## What we built

We connect dendritic biophysics to functional tuning by modeling how back-propagating action potential (bAP) amplitude shapes calcium influx through VGCCs and NMDARs, which in turn sets the balance of depression and potentiation in STDP. Branch-specific attenuation of bAPs produces a divergence in depression/potentiation ratio across compartments, creating distinct tuning regimes within a single neuron.

<hr />

## Key results

- Distal branches with complex morphology show reduced bAP-evoked VGCC calcium but preserved NMDAR amplification, predicting reduced depression with typical potentiation.
- A calibrated transfer function maps simulated calcium to plasticity magnitude, yielding branch-specific STDP rules.
- Reduced depression stabilizes weakly correlated inputs, shifting tuning thresholds in distal-complex compartments.
- A toy visual input model reproduces retinotopically displaced (coaxial) tuning on distal-complex branches.

<hr />

## Prediction

Branch-to-branch variability in distal tuning should be explained by local bAP amplitude. Morphological complexity serves as a practical proxy for that amplitude, predicting that coaxial inputs preferentially populate complex distal branches.

## Explore

<div class="grid">
  <div class="panel">
    <h3>Model details</h3>
    <p>A full description of the plasticity transfer function model and calibration pipeline.</p>
    <p><a href="{{ "/model/" | relative_url }}">Read the model overview</a></p>
  </div>
  <div class="panel">
    <h3>Figures</h3>
    <p>Static and interactive figure explorations (coming soon).</p>
    <p><a href="{{ "/figures/" | relative_url }}">Browse figures</a></p>
  </div>
  <div class="panel">
    <h3>Code</h3>
    <p>Scripts and configurations for reproducing simulations and figures.</p>
    <p><a href="{{ "/code/" | relative_url }}">View code resources</a></p>
  </div>
</div>

<hr />

## Coming next

<div class="figure-placeholder">
  <strong>Interactive figures</strong>
  <p class="callout">A lightweight figure explorer inspired by the visual clarity of Anthropic and the Transformers blog, adapted to a neutral scientific palette.</p>
</div>
