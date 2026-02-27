# Plasticity Transfer Function Model

## Overview

The plasticity transfer function model simulates how action potential (AP) waveforms of varying amplitudes lead to calcium influx through two distinct channels—NMDA receptors (NMDARs) and voltage-gated calcium channels (VGCCs)—and how this calcium influx drives synaptic plasticity (LTP and LTD). The model integrates biophysical channel kinetics, calcium dynamics, and experimental data from calcium buffering experiments to predict plasticity magnitude as a function of postsynaptic depolarization.

## Core Workflow

### 1. Main Simulation Pipeline (`run_simulations`)

The primary workflow is orchestrated by the `run_simulations` function, which executes the following steps:

#### Step 1.1: Initialize Parameters and Experimental Constraints

  - **Buffer Selection**: The model can use calcium buffering data from BAPTA, EGTA, or an average of both buffers from Nevian & Sakmann (2006)
  - **AP Amplitude Range**: Generates a range of AP amplitudes from 0 to a maximum value (default: 100 mV)
  - **Biophysical Parameters**:
    - Intracellular calcium: 75 nM
    - Extracellular calcium: 1.5 μM
    - Temperature: 310.15 K
    - Time resolution: 0.01 ms
    - Simulation duration: 5 ms

#### Step 1.2: Initialize Channel Models

Two biophysical channel models are instantiated:

  **NMDA Receptor (NMDAR)**:
  - Models voltage-dependent Mg²⁺ block
  - Kinetics defined by:
    - k_off = exp(0.017V + 0.96) [ms⁻¹]
    - k_on = [Mg²⁺]exp(-0.045V - 6.97) [ms⁻¹·μM⁻¹]
  - Open probability: P_open = k_off / (k_on + k_off)
  - Time constant: τ = 1 / (k_on + k_off)

  **Voltage-Gated Calcium Channel (VGCC)**:
  - Two-gate model (activation m, inactivation h)
  - Rate constants:
    - α_m(V) = 0.055(-27-V) / (exp((-27-V)/3.8) - 1)
    - β_m(V) = 0.94·exp((-75-V)/17)
    - α_h(V) = 0.000457·exp((-13-V)/50)
    - β_h(V) = 0.0065 / (exp((-V-15)/28) + 1)
  - Open probability: P_open = m² · h

#### Step 1.3: Generate AP Waveforms and Simulate Channel Responses

For each AP amplitude in the range:

  a. **Create AP Waveform**: Generate a quadratic AP waveform with:
    - V(t) = V_base + V_amp - (at)² for -V_dur/2 < t < V_dur/2
    - Duration: 1 ms
    - Baseline voltage: -70 mV

  b. **Numerical Integration**: Use Euler's method to integrate channel gate dynamics:
    - dm/dt = α_m(1-m) - β_m·m
    - dh/dt = α_h(1-h) - β_h·h
    - dn/dt = k_off(1-n) - k_on·n

  c. **Calculate Open Probabilities**:
    - VGCC: P_vgcc = m² · h
    - NMDAR: P_nmdar = n

  d. **Compute Calcium Currents**: Using the Goldman-Hodgkin-Katz equation (modified):
    - I_Ca = P_open · V · ([Ca]_in - [Ca]_out·exp(-2VF/RT)) / (1 - exp(-2VF/RT))

#### Step 1.4: Integrate Calcium Influx

For each AP amplitude:

  - **NMDAR Calcium Influx**: Integrate the NMDAR calcium current over time
  - **VGCC Calcium Influx**: Integrate the VGCC calcium current over time
  - Store peak open probabilities and total calcium influx for each channel

### 2. Plasticity Transfer Function Estimation (`measure_transfer_functions`)

The `measure_transfer_functions` function takes the simulated calcium influx data and maps it to plasticity magnitude:

#### Step 2.1: Scale Calcium Influx Using Experimental Data

  **Experimental Calibration**:
  - Load experimental data eLife paper measuring calcium transients in dendritic spines
  - Extract two key quantities:
    - `max_nl_component`: Maximum nonlinear calcium component (primarily NMDAR-mediated)
    - `max_ap_only`: Maximum calcium influx from AP alone (primarily VGCC-mediated)
  - Calculate relative scaling: `relative_ltp_scale = max_nl_component / max_ap_only`

  **Normalize Simulated Calcium**:
  - Scale NMDAR calcium: `effective_ca_ltp = (1 / nmdar_peak / relative_ltp_scale) × nmdar_integral_ca`
  - Scale VGCC calcium: `effective_ca_ltd = (1 / vgcc_peak) × vgcc_integral_ca`
  - This converts arbitrary calcium units into relative concentrations where the maximum possible influx = 1.0

#### Step 2.2: Generate Transfer Functions from Buffer Experiments

From Nevian & Sakmann (2006) pharmacological experiments with BAPTA/EGTA:

  a. **Load Buffer-Plasticity Relationship**:
    - CSV data extracted from Figure 4 (Panels B & D) of Nevian & Sakmann (2006)
    - Contains measurements of LTP and LTD magnitude at different buffer concentrations
    - Fit sigmoid functions to both datasets: `y = L / (1 + exp(-k(x - x₀))) + b`

  b. **Model Buffer-Calcium Relationship**:
    - Assumes buffer capacity regime: [CaB]/[Ca] = [B]/K_d
    - Buffer dissociation constant: K_d = 250 μM
    - Remaining calcium after buffering: Ca_remaining = 1 / (1 + [Buffer]/K_d)

  c. **Construct Transfer Functions**:
    - Create mapping from calcium concentration to plasticity magnitude
    - LTP transfer function: maps effective NMDAR calcium to LTP magnitude
    - LTD transfer function: maps effective VGCC calcium to LTD magnitude
    - Resolution: 10,001 points over buffer concentration range 0-10 mM

#### Step 2.3: Map Calcium to Plasticity

For each simulated AP amplitude:

  - **Find Closest Calcium Concentration**: Match the effective calcium from simulations to the transfer function
  - **Extract Plasticity Magnitude**:
    - LTP: Lookup plasticity value at effective NMDAR calcium concentration
    - LTD: Lookup plasticity value at effective VGCC calcium concentration (sign-flipped)
  - Record interpolation error for quality control

### 3. Output Data Structure

The combined output from `run_simulations` includes:

  **Simulation Parameters**:
  - Time vectors, voltage ranges, AP amplitudes
  - Channel parameters (NMDAR, VGCC)
  - Calcium concentrations (internal/external)

  **Biophysical Responses** (for each AP amplitude):
  - Voltage trace: `v_trace`
  - Channel open probabilities: `nmdar_p`, `vgcc_p`
  - Calcium currents: `nmdar_ica`, `vgcc_ica`
  - Integrated calcium influx: `nmdar_integral_ca`, `vgcc_integral_ca`

  **Plasticity Predictions**:
  - `LTP`: Array of LTP magnitudes across AP amplitudes
  - `LTD`: Array of LTD magnitudes across AP amplitudes
  - `error_ltp_ca_estimate`: Interpolation error for LTP
  - `error_ltd_ca_estimate`: Interpolation error for LTD

## Mathematical Framework

### Channel Kinetics

  **First-order kinetics**: All channel gates follow first-order differential equations of the form:
  ```
  dx/dt = α(V)(1 - x) - β(V)x
  ```
  where x is the gate variable and α, β are voltage-dependent rate constants.

### Calcium Current

  **Goldman-Hodgkin-Katz formulation**:
  ```
  I_Ca = P_open · V · ([Ca²⁺]_in - [Ca²⁺]_out · exp(-2VF/RT)) / (1 - exp(-2VF/RT))
  ```
  This accounts for the electrochemical driving force and concentration gradient.

### Buffer Capacity Model

  **Relationship between buffer and free calcium**:
  ```
  [Ca_remaining] = [Ca_free] / [Ca_total] = 1 / (1 + κ)
  κ = [Buffer] / K_d
  ```
  where κ is the buffer capacity.

### Sigmoid Plasticity Function

  **Empirical relationship from Nevian experiments**:
  ```
  Plasticity(Buffer) = L / (1 + exp(-k(Buffer - Buffer₀))) + b
  ```
  Converted to calcium domain:
  ```
  Plasticity(Ca) = f(1 / (1 + [Buffer]/K_d))
  ```

## Key Assumptions

1. **Channel conductance scaling**: The model assumes relative (not absolute) conductances for NMDAR and VGCC, scaled by experimental calcium imaging data

2. **Linearity assumption**: The reduction in plasticity by calcium buffers is assumed to linearly correspond to the reduction in free calcium concentration

3. **Steady-state neglect**: The model integrates transient calcium influx without explicitly modeling calcium buffering, extrusion, or diffusion dynamics

4. **Glutamate binding**: NMDAR open probability depends only on voltage (Mg²⁺ block); glutamate binding is assumed saturated

5. **Single-compartment**: All channels exist in the same dendritic spine compartment with uniform voltage

## Integration with Experimental Data

The model critically depends on two experimental datasets:

1. **Nevian & Sakmann (2006)**: Provides the relationship between calcium buffering and plasticity magnitude through BAPTA/EGTA pharmacology

2. **Landau et al. (2022, eLife)**: Provides the relationship between AP waveforms and actual calcium transients in dendritic spines, used to scale the relative contributions of NMDAR vs VGCC

This integration allows the model to predict plasticity outcomes from first principles of channel biophysics while remaining constrained by empirical measurements of calcium dynamics and plasticity induction.

## Applications

This model enables:

- **Prediction of plasticity rules**: Estimate LTP/LTD as a function of postsynaptic depolarization amplitude
- **Testing perturbations**: Simulate effects of changing channel properties, buffer concentrations, or AP waveforms
- **Mechanistic insight**: Decompose plasticity into contributions from NMDAR (LTP) and VGCC (LTD) pathways
- **STDP predictions**: When combined with pre/post spike timing, predict spike-timing-dependent plasticity curves

## Detailed Methods: Measuring the Transfer Function

This section provides an in-depth explanation of how the model converts simulated calcium influx into predicted plasticity magnitude—the core of the transfer function measurement.

### Part 1: Calculating Effective Calcium Concentration

The biophysical simulations produce calcium influx measurements in arbitrary units because we do not know the absolute maximum conductance of NMDARs and VGCCs in dendritic spines. To convert these simulated values into meaningful calcium concentrations that can be related to plasticity, we perform a multi-step normalization procedure.

#### The Scaling Challenge

The simulations yield two quantities for each AP amplitude:
- `nmdar_integral_ca`: Total calcium influx through NMDARs (arbitrary units)
- `vgcc_integral_ca`: Total calcium influx through VGCCs (arbitrary units)

These values represent the time integral of calcium current, but they are in arbitrary units for two reasons:
1. The Goldman-Hodgkin-Katz equation we use computes current per unit conductance
2. We do not have measurements of the actual maximum channel conductance in dendritic spines

Moreover, NMDAR and VGCC conductances cannot be directly compared because we simulate them independently—each represents a relative calcium influx as a function of voltage, but with unknown scaling between the two channel types.

#### Solution: Experimental Calibration

To resolve this, we leverage experimental calcium imaging data from Landau et al. (2022, eLife), which measured actual calcium transients in dendritic spines under different stimulation conditions:

1. **AP-only condition** (`max_ap_only`): Maximum calcium transient evoked by backpropagating action potentials alone. This primarily reflects VGCC-mediated calcium influx.

2. **Nonlinear component** (`max_nl_component`): Maximum supralinear calcium signal observed when glutamate uncaging is paired with an AP, beyond the linear sum of individual responses. This nonlinearity primarily arises from NMDAR activation, which requires both glutamate binding and postsynaptic depolarization to relieve Mg²⁺ block.

The ratio of these experimental measurements provides the relative scaling between NMDAR and VGCC calcium contributions:

  ```
  relative_ltp_scale = max_nl_component / max_ap_only
  ```

#### Normalization Procedure

The core problem is that our simulations produce calcium currents in units of current per unit conductance, but we don't know the actual unit conductance of NMDARs or VGCCs in dendritic spines. We cannot simply compare the simulated NMDAR and VGCC calcium values because they are on different, arbitrary scales.

**The Solution**: We assume that the experimental measurements successfully evoked near-maximal channel activity, allowing us to use the experimental data as a "ground truth" for the relative calcium contributions of each channel type.

**Step 1**: Extract peak values from simulations
  ```
  nmdar_peak = max(nmdar_integral_ca)  # Maximum NMDAR calcium across all AP amplitudes
  vgcc_peak = max(vgcc_integral_ca)    # Maximum VGCC calcium across all AP amplitudes
  ```

**Step 2**: Calculate conversion factors using experimental calibration

Both channel types undergo the same two-step normalization logic:
1. **Scale relative to maximum**: Divide by the peak value to express calcium as a fraction of the maximum achievable
2. **Scale by experimental ratio**: Account for the experimentally measured difference in calcium contributions

For VGCCs:
  ```
  ltd_ca_to_buffer = 1.0 / vgcc_peak
  ```
This scales VGCC calcium relative to its maximum. We use VGCC as the reference, so no additional experimental correction is needed.

For NMDARs:
  ```
  ltp_ca_to_buffer = 1.0 / nmdar_peak / relative_ltp_scale
  ```
This performs the same "relative to maximum" scaling as VGCC, but adds the experimental correction factor:
- First term (`1.0 / nmdar_peak`): Normalizes NMDAR calcium relative to its simulated maximum
- Second term (`/ relative_ltp_scale`): Corrects for the experimental observation that NMDAR-mediated calcium (nonlinear component) is `relative_ltp_scale` times larger than VGCC-mediated calcium (AP-only) when both channels are maximally activated

**Step 3**: Apply scaling to all data points
  ```
  effective_ca_ltp = ltp_ca_to_buffer × nmdar_integral_ca
  effective_ca_ltd = ltd_ca_to_buffer × vgcc_integral_ca
  ```

This converts arbitrary simulation units into experimentally calibrated relative calcium concentrations.

#### Interpretation

The resulting `effective_ca_ltp` and `effective_ca_ltd` arrays now represent relative calcium concentrations on a scale where:
- A value of 1.0 corresponds to the maximum calcium influx achievable across the tested AP amplitude range
- The values are scaled to match the experimentally observed ratio between NMDAR and VGCC contributions
- These normalized values can be meaningfully compared to the calcium-plasticity transfer functions derived from buffering experiments

**Critical assumption**: This approach assumes that the maximum AP amplitude tested in our simulations (default: 100 mV) evokes near-maximal calcium influx through both channel types. If the AP amplitude range is insufficient to saturate channel opening, the normalization will be inaccurate.

### Part 2: Constructing the Plasticity Transfer Function

The `plasticity_transfer_function` converts the buffer-plasticity relationship measured by Nevian & Sakmann (2006) into a calcium-plasticity relationship that can be applied to our simulations.

#### The Buffer Capacity Regime

The key insight is that calcium buffers like BAPTA and EGTA work by binding free calcium ions, thereby reducing the effective calcium concentration available to trigger plasticity. In the "buffer capacity regime"—which applies when buffer concentration greatly exceeds calcium concentration—the relationship between buffer concentration and free calcium is:

  ```
  κ = [Buffer] / K_d
  [Ca_remaining] = [Ca_free] / [Ca_total] = 1 / (1 + κ)
  ```

where:
- `K_d` = 250 μM is the dissociation constant for the buffer
- `κ` is the buffer capacity
- `[Ca_remaining]` is the fraction of calcium that remains unbound

This means that adding more buffer linearly reduces the free calcium concentration available to activate plasticity machinery.

#### From Buffer to Calcium Space

The Nevian & Sakmann experiments measured plasticity magnitude as a function of buffer concentration, yielding sigmoid curves that were fit with:

  ```
  Plasticity(Buffer) = L / (1 + exp(-k(Buffer - Buffer₀))) + b
  ```

where `L`, `k`, `Buffer₀`, and `b` are fitted parameters (different for LTP and LTD).

The `plasticity_transfer_function` performs the following transformation:

**Step 1**: Generate buffer concentration range
  ```
  buffer_concentration = linspace(0, max_buffer_concentration, num_points)
  ```
This creates a dense grid (default: 10,001 points from 0 to 10 mM) of buffer concentrations.

**Step 2**: Evaluate sigmoid at each buffer concentration
  ```
  transfer = sigmoid(buffer_concentration, L, x₀, k, b)
  ```
This gives plasticity magnitude as a function of buffer concentration.

**Step 3**: Enforce sign constraints
  ```
  transfer = max(transfer, 0)  if LTP
  transfer = min(transfer, 0)  if LTD
  ```
LTP should be non-negative; LTD should be non-positive.

**Step 4**: Calculate free calcium at each buffer concentration
  ```
  κ = buffer_concentration / K_d
  Ca_remaining = 1.0 / (1 + κ)
  ```
This maps each buffer concentration to the corresponding fraction of free calcium.

**Step 5**: Reverse the arrays
  ```
  return Ca_remaining[::-1], transfer[::-1]
  ```
Because higher buffer concentrations correspond to *lower* free calcium, we reverse both arrays so they are ordered from low calcium → high calcium. This makes the lookup more intuitive: index 0 corresponds to low calcium (high buffer), and the final index corresponds to high calcium (low/no buffer).

#### Result

The function returns two arrays:
1. `ca_concentration`: Calcium concentrations from low to high (in relative units where 1.0 = no buffer)
2. `plasticity_transfer`: Corresponding plasticity magnitudes

This provides a lookup table: given an effective calcium concentration from our simulations, we can find the predicted plasticity magnitude.

#### Applying the Transfer Function

For each AP amplitude in our simulations:

1. Look up the simulated effective calcium: `effective_ca_ltp[i]` or `effective_ca_ltd[i]`
2. Find the nearest calcium value in the transfer function: `idx = argmin(|ca_concentration - effective_ca|)`
3. Extract the corresponding plasticity: `plasticity[i] = plasticity_transfer[idx]`

This yields predicted LTP and LTD magnitudes as a function of AP amplitude, completing the transformation from biophysical channel properties → calcium influx → plasticity outcomes.

#### Key Insight

The transfer function effectively says: "If adding X mM of buffer reduces plasticity by Y%, then a calcium concentration that is reduced by the same factor should also reduce plasticity by Y%." This assumes that:
- Buffers act solely by reducing free calcium (no other molecular effects)
- The relationship between free calcium and plasticity is the same in buffered and unbuffered conditions
- The plasticity machinery senses free calcium concentration, not buffered calcium

These assumptions allow us to "invert" the buffer experiment to infer the underlying calcium-plasticity relationship, which can then be applied to our biophysical simulations.

