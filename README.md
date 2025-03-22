# plasticy-modeling

This repo contains the code and data required to generate the figures of our upcoming manuscript. 
Eventually I might want to turn this into a useful piece of software -- but right now it's simply
for keeping the modeling code and figure-making scripts well organized. So, with that in mind, the
README is going to just be a map of figures and how to make them. 

## Section on STDP Predictions
This figure is meant to serve as a link between our experimental data, biophysical modeling, and
the predictions that motivate our STDP models. They start with the biophysical conductance model
from Figure 8 of the experimental paper (https://elifesciences.org/articles/76993). We argue that
the relative calcium influx evoked by APs of various amplitudes maps linearly onto the amount of 
LTP or LTD evoked by an STDP pairing protocol. We use the dose-dependent curves from [Nevian &
Sakmann 2006](https://www.jneurosci.org/content/26/43/11001) in Figure 4 panels B & D to infer
how changing the effective calcium concentration affects the magnitude of LTP or LTD. 

There is a [script](./scripts/stdp_experimental_prediction.py) for generating the key figures, in which
the main loop automatically generates figures. (I still need to configure saving for it to save
them somewhere). 

- ``run_simulations``: will run simulations with quadratic APs. This is sufficient to produce the
data required for reproducing Figure 8 of the experimental paper. It produces a data dict with 
all the results. 
- ``show_nevian_reconstruction``: will show our reconstructionof the Nevian & Sakmann 2006 results
to demonstrate that we did a good job (you have to compare by eye with the paper...). 
- ``show_estimated_plasticity``: will build transfer functions from the simulations and the model
derived from Nevian & Sakmann to demonstrate our estimate of how varying AP amplitude affects the 
amount of plasticity evoked by positive pairings (LTP) or negative pairings (LTD).
   - NOTE: This needs to be revisited a bit because the calcium concentration at which the plasticity
   is fully blocked is probably not 0mM! (E.g. there might be a minimum effective concentration).

## Section on IaF Correlation
- In iaf_correlation.ipynb the main result is in the second figure, showing the weight as a function of input correlation for different
d/p ratios and locations on the branch. It works!!!
- Need to add the single example figure and a summary plot and a good illustration of the correlated source modell...
- Results / iaf_runs / ica / 20250320 has the results for the updated correlation experiment with new names (p, ds, dc)

## Section on Hofer Reconstructions
- Results / iaf_runs / hofer / 20250320 has the results for the hofer experiment
- Consider the possibility of allowing each pixel to have it's own interval? 

## Software Development Goals
- Massive progress on configuration. Got some finishing touches I think, then ready. 
  - Create system for updating specific parameters based on an otherwise completed config (or maybe just after?)
    - The hints of this system are in the _preparing_grid_search directory
- Parallelization plan:
  - Also check if concatenating synapse groups makes them faster... (e.g. 10x100 synapses vs 1x1000...)
  - If so might be good to add a "concatenate" and "chunk" method that can be used to speed up parallel simulations
- Analysis organization:
  - Right now the iaf_correlation.ipynb and the iaf_hofer.ipynb are both doing hard coded analysis stuff with a lot
    of repeated code. Gotta create centralized analysis functions that can be reused (preferably across experiments?) 

# Figure Mapping
- Overall Notes: 
  - Proximal: Black, Distal-Simple: Dark-Gray, Distal-Complex: Blue
  - Show correlation matrix of inputs IFF it's easy to see how the input correlations
    map onto learned weights in the weight trajectory graph...
  - When redoing the big runs, change names to proximal, distal-simple, distal-complex!!!!
  - Make a plotting support class for the different groups with classmethods for colors, names
    etc etc anything else for consistency and reuse of common motifs. 

- Figure 1: Experimental data exposition:
  - Panel A: Schematic of dendritic branch definitions (proximal, distal-simple, distal-complex)
  - Panel B: Schematic of amplification and calcium influx and how they're measured
  - Panel C: Average AP-evoked calcium influx for each group (p, ds, dc)
  - Panel D: Average AP-evoked amplification for each group (neuron sims if needed!)

- Figure 2: Experimental predictions for STDP
  - Panel A: Simulation of expected NMDAR and VGCC dependent calcium influx for each group
  - Panel B: Nevian reconstruction with relabeled curves as NMDA/VGCC dependence
  - Panel C: Estimated plasticity transfer function
    
- Figure 3: STDP Model
  - Panel A: Schematic of STDP model neuron
  - Panel B: Schematic of STDP model STDP rules

- Figure 4: IaF Correlation Model
  - Panel A: Demonstration of stimulus structure for correlation model
  - Panel B: Demonstration of IaF PSTH (in column temporally aligned with A)
  - Panel C: Demonstration of IaF Input Weights for the three groups (in column temporally aligned with A)
  - ???????: Correlation matrix of inputs?
  - Panel D: Summary data of net weight for each group
    - Note: This will be horizontal instead of vertical, so create a nice schematic to indicate input correlation
      that can be reused in panel A / C / D / ??? for continuity.
  - Panel E: Summary statistics (correlation FWHM for each group as a function of D/P ratio)

- Figure 5: Hofer Prediction Exposition
  - Panel A: Illustration of visual environment and receptive fields
  - ???????: Correlation matrix of inputs?
  - Panel B: Show how we display "net receptive fields"
  - Panel C: Show example initial / post-learning weight maps for each group (low edge, high D/P ratio)
  - Panel D: Same as above but for high edge probability and low D/P ratio

- Figure 6: Hofer Summary Results
  - Panel A: Schematic of "co-axial space" (try to make it look like Figure 3 from Iacaruso)
  - Panel B: Summary curves with similar color scheme as Figure 4D showing contribution from each group
    - (Will need to break this into different plots for different edges -- might just show distal-complex!)
  - Panel C: Overall summary heatmap comparing edge probability and D/P ratio to net weight on co-axial
  - Panel D: Replication of Figure 3D from Iacaruso showing how orientation tuning is aligned in co-axial space





-- Need to figure out what's up with the new no-replacement hofer runs... (see iaf_testing_sims)
   -- problem is that there's too much requirement for inputs? Maybe not enough decorrelation?
   -- Solution ideas:
      - Increase "concentration" of input tuning
      - Vary the baseline vs driven rate of the inputs
      - Check what's going on with homeostatic tuning as well