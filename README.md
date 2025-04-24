# plasticy-modeling

This repo contains the code and data required to generate the figures of our upcoming manuscript. 
Eventually I might want to turn this into a useful piece of software -- but right now it's simply
for keeping the modeling code and figure-making scripts well organized. So, with that in mind, the
README is going to just be a map of figures and how to make them. 

## Installation
I didn't include pytorch in the pyproject.toml dependencies for the usual reason.

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
- Results / iaf_runs / correlated / 20250320 has the results for the updated correlation experiment with new names (p, ds, dc)
- Results / iaf_runs / correlated / 20250324 has full_output results for the correlated model for making example figures

## Section on Hofer Reconstructions
- Results / iaf_runs / hofer / 20250320 has the results for the hofer experiment (this without no edge enforced)
- Results / iaf_runs / hofer / 20250417 has full_output results for hofer with source rates included
- Results / iaf_runs / hofer / 20250421 has the results for the hofer experiment with forced no edge!
- Results / iaf_runs / hofer_replacement / 20250423 has the results for the hofer replacement experiment (typical parameters)
- Results / iaf_runs / hofer / 20250423 has the results for the hofer replacement experiment with independent noise (at 0.1)

# Figure Mapping
- Overall Notes: 
  - Show correlation matrix of inputs IFF it's easy to see how the input correlations
    map onto learned weights in the weight trajectory graph...
  - It would be cool to do hierarchical clustering and then show it with this order!

- Figure 6: Hofer Summary Results
  - Panel A: Schematic of "co-axial space" (try to make it look like Figure 3 from Iacaruso)
            - DONE
  - Panel B: Summary curves with similar color scheme as Figure 4D showing contribution from each group
     - (Will need to break this into different plots for different edge probabilities -- might just show distal-complex!)
  - Panel C: Overall summary heatmap comparing edge probability and D/P ratio to net weight on co-axial
  - Panel D: Replication of Figure 3D from Iacaruso showing how orientation tuning is aligned in co-axial space

-- Try hofer_noinhibition.yaml