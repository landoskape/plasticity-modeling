# plasticy-modeling

This repo contains the code and data required to generate the figures of our upcoming manuscript. 
Eventually I might want to turn this into a useful piece of software -- but right now it's simply
for keeping the modeling code and figure-making scripts well organized. So, with that in mind, the
README is going to just be a map of figures and how to make them. 

## Figure on STDP Predictions
This figure is meant to serve as a link between our experimental data, biophysical modeling, and
the predictions that motivate our STDP models. They start with the biophysical conductance model
from Figure 8 of the experimental paper (https://elifesciences.org/articles/76993). We argue that
the relative calcium influx evoked by APs of various amplitudes maps linearly onto the amount of 
LTP or LTD evoked by an STDP pairing protocol. We use the dose-dependent curves from [Nevian &
Sakmann 2006](https://www.jneurosci.org/content/26/43/11001) in Figure 4 panels B & D to infer
how changing the effective calcium concentration affects the magnitude of LTP or LTD. 

There is a [script](./scripts/figure_stdp_prediction.py) for generating the key figures, in which
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

## Figure(s) on Hofer Reconstructions
These aren't ready but I'll describe notes on progress here. 
Key points:
- the circularBasalApical1 directory probably has the main code source for replicating the Hofer reconstructions that I used in my thesis. 
- the stdp_cModel1 has the source code for replicating the correlation tuning for a single source

## Figure(s) on IaF Correlation
- In iaf_correlation.ipynb the main result is in the second figure, showing the weight as a function of input correlation for different
d/p ratios and locations on the branch. It works!!!
- Need to add the single example figure and a summary plot and a good illustration of the correlated source modell...

## Current Progress:
- IaF Implementation works and recovers the ICA result. The parameter control is a bit rough right
now though, so it needs to be designed better for more streamlined and clearer parameter choices.
- Gotta use it to make a single source example and also prep the real Hofer model!!!
- Massive progress on configuration. Got some finishing touches I think, then ready. 
  - Create system for updating specific parameters based on an otherwise completed config (or maybe just after?)
    - The hints of this system are in the _preparing_grid_search directory
- Parallelization plan:
  - Also check if concatenating synapse groups makes them faster... (e.g. 10x100 synapses vs 1x1000...)
  - If so might be good to add a "concatenate" and "chunk" method that can be used to speed up parallel simulations