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
Key point: the circularBasalApical1 directory probably has the main code source for replicating
the Hofer reconstructions that I used in my thesis. 

## Current Progress:
- IaF Implementation works and recovers the ICA result. The parameter control is a bit rough right
now though, so it needs to be designed better for more streamlined and clearer parameter choices.
- Gotta use it to make a single source example and also prep the real Hofer model!!!

- Let's get OmegaConf working for good model construction!!!