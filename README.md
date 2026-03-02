# plasticy-modeling

This repo contains the code and data required to generate the figures of our upcoming manuscript. 
Eventually I might want to turn this into a useful piece of software -- but right now it's simply
for keeping the modeling code and figure-making scripts well organized. So, with that in mind, the
README is going to just be a map of figures and how to make them. 

## Installation
Git clone then pip install locally. 
```bash
git clone https://github.com/landoskape/plasticity-modeling
cd plasticity-modeling
conda create -n plasticity-modeling python=3.11
conda activate plasticity-modeling
pip install -e .
conda install numba
# If you want to run the classic hebbian models, you need to install torch
# follow: https://pytorch.org/get-started/locally/ for your computer
```

## Reproducing All Results

A single-command pipeline reproduces all data and figures from scratch:

```bash
python run_pipeline.py
```


> [!WARNING] The full pipeline (conductance + correlation + hofer + figures) can take hours to days
> depending on your hardware because the STDP simulations are slow and are run sequentially. To get a
> quick test you can reduce the duration within pipeline.yaml, but it might not get to steady-state!


### Per-step usage

```bash
python run_pipeline.py --steps conductance         # just conductance data
python run_pipeline.py --steps correlation hofer   # just IAF simulations
python run_pipeline.py --steps figures             # just regenerate figures
python run_pipeline.py --steps conductance figures # reruns conductance and regenerate figures
python run_pipeline.py --force                     # re-run even if outputs exist
python run_pipeline.py --peek                      # print what would run/skip without running
```

`--peek` checks each selected step's expected outputs and reports whether the step would run or be skipped,
then exits without launching any scripts.

### Reproducing exact manuscript figures

To generate figures using the exact simulation data from the manuscript:

1. Download the publication data from [TBD] into `results/iaf_runs/`
2. Run: `python run_pipeline.py --config manuscript.yaml --steps figures`

### Using your own runs

If you have existing simulation runs and want the pipeline to use them (instead of
re-running simulations), create a `pipeline_local.yaml` in the repo root:

```yaml
correlation:
  example_run_name: "my_example_run"
  full_run_name: "my_full_run"
hofer:
  run_name: "my_hofer_run"
```

This file is gitignored. The pipeline will skip simulation steps if the named directories
already exist under `results/iaf_runs/`, and pass the run names to figure generation.

### Direct script usage

Each script also works standalone:

```bash
python scripts/conductance_data.py --num_ap_amplitudes 10          # quick test
python scripts/iaf_correlation.py --run_name test --repeats 1 --duration 100
python scripts/make_figures.py --figures 4 5 6 --correlated-full-run 20260119 --hofer-run jan21_full1_hofer_20260121
```

