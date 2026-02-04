# Running `plasticity-modeling` on Myriad with Conda

This guide explains **why Conda is the right choice on Myriad**, how to **set it up once**, and how to **use it every time you log in or submit jobs**.

It is written to avoid the common Myriad pitfalls:
- Python bundle vs venv conflicts
- Missing SciPy inside venvs
- `typing_extensions` / `pydantic` breakage
- Slow or impossible source builds

---

## Why Conda (on Myriad)

Myriad’s `python3/recommended` module is **not a normal Python install**:
- It is a *bundle* with its own internal virtualenv
- Some packages (e.g. SciPy) live inside that bundle venv
- Creating your own `venv` hides those packages

Conda avoids this entirely by giving you a **fully self-contained environment** that:
- Includes SciPy, NumPy, Numba, scikit-learn, etc.
- Does not depend on Myriad’s Python bundles
- Behaves identically on login nodes and compute nodes

For long-running batch jobs, this is the most reliable option.

---

## One-time setup (do this once)

### 1. Log in and go to Scratch
```bash
ssh <uclid>@myriad.rc.ucl.ac.uk
cd ~/Scratch
```

Always work in **Scratch**, not `$HOME`.

---

### 2. Load Miniconda
```bash
module purge
module load python/miniconda3/24.3.0-0
```

---

### 3. Create a Conda environment in Scratch
```bash
mkdir -p ~/Scratch/conda-envs
conda create -y -p ~/Scratch/conda-envs/iaf python=3.9
conda activate ~/Scratch/conda-envs/iaf
```

This environment lives entirely in Scratch and will not interfere with system Python.

---

### 4. Install dependencies (compiled stack via conda-forge)
```bash
conda install -y -c conda-forge \
  numpy scipy numba pandas scikit-learn matplotlib tqdm pyyaml pydantic ipykernel
```

Install any pip-only packages afterwards:
```bash
pip install syd freezedry eval_type_backport
```

---

### 5. Clone your repo (once)
```bash
cd ~/Scratch
git clone <YOUR_GITHUB_REPO_URL>
cd plasticity-modeling
```

---

### 6. Install your package (no dependency resolution)
```bash
pip install -e . --no-deps
```

---

### 7. Smoke test
```bash
python - <<'PY'
import numpy, scipy, numba, sklearn, pydantic
import scripts.iaf_correlation
print("Send it")
PY
```

---

## Every time you log in (or start a new shell)

```bash
module purge
module load python/miniconda3/24.3.0-0
conda activate ~/Scratch/conda-envs/iaf
cd ~/Scratch/plasticity-modeling
```

You are now ready to run code or submit jobs.

---

## Using Conda in batch jobs (critical)

In **every** `qsub` script, include this at the top:

```bash
module purge
module load python/miniconda3/24.3.0-0
conda activate ~/Scratch/conda-envs/iaf
```

This guarantees:
- Same Python
- Same libraries
- Same behavior as your login tests

---

## Minimal example batch job

```bash
#!/bin/bash -l
#$ -l h_rt=01:00:00
#$ -l mem=4G
#$ -N iaf_test
#$ -wd /home/<uclid>/Scratch/plasticity-modeling

module purge
module load python/miniconda3/24.3.0-0
conda activate ~/Scratch/conda-envs/iaf

python scripts/iaf_correlation.py \
  --distal_dp_ratios 1.0 \
  --repeats 1
```

Submit with:
```bash
qsub iaf_test.sh
```

---

## Dynamic work queue (recommended)

For long runs with variable completion times, use a shared SQLite queue and worker array. Each worker keeps pulling tasks until the queue is empty or walltime is nearly up.

### 1. Build the queue

Correlated runs:
```bash
python cluster/build_queue.py \
  --mode correlated \
  --config correlated \
  --repeats 10 \
  --duration 9600 \
  --exp-folder jan20_full1
```

Hofer reconstruction runs:
```bash
python cluster/build_queue.py \
  --mode hofer \
  --config hofer_replacement \
  --repeats 10 \
  --duration 9600 \
  --exp-folder jan21_full1_hofer_replacement
```

This creates `cluster/queue.sqlite` with one task per `(dp_ratio_index, repeat)`.

### 2. Submit worker array

Use the provided worker script:
```bash
qsub cluster/iaf_worker_array.sh
```

Edit `cluster/iaf_worker_array.sh` to set worker count (`-t 1-<N>`), walltime, and queue path.

### 3. Monitor status

```bash
python cluster/queue_status.py --queue ~/Scratch/plasticity-modeling/cluster/queue.sqlite
python cluster/queue_status.py --queue ~/Scratch/plasticity-modeling/cluster/queue.sqlite --show-failed --limit 5
```

Task logs land in `cluster/queue_logs/`.

### 4. Graceful retries

Retries are automatic up to `--max-attempts` in the worker script. To requeue failed tasks manually:
```bash
python cluster/queue_status.py --queue ~/Scratch/plasticity-modeling/cluster/queue.sqlite --requeue-failed --reset-attempts
```

---

## Optuna studies on the cluster

For Optuna tuning, multiple workers share a single SQLite database inside a run directory.
All array tasks **must** point to the same run directory to participate in the same study.

### 1. Pick a study name and run date

The Optuna array script builds the run directory as:
```
~/Scratch/plasticity-modeling/results/optuna_runs/<STUDY_NAME>_<YYYYMMDD>
```

### 2. Submit the Optuna worker array

Use the provided script:
```bash
qsub cluster/optuna_worker_array.sh
```

Edit `cluster/optuna_worker_array.sh` to set:
- `STUDY_NAME`
- `N_TRIALS_PER_TASK`
- `DURATION`, `NUM_NEURONS`
- `STORAGE_TIMEOUT` (SQLite lock timeout)
- Array size (`-t 1-<N>`)

The script creates the run directory automatically if it does not exist.

### 3. Inspect the study

After runs complete, review results with:
```bash
python scripts/optuna_review.py --run-dir ~/Scratch/plasticity-modeling/results/optuna_runs/<STUDY_NAME>_<YYYYMMDD>
python scripts/optuna_review.py --run-dir ~/Scratch/plasticity-modeling/results/optuna_runs/<STUDY_NAME>_<YYYYMMDD> --compact
```

---

## Multi-stage Optuna test workflow

Use this progression to avoid costly failed cluster runs.

### 1. Local direct test (single trial)

```bash
python scripts/optuna_proximal_tuning.py \
  --study-name proximal_entropy_test \
  --n-trials 1 \
  --duration 60 \
  --num-neurons 1
```

This creates a local run dir under `results/optuna_runs/` and verifies the end-to-end loop.

### 1.1. Local “array-like” test (optional)

There is no true array locally, but you can run two workers sequentially against the same run dir:
```bash
python cluster/optuna_worker.py \
  --study-name proximal_entropy_test \
  --run-dir results/optuna_runs/proximal_entropy_test_YYYYMMDD \
  --n-trials 1 \
  --duration 60 \
  --num-neurons 1
```

Repeat once to confirm it resumes the same study without errors.

### 2. Cluster direct test (single worker)

Run a single worker without an array to validate the environment and storage:
```bash
python cluster/optuna_worker.py \
  --study-name proximal_entropy_smoke \
  --run-dir ~/Scratch/plasticity-modeling/results/optuna_runs/proximal_entropy_smoke_YYYYMMDD \
  --n-trials 1 \
  --duration 120 \
  --num-neurons 1
```

### 3. Cluster array test (short run)

Edit `cluster/optuna_worker_array_test.sh` with short duration (e.g., 120s) and small trials per task,
then submit a small array (e.g., `-t 1-5`) with `qsub cluster/optuna_worker_array_test.sh`

### 4. Cluster full array run

Choose full test settings in `cluster/optuna_worker_array.sh` and submit the full optimization run with 
`qsub cluster/optuna_worker_array.sh`. 

---

## Reproducibility (recommended)

Freeze the environment so you can recreate it later:

```bash
conda env export -p ~/Scratch/conda-envs/iaf > env_iaf.yml
```

---

## Key takeaways

- Do **not** mix `python3/recommended`, `venv`, and `pip --user`
- Use **Conda** for a clean, stable scientific stack
- Keep environments in **Scratch**
- Always activate Conda in batch jobs
- Test interactively before scaling to arrays

This setup is stable, fast, and HPC-safe.
