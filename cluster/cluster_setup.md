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
print("All good")
PY
```

If this prints `All good`, the environment is correct.

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
