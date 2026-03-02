# Installation

## Prerequisites

- Python 3.9+ (3.11 recommended)
- [Conda](https://docs.conda.io/en/latest/) (for managing numba dependency)

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/landoskape/plasticity-modeling
cd plasticity-modeling
```

### 2. Create a conda environment

```bash
conda create -n plasticity-modeling python=3.11
conda activate plasticity-modeling
```

### 3. Install the package

```bash
pip install -e .
```

This installs the `src` package in editable mode along with all core dependencies (numpy, scipy, matplotlib, pydantic, etc.).

### 4. Install numba

Numba must be installed via conda (not pip) for proper LLVM support:

```bash
conda install numba
```

### Optional: PyTorch for classical Hebbian models

If you want to run the classical Hebbian learning models, follow the [PyTorch installation guide](https://pytorch.org/get-started/locally/)
for pytorch and GPU support.

### Optional: Documentation dependencies

To build the documentation locally:

```bash
pip install -e ".[docs]"
```

## Verify Installation

```bash
python -c "import src; print(src.__version__)"
```

You should see the version string printed without errors. You can also verify the pipeline is ready:

```bash
python run_pipeline.py --peek
```

This will print which steps would run or be skipped, without actually executing anything.
