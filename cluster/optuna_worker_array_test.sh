#!/bin/bash -l
#$ -l h_rt=00:10:00
#$ -l mem=2G
#$ -N optuna_proximal
#$ -wd /home/skgta69/Scratch/plasticity-modeling
#$ -t 1-4

module purge
module load python/miniconda3/24.3.0-0
conda activate /home/skgta69/Scratch/conda-envs/iaf

# ---------- user choices ----------
STUDY_NAME=proximal_entropy_test
RUN_DATE=${RUN_DATE:-$(date +%Y%m%d)}
RUN_DIR=/home/skgta69/Scratch/plasticity-modeling/results/optuna_runs/${STUDY_NAME}_${RUN_DATE}
N_TRIALS_PER_TASK=50
DURATION=800
NUM_NEURONS=3
STORAGE_TIMEOUT=60
# ---------------------------------

mkdir -p "$RUN_DIR"

python cluster/optuna_worker.py \
  --study-name "$STUDY_NAME" \
  --run-dir "$RUN_DIR" \
  --n-trials "$N_TRIALS_PER_TASK" \
  --duration "$DURATION" \
  --num-neurons "$NUM_NEURONS" \
  --storage-timeout "$STORAGE_TIMEOUT"
