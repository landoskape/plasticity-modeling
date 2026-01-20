#!/bin/bash -l
#$ -l h_rt=03:00:00
#$ -l mem=4G
#$ -N iaf_array_full
#$ -wd /home/skgta69/Scratch/plasticity-modeling
#$ -t 1-50

module purge
module load python/miniconda3/24.3.0-0
conda activate /home/skgta69/Scratch/conda-envs/iaf

# ---------- user choices ----------
CONFIG="correlated"
EXP_NAME="jan20_full1"
R=10
# ---------------------------------

dp_ratio_index=$(( ($SGE_TASK_ID - 1) / R ))
repeat=$(( ($SGE_TASK_ID - 1) % R ))

echo "Task $SGE_TASK_ID: dp_ratio_index=$dp_ratio_index repeat=$repeat"

python scripts/iaf_correlation.py \
  --config "$CONFIG" \
  --repeats "$R" \
  --dp_ratio_index "$dp_ratio_index" \
  --repeat "$repeat" \
  --exp_folder "$EXP_NAME"
  --duration 9600