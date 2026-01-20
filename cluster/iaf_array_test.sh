#!/bin/bash -l
#$ -l h_rt=00:10:00
#$ -l mem=4G
#$ -N iaf_array_test
#$ -wd /home/skgta69/Scratch/plasticity-modeling
#$ -t 1-10

module purge
module load python/miniconda3/24.3.0-0
conda activate /home/skgta69/Scratch/conda-envs/iaf

# ---------- user choices ----------
CONFIG="correlated"
EXP_NAME="jan20_test1"
R=10 # must match --repeats and number of tasks!
# ---------------------------------

# Map array task id (1..N) -> (dp_ratio_index, repeat)
dp_ratio_index=$(( ($SGE_TASK_ID - 1) / R ))
repeat=$(( ($SGE_TASK_ID - 1) % R ))

echo "Task $SGE_TASK_ID: dp_ratio_index=$dp_ratio_index repeat=$repeat config=$CONFIG exp=$EXP_NAME"

python scripts/iaf_correlation.py \
  --config "$CONFIG" \
  --repeats "$R" \
  --dp_ratio_index "$dp_ratio_index" \
  --repeat "$repeat" \
  --exp_folder "$EXP_NAME"
  --duration 1