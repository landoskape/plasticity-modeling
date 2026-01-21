#!/bin/bash -l
#$ -l h_rt=03:00:00
#$ -l mem=4G
#$ -N iaf_hofer_full
#$ -wd /home/skgta69/Scratch/plasticity-modeling
#$ -t 1-50

module purge
module load python/miniconda3/24.3.0-0
conda activate /home/skgta69/Scratch/conda-envs/iaf

# ---------- user choices ----------
CONFIG="hofer_replacement"
EXP_NAME="jan21_full1_hofer_replacement"
R=10  # repeats
# Default distal_dp_ratios: [1.0, 1.025, 1.05, 1.075, 1.1] = 5 ratios
# Total tasks = 5 * 10 = 50 (edge probabilities are looped within each task)
# ---------------------------------

# Map array task id (1..N) -> (dp_ratio_index, repeat)
# Edge probabilities are looped within each task
task_id_0=$(( $SGE_TASK_ID - 1 ))
repeat=$(( $task_id_0 % R ))
dp_ratio_index=$(( $task_id_0 / R ))

echo "Task $SGE_TASK_ID: dp_ratio_index=$dp_ratio_index repeat=$repeat"
echo "  (will loop through all edge probabilities within this task)"

python scripts/iaf_hofer_reconstruction.py \
  --config "$CONFIG" \
  --repeats "$R" \
  --dp_ratio_index "$dp_ratio_index" \
  --repeat "$repeat" \
  --exp_folder "$EXP_NAME" \
  --duration 9600
