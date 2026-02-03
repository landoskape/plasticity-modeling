#!/bin/bash -l
#$ -l h_rt=06:00:00
#$ -l mem=4G
#$ -N iaf_worker
#$ -wd /home/<uclid>/Scratch/plasticity-modeling
#$ -t 1-20

module purge
module load python/miniconda3/24.3.0-0
conda activate /home/<uclid>/Scratch/conda-envs/iaf

# ---------- user choices ----------
QUEUE=/home/<uclid>/Scratch/plasticity-modeling/cluster/queue.sqlite
WALLTIME_SECONDS=$((6 * 60 * 60))
MAX_ATTEMPTS=2
# ---------------------------------

python cluster/worker.py \
  --queue "$QUEUE" \
  --walltime-seconds "$WALLTIME_SECONDS" \
  --stop-seconds-before 600 \
  --poll-seconds 10 \
  --max-attempts "$MAX_ATTEMPTS"
