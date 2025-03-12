#!/bin/bash
#SBATCH --job-name=dp_ratio_array       # create a short name for your job
#SBATCH --partition=kempner             # partition
#SBATCH --account=kempner_bsabatini_lab # account needed for kempner partition
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks-per-node=1             # total number of tasks per node
#SBATCH --cpus-per-task=64              # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1                    # number of allocated gpus per node
#SBATCH --mem=64G                       # total memory per node (4 GB per cpu-core is default)
#SBATCH --time=04:00:00                 # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin               # send email when job begins
#SBATCH --mail-type=end                 # send email when job ends
#SBATCH --array=0-4                     # array job with 5 tasks (for 5 dp_ratio values)

# we need to define the job name directly since it isn't a slurm environment variable
JOB_NAME="dp_ratio_array"

# Create a directory structure for this job
JOB_FOLDER="./slurm_jobs"  # Default, can be overridden by slurm_settings.txt if it exists
if [ -f "cluster/slurm_settings.txt" ]; then
    source cluster/slurm_settings.txt
fi

JOB_DIR=${JOB_FOLDER}/${JOB_NAME}-${SLURM_JOB_ID} # make a specific directory for this particular job
mkdir -p $JOB_DIR

# define a unique log file in the right place
logfile="${JOB_DIR}/slurm-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
echo "Writing to ${logfile}"

# load python and activate our conda environment
module purge
module load python
source activate pointersequencer  # Change this to your conda environment name

# record the start time
start_time=$(date +%s)

# Define the dp_ratio values to use (must match the array size above)
DP_RATIOS="0.9,1.0,1.1,1.2,1.3"

# Define the number of simulations to run for each dp_ratio
NUM_SIMULATIONS=5

# Define the experiment type (correlated or ica)
EXPERIMENT_TYPE="correlated"

# Define the output directory
OUTPUT_DIR="results/dp_ratio_array_${SLURM_JOB_ID}"

# Run the Python script with the appropriate array index
python run_dp_ratio_simulations.py \
    --array_index ${SLURM_ARRAY_TASK_ID} \
    --experiment_type ${EXPERIMENT_TYPE} \
    --num_simulations ${NUM_SIMULATIONS} \
    --output_dir ${OUTPUT_DIR} \
    --dp_ratios ${DP_RATIOS} \
    >> $logfile 2>&1

# record the end time
end_time=$(date +%s)

# measure the time elapsed for the core part of the job in the logfile
total_time=$((end_time-start_time))
echo "Total Time= "$total_time" seconds" >> $logfile 