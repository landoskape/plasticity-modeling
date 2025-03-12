#!/bin/bash
# Wrapper script to submit the dp_ratio_array SLURM job with configurable parameters

# Default values
DP_RATIOS="1.0, 1.025, 1.05, 1.075, 1.1"
NUM_SIMULATIONS=20
EXPERIMENT_TYPE="correlated"
ARRAY_SIZE="0-4"  # Must match the number of values in DP_RATIOS
TIME_LIMIT="04:00:00"
CONDA_ENV="plasticity-modeling"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dp_ratios)
            DP_RATIOS="$2"
            # Count the number of values to set the array size
            IFS=',' read -ra RATIO_ARRAY <<< "$DP_RATIOS"
            ARRAY_SIZE="0-$((${#RATIO_ARRAY[@]}-1))"
            shift 2
            ;;
        --num_simulations)
            NUM_SIMULATIONS="$2"
            shift 2
            ;;
        --experiment_type)
            EXPERIMENT_TYPE="$2"
            shift 2
            ;;
        --time_limit)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create a temporary SLURM script with the specified parameters
TMP_SLURM_SCRIPT="dp_ratio_array_tmp.slurm"
cp dp_ratio_array.slurm $TMP_SLURM_SCRIPT

# Update the parameters in the temporary script
sed -i "s/^#SBATCH --array=.*/#SBATCH --array=${ARRAY_SIZE}/" $TMP_SLURM_SCRIPT
sed -i "s/^#SBATCH --time=.*/#SBATCH --time=${TIME_LIMIT}/" $TMP_SLURM_SCRIPT
sed -i "s/source activate .*/source activate ${CONDA_ENV}/" $TMP_SLURM_SCRIPT
sed -i "s/DP_RATIOS=.*/DP_RATIOS=\"${DP_RATIOS}\"/" $TMP_SLURM_SCRIPT
sed -i "s/NUM_SIMULATIONS=.*/NUM_SIMULATIONS=${NUM_SIMULATIONS}/" $TMP_SLURM_SCRIPT
sed -i "s/EXPERIMENT_TYPE=.*/EXPERIMENT_TYPE=\"${EXPERIMENT_TYPE}\"/" $TMP_SLURM_SCRIPT

# Submit the job
echo "Submitting SLURM array job with the following parameters:"
echo "  DP_RATIOS: $DP_RATIOS"
echo "  ARRAY_SIZE: $ARRAY_SIZE"
echo "  NUM_SIMULATIONS: $NUM_SIMULATIONS"
echo "  EXPERIMENT_TYPE: $EXPERIMENT_TYPE"
echo "  TIME_LIMIT: $TIME_LIMIT"
echo "  CONDA_ENV: $CONDA_ENV"

JOB_ID=$(sbatch $TMP_SLURM_SCRIPT | awk '{print $4}')
echo "Job submitted with ID: $JOB_ID"

# Clean up the temporary script
rm $TMP_SLURM_SCRIPT

# Create the cluster directory and a basic slurm_settings.txt if they don't exist
mkdir -p cluster
if [ ! -f "cluster/slurm_settings.txt" ]; then
    echo "# SLURM settings" > cluster/slurm_settings.txt
    echo "JOB_FOLDER=\"./slurm_jobs\"" >> cluster/slurm_settings.txt
    echo "Created basic cluster/slurm_settings.txt file"
fi

echo "Done!" 