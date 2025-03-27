#!/bin/bash

# Script to run training on SLURM cluster

#SBATCH --job-name=continual-learning
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=16G
#SBATCH --partition=<partition_name>
#SBATCH --qos=<qos_name>
#SBATCH --gres=gpu:1
#SBATCH --chdir=<working_directory>
#SBATCH --output=<output_file>
#SBATCH --error=<error_file>
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<email>

# Enable aliases
shopt -s expand_aliases

# Print job info if running on SLURM
if [ -n "$SLURM_JOB_ID" ] || [ -n "$SLURM_JOBID" ] || [ -n "$SLURM_CLUSTER_NAME" ]; then
  alias python="srun python"
  # print info about current job
  scontrol show job "$SLURM_JOB_ID"
fi

# Activate virtual environment
source .venv/bin/activate

# Debugging
set -v
set -e
set -x

# Run the script
python -m src.main
