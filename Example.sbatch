#!/bin/bash

#SBATCH --job-name=array
#SBATCH --output=array_%A_%a.out
#SBATCH --error=array_%A_%a.err
#SBATCH --array=0-100
#SBATCH --time=10:00:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=1
#SBATCH --mem=3G
#SBATCH --account=pi-amurugan
# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

# Add lines here to run your computations.
python kernel.py
