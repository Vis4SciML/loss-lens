#!/bin/bash
#SBATCH -A m636
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 64
#SBATCH --qos="regular"
#SBATCH --gpus-per-task=1

export SLURM_CPU_BIND="cores"
module load python
conda activate losslens
module load mongodb
srun sh run_case_studies.sh
conda deactivate
