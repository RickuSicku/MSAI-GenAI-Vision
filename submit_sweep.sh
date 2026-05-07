#!/bin/bash
#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name=vae_hyperband
#SBATCH --output=vae_sweep_%j.log

# Clear out defaults and load only the modern Mamba module
module purge
module load mamba/24.3.0

# Safely activate the environment using the absolute path
source activate /gpfs/home/cqp0132/vae_sweep_project/vae_env

# Execute your hyperparameter search script
/gpfs/home/cqp0132/vae_sweep_project/vae_env/bin/python /gpfs/home/cqp0132/vae_sweep_project/parameter_search.py