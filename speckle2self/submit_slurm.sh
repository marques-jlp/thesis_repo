#!/bin/bash
#SBATCH --job-name=speckle2self_train
#SBATCH --output=logs/job_%j.log
#SBATCH --error=logs/job_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Load Python module and activate virtual environment
module load python/3.11.7
source ~/storage/Dissertation/s2f_venv/bin/activate

# Run your Python script
python3 -u train.py "$@"