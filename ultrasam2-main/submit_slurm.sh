#!/bin/bash
#SBATCH --job-name=ultrasam2_train
#SBATCH --output=logs/job_%j.log
#SBATCH --error=logs/job_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=<your-gpu-node>  # replace with your cluster's GPU node name
#SBATCH --gres=gpu:nvidia-rtx-a6000:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

mkdir -p logs

# SLURM opens --output/--error before this script body runs, so they can only
# target a path that already exists (logs) -- the job ID isn't known until
# submission, so the per-job folder can't be pre-created. Once running, move
# those files into a fresh per-job folder instead: a same-filesystem rename
# doesn't disturb the file descriptor SLURM is still writing to, so nothing
# already written (or written after) is lost.
JOB_DIR="logs/job_${SLURM_JOB_ID}"
mkdir -p "$JOB_DIR"
mv "logs/job_${SLURM_JOB_ID}.log" "$JOB_DIR/job_${SLURM_JOB_ID}.log"
mv "logs/job_${SLURM_JOB_ID}.err" "$JOB_DIR/job_${SLURM_JOB_ID}.err"

# Snapshot this run's train_stats.json into the job folder too. Runs on any
# script exit (including SLURM hitting the time limit, which sends SIGTERM
# before SIGKILL) so the snapshot happens even if training gets cut off.
TRAIN_STATS="../../sam2_logs/configs/sam2.1_training/sam2.1_hiera_b+_ultrasound_finetune/logs/train_stats.json"
snapshot_train_stats() {
    [ -f "$TRAIN_STATS" ] && cp "$TRAIN_STATS" "$JOB_DIR/train_stats.json"
}
trap snapshot_train_stats EXIT

# Load Python module and activate virtual environment
source ~/storage/Dissertation/ultrasam2_venv/bin/activate

# Run training
python3 -u training/train.py \
    --config configs/sam2.1_training/sam2.1_hiera_b+_ultrasound_finetune \
    --num-gpus 1