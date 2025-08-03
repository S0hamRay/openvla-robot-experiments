#!/bin/bash
#SBATCH --job-name=openvla_eval_cpu
#SBATCH --output=openvla_cpu.out
#SBATCH --error=openvla_cpu.err
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Initialize Conda
source ~/../../user_data/sraychau/miniconda3/etc/profile.d/conda.sh
conda activate VLAVenv2

# Go to project directory
cd ~/../../user_data/sraychau/Storage/ActualProject

# Set PYTHONPATH to include all necessary directories
export PYTHONPATH=.:./openvla-main:./Lec14a_camera_sensor

# Create output directories
mkdir -p rollouts_mujoco_simple
mkdir -p sparse_code_images

# Run the improved evaluation script with CPU-only mode
python openvla-main/experiments/robot/run_mujoco_simple_eval.py \
  --pretrained_checkpoint OpenVLA_Model/ \
  --num_episodes 10 \
  --max_steps 50 \
  --image_width 128 \
  --image_height 128 \
  --seed 42 \
  --rollout_dir rollouts_mujoco_simple \
  --center_crop \
  --force_cpu

echo "CPU evaluation completed successfully!"
echo "Check rollouts_mujoco_simple/ for video outputs"
echo "Check sparse_code_images/ for visualization plots" 
