#!/bin/bash

# Local run script for testing the improved evaluation
# This script can be run directly without SLURM

echo "Starting local evaluation with improved headless environment..."

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

# Run the improved evaluation script with reduced parameters for local testing
python openvla-main/experiments/robot/run_mujoco_simple_eval.py \
  --pretrained_checkpoint OpenVLA_Model/ \
  --num_episodes 2 \
  --max_steps 30 \
  --image_width 128 \
  --image_height 128 \
  --seed 42 \
  --rollout_dir rollouts_mujoco_simple

echo "Local evaluation completed!"
echo "Check rollouts_mujoco_simple/ for video outputs"
echo "Check sparse_code_images/ for visualization plots" 