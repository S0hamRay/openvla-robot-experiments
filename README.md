# OpenVLA Robot Experiments

This repository contains experiments and evaluations for OpenVLA (Open Vision Language Action) models in robotic environments using MuJoCo simulation.

## Project Structure

- `openvla-main/` - Main OpenVLA implementation and experiments
- `Lec14a_camera_sensor/` - Camera sensor implementations for MuJoCo
- `lerobot/` - LeRobot framework integration
- `dlimp/` - Data loading and processing utilities
- `submissions/` - Experiment submissions and results
- `OpenVLA_Model/` - Model checkpoints and configurations

## Setup Instructions

### Environment Setup

1. Install Miniconda if not already installed
2. Create and activate the conda environment:
   ```bash
   source ../../miniconda3/etc/profile.d/conda.sh
   conda activate VLAVenv2
   ```

### Running Experiments

1. Navigate to the project directory:
   ```bash
   cd ActualProject
   ```

2. Run the evaluation script:
   ```bash
   python run_mujoco_simple_eval_fixed.py
   ```

## Key Files

- `run_mujoco_simple_eval_fixed.py` - Main evaluation script for MuJoCo experiments
- `run_eval_local.sh` - Shell script for running local evaluations
- `openvla_eval_output.ipynb` - Jupyter notebook with evaluation results
- `ENHANCED_SIMULATION_SUMMARY.md` - Detailed simulation summary and findings

## Dependencies

- MuJoCo
- PyTorch
- OpenVLA
- LeRobot
- Various robotics and computer vision libraries

## Notes

- The project uses MuJoCo for physics simulation
- Camera sensors are implemented for vision-based tasks
- Evaluation results are stored in the `submissions/` directory
- Model checkpoints are managed in the `OpenVLA_Model/` directory
