# Mujoco + OpenVLA Rollout and Sparse Code Visualization
# Copy this into a new Jupyter notebook

# Parameters (for papermill)
pretrained_checkpoint = "OpenVLA_Model/"
num_episodes = 2
max_steps = 50

# --- Imports ---
import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Union, Optional
import imageio
from sklearn.manifold import TSNE
# Optional: import umap
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("UMAP not available, will use t-SNE only")

from Lec14a_camera_sensor.mujoco_gym_env import MujocoSimpleEnv
from experiments.robot.robot_utils import (
    get_model,
    get_action,
    set_seed_everywhere,
    invert_gripper_action,
)
from experiments.robot.openvla_utils import get_processor
from prismatic.models.vlas.sparse_autoencoder import SparseAutoencoderTorch

# --- Config ---
class GenerateConfig:
    model_family = 'openvla'
    pretrained_checkpoint = pretrained_checkpoint  # Use papermill parameter
    load_in_8bit = False
    load_in_4bit = False
    center_crop = False
    unnorm_key = 'bridge_orig'
    num_episodes = num_episodes  # Use papermill parameter
    max_steps = max_steps  # Use papermill parameter
    image_width = 128
    image_height = 128
    seed = 42
    rollout_dir = './rollouts_mujoco_simple'

cfg = GenerateConfig()
os.makedirs(cfg.rollout_dir, exist_ok=True)
print(f"Configuration:")
print(f"  Checkpoint: {cfg.pretrained_checkpoint}")
print(f"  Episodes: {cfg.num_episodes}")
print(f"  Max steps: {cfg.max_steps}")
print(f"  Rollout dir: {cfg.rollout_dir}")

# Model and Environment Setup
print("Setting up model and environment...")
set_seed_everywhere(cfg.seed)
model = get_model(cfg)
processor = get_processor(cfg)
env = MujocoSimpleEnv(image_width=cfg.image_width, image_height=cfg.image_height)
print('Environment and model loaded successfully.')

# CRITICAL: Create and attach sparse autoencoder
# Get action dimension from the model
action_dim = model.get_action_dim(cfg.unnorm_key)
print(f"Action dimension: {action_dim}")

# Create sparse autoencoder for the action space
sparse_ae = SparseAutoencoderTorch(
    input_size=action_dim,      # 7D actions (or whatever your action space is)
    hidden_size=25,             # Sparse code dimension
    rho=0.01,                   # Target sparsity (1% activation)
    beta=3.0,                   # Sparsity penalty weight
    l2_reg=0.0001               # L2 regularization
)

# Attach the sparse autoencoder to the model
model.sparse_autoencoder = sparse_ae
print(f"Sparse autoencoder attached to model:")
print(f"  Input size: {sparse_ae.input_size}")
print(f"  Hidden size: {sparse_ae.hidden_size}")
print(f"  Target sparsity (rho): {sparse_ae.rho}")
print(f"  Sparsity penalty (beta): {sparse_ae.beta}")

# Run rollouts and collect sparse codes
def save_rollout_video(rollout_images, idx, success, rollout_dir):
    mp4_path = os.path.join(rollout_dir, f'episode_{idx}_success_{success}.mp4')
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f'Saved rollout MP4 at path {mp4_path}')
    return mp4_path

all_sparse_codes = []
print(f"Starting {cfg.num_episodes} episodes...")

for ep in range(cfg.num_episodes):
    print(f'Starting episode {ep}')
    obs, _ = env.reset()
    done = False
    t = 0
    rollout_images = []
    sparse_codes_episode = []
    
    while not done and t < cfg.max_steps:
        observation = {
            'full_image': obs['image'],
            'state': obs['proprio'],
        }
        action, sparse_code = get_action(cfg, model, observation, task_label='move block', processor=processor)
        if sparse_code is not None:
            sparse_codes_episode.append(sparse_code)
            print(f"  Step {t}: Got sparse code shape {sparse_code.shape}")
        else:
            print(f"  Step {t}: No sparse code returned")
        action = invert_gripper_action(action)
        obs, reward, done, _, info = env.step(action)
        rollout_images.append(observation['full_image'])
        t += 1
    
    save_rollout_video(rollout_images, ep, success=done, rollout_dir=cfg.rollout_dir)
    if sparse_codes_episode:
        sparse_codes_arr = np.concatenate(sparse_codes_episode, axis=0)
        np.save(os.path.join(cfg.rollout_dir, f'episode_{ep}_sparse_codes.npy'), sparse_codes_arr)
        all_sparse_codes.append(sparse_codes_arr)
        print(f'Saved sparse codes for episode {ep}, shape: {sparse_codes_arr.shape}')
    else:
        print(f'No sparse codes collected for episode {ep}')

env.close()
print("All episodes finished.")

# Load and visualize sparse codes
sparse_codes = []
for fname in os.listdir(cfg.rollout_dir):
    if fname.endswith('_sparse_codes.npy'):
        arr = np.load(os.path.join(cfg.rollout_dir, fname))
        sparse_codes.append(arr)
        print(f"Loaded {fname}: {arr.shape}")

if sparse_codes:
    sparse_codes = np.concatenate(sparse_codes, axis=0)
    print(f'\nTotal sparse codes loaded: {sparse_codes.shape[0]} samples, {sparse_codes.shape[1]} dimensions')
else:
    print('No sparse code files found!')
    sparse_codes = None

# Visualization with plot saving
if sparse_codes is not None and len(sparse_codes) > 0:
    print("Running t-SNE...")
    # t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(sparse_codes)-1), random_state=42)
    codes_2d = tsne.fit_transform(sparse_codes)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(codes_2d[:, 0], codes_2d[:, 1], alpha=0.7, s=50)
    plt.title('t-SNE of Sparse Codes (Action Manifold)', fontsize=14, fontweight='bold')
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.rollout_dir, 'tsne_sparse_codes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to {os.path.join(cfg.rollout_dir, 'tsne_sparse_codes.png')}")
    
    # UMAP (optional)
    if HAS_UMAP:
        print("Running UMAP...")
        reducer = umap.UMAP(n_neighbors=min(15, len(sparse_codes)-1), random_state=42)
        codes_umap = reducer.fit_transform(sparse_codes)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(codes_umap[:, 0], codes_umap[:, 1], alpha=0.7, s=50)
        plt.title('UMAP of Sparse Codes (Action Manifold)', fontsize=14, fontweight='bold')
        plt.xlabel('Component 1', fontsize=12)
        plt.ylabel('Component 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.rollout_dir, 'umap_sparse_codes.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved UMAP plot to {os.path.join(cfg.rollout_dir, 'umap_sparse_codes.png')}")

# Sparsity analysis with plot saving
if sparse_codes is not None and len(sparse_codes) > 0:
    # Calculate statistics
    avg_activations = np.mean(sparse_codes, axis=0)
    sparsity = np.mean(sparse_codes > 0.1)  # Consider > 0.1 as active
    
    print(f"\n=== SPARSE CODE ANALYSIS ===")
    print(f"Total samples: {sparse_codes.shape[0]}")
    print(f"Code dimensions: {sparse_codes.shape[1]}")
    print(f"Fraction of active features: {sparsity:.3f}")
    print(f"Mean activation: {np.mean(avg_activations):.4f}")
    print(f"Max activation: {np.max(avg_activations):.4f}")
    print(f"Min activation: {np.min(avg_activations):.4f}")
    print(f"Most active feature: {np.argmax(avg_activations)} (activation: {np.max(avg_activations):.4f})")
    print(f"Least active feature: {np.argmin(avg_activations)} (activation: {np.min(avg_activations):.4f})")
    
    # Plot average activation per feature
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(len(avg_activations)), avg_activations, alpha=0.7)
    plt.title('Average Activation per Sparse Feature', fontsize=12, fontweight='bold')
    plt.xlabel('Feature Index', fontsize=10)
    plt.ylabel('Average Activation', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Histogram of all activations
    plt.subplot(1, 2, 2)
    plt.hist(sparse_codes.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Sparse Code Activations', fontsize=12, fontweight='bold')
    plt.xlabel('Activation Value', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.rollout_dir, 'sparsity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved sparsity analysis plot to {os.path.join(cfg.rollout_dir, 'sparsity_analysis.png')}")
    
    # Additional analysis: Feature correlation matrix
    plt.figure(figsize=(8, 6))
    correlation_matrix = np.corrcoef(sparse_codes.T)
    plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Sparse Code Feature Correlations', fontsize=12, fontweight='bold')
    plt.xlabel('Feature Index', fontsize=10)
    plt.ylabel('Feature Index', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.rollout_dir, 'feature_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature correlation plot to {os.path.join(cfg.rollout_dir, 'feature_correlations.png')}")
    
else:
    print("No sparse codes to analyze!")

print(f"\n=== SUMMARY ===")
print(f"All plots saved to: {cfg.rollout_dir}")
print(f"Files created:")
for fname in os.listdir(cfg.rollout_dir):
    if fname.endswith('.png'):
        print(f"  - {fname}")
    elif fname.endswith('_sparse_codes.npy'):
        print(f"  - {fname}")
    elif fname.endswith('.mp4'):
        print(f"  - {fname}")
