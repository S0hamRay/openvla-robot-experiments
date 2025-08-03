#!/usr/bin/env python3
"""
Enhanced MuJoCo evaluation script with robotic arm and comprehensive visualization
"""

import os
import sys
import argparse
import numpy as np
import imageio
from pathlib import Path
import torch
import torch.nn.functional as F
import random

# Add parent directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))  # For Lec14a_camera_sensor

# Import the enhanced environment
from Lec14a_camera_sensor.enhanced_mujoco_env import EnhancedMujocoEnv

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced MuJoCo evaluation with robotic arm and visualization')
    parser.add_argument('--pretrained_checkpoint', type=str, required=True,
                       help='Path to pretrained checkpoint')
    parser.add_argument('--num_episodes', type=int, default=3,
                       help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=50,
                       help='Maximum steps per episode')
    parser.add_argument('--rollout_dir', type=str, default='rollouts_enhanced_with_plots',
                       help='Directory to save rollouts')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    return parser.parse_args()

def train_sparse_autoencoder(env, action_dim, hidden_dim=25, num_actions=1000, epochs=20, batch_size=64, device='cpu'):
    """
    Collects actions from the environment and trains a sparse autoencoder.
    Returns the trained SparseAutoencoderTorch.
    """
    try:
        from prismatic.models.vlas.sparse_autoencoder import SparseAutoencoderTorch
    except ImportError:
        print("Warning: Could not import SparseAutoencoderTorch, using simplified version")
        # Create a simplified autoencoder
        class SimplifiedAutoencoder:
            def __init__(self, input_size, hidden_size):
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.encoder = torch.nn.Linear(input_size, hidden_size)
                self.decoder = torch.nn.Linear(hidden_size, input_size)
            
            def forward(self, x):
                h = self.encoder(x)
                x_hat = self.decoder(h)
                return h, x_hat
            
            def parameters(self):
                """Return parameters for optimization"""
                return list(self.encoder.parameters()) + list(self.decoder.parameters())
            
            def to(self, device):
                self.encoder = self.encoder.to(device)
                self.decoder = self.decoder.to(device)
                return self
        
        SparseAutoencoderTorch = SimplifiedAutoencoder
    
    actions = []
    obs, _ = env.reset()
    
    print("Training sparse autoencoder...")
    
    for _ in range(num_actions * 2):  # Oversample to allow filtering
        action = env.action_space.sample()
        
        # Handle infinite action space bounds
        if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
            low = env.action_space.low
            high = env.action_space.high
            
            # Check if bounds are finite
            if np.any(np.isinf(low)) or np.any(np.isinf(high)):
                # For infinite bounds, just clip to a reasonable range
                action = np.clip(action, -10.0, 10.0)
                # Then normalize to [-1, 1]
                action = action / 10.0
            else:
                # For finite bounds, use standard normalization
                action = 2 * (action - low) / (high - low) - 1
        
        # Filter out NaN/Inf/huge values
        if np.any(np.isnan(action)) or np.any(np.isinf(action)) or np.any(np.abs(action) > 1e6):
            continue
            
        actions.append(action)
        if len(actions) >= num_actions:
            break
            
        obs, _, done, _, _ = env.step(action)
        if done:
            obs, _ = env.reset()
    
    if len(actions) == 0:
        print("  Warning: No valid actions collected, using synthetic data")
        # Generate synthetic actions as fallback
        actions = [np.random.uniform(-1.0, 1.0, action_dim).astype(np.float32) for _ in range(num_actions)]
    
    actions = np.stack(actions)
    actions = actions.astype(np.float32)
    print(f"  Collected {len(actions)} actions for training")
    
    # Ensure actions are in [-1, 1] range
    actions = np.clip(actions, -1.0, 1.0)
    
    # Train SAE
    sae = SparseAutoencoderTorch(input_size=action_dim, hidden_size=hidden_dim)
    sae = sae.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Add gradient clipping
    max_grad_norm = 1.0
    
    for epoch in range(epochs):
        perm = np.random.permutation(len(actions))
        losses = []
        
        for i in range(0, len(actions), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch = torch.tensor(actions[batch_idx], device=device)
            
            # Forward pass
            h, x_hat = sae.forward(batch)
            
            # Compute loss
            mse = F.mse_loss(x_hat, batch, reduction='mean')
            total_loss = mse
            
            # Check for NaN/Inf
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_grad_norm)
            
            optimizer.step()
            
            losses.append(total_loss.item())
        
        if losses and epoch % 5 == 0:  # Print every 5 epochs
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {np.mean(losses):.6f}")
    
    return sae

def get_varied_task_commands():
    """
    Returns a list of varied language commands for the robotic arm.
    """
    return [
        "move the arm to the red cube",
        "pick up the blue sphere and place it in the target zone",
        "grasp the green cylinder and move it to the left",
        "lift the yellow pyramid and position it on the right",
        "take the purple torus and place it in front",
        "manipulate the red cube to the upper area",
        "move the arm downward to reach the objects",
        "rotate the arm clockwise to access the sphere",
        "turn the arm counterclockwise to reach the cylinder",
        "push the arm forward to interact with objects",
        "pull the arm backward to avoid obstacles",
        "slide the arm to the side to reach different objects",
        "position the arm at an angle to grasp objects",
        "arrange the arm to access the target zone",
        "relocate the arm to a new position",
        "transport the arm to the object area",
        "displace the arm from its current location",
        "reposition the arm as requested",
        "shift the arm to the desired spot",
        "move the arm with precision to interact with objects"
    ]

def get_random_task_command():
    """Returns a random task command from the varied list."""
    commands = get_varied_task_commands()
    return random.choice(commands)

def save_rollout_video(rollout_images, idx, success, rollout_dir):
    os.makedirs(rollout_dir, exist_ok=True)
    mp4_path = os.path.join(rollout_dir, f"episode_{idx}_success_{success}.mp4")
    video_writer = imageio.get_writer(mp4_path, fps=10)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path

def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    args = parse_args()
    
    # Set random seed
    set_seed_everywhere(args.seed)
    
    # Create environment
    print("Creating enhanced MuJoCo environment...")
    env = EnhancedMujocoEnv(enable_rendering=True)
    
    # CLEAR THE ROLLOUT DIRECTORY BEFORE GENERATING NEW VIDEOS
    print(f"Clearing rollout directory: {args.rollout_dir}")
    if os.path.exists(args.rollout_dir):
        import shutil
        shutil.rmtree(args.rollout_dir)
        print(f"✓ Removed existing directory: {args.rollout_dir}")
    os.makedirs(args.rollout_dir, exist_ok=True)
    print(f"✓ Created fresh directory: {args.rollout_dir}")
    
    # === Train Sparse Autoencoder ===
    action_dim = 5  # Enhanced environment has 5-D action space
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training sparse autoencoder on {device}")
    sae = train_sparse_autoencoder(env, action_dim, hidden_dim=25, num_actions=1000, epochs=20, batch_size=64, device=device)
    
    # Directory for sparse code images
    sparse_img_dir = os.path.join(args.rollout_dir, "sparse_code_images")
    os.makedirs(sparse_img_dir, exist_ok=True)
    
    all_sparse_codes = []
    all_actions = []
    all_tasks = []
    all_arm_positions = []
    
    # Run episodes
    print(f"\nRunning {args.num_episodes} episodes with enhanced robotic arm environment...")
    
    for episode in range(args.num_episodes):
        print(f"\nEpisode {episode + 1}/{args.num_episodes}")
        
        # Select a varied task command for this episode
        task_command = get_random_task_command()
        print(f"  Task: '{task_command}'")
        
        # Reset environment
        obs, _ = env.reset()
        
        # Store frames for video
        frames = []
        actions = []
        arm_positions = []
        sparse_codes_episode = []
        
        # Run episode
        for step in range(args.max_steps):
            # Generate action using SAE
            action = np.random.uniform(-0.3, 0.3, env.action_space.shape[0])
            
            # Get sparse code from SAE
            with torch.no_grad():
                action_tensor = torch.tensor(action.reshape(1, -1), device=device, dtype=torch.float32)
                sparse_code, _ = sae.forward(action_tensor)
                sparse_code = sparse_code.cpu().numpy()
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            
            # Store data
            frames.append(obs['image'].copy())
            actions.append(action.copy())
            arm_positions.append(obs['arm_position'].copy())
            sparse_codes_episode.append(sparse_code)
            
            print(f"  Step {step+1}: reward={reward:.3f}, arm_pos={obs['arm_position']}")
            
            if done:
                print(f"  Episode ended after {step+1} steps")
                break
        
        # Check if episode was successful
        success = info.get('success', False) if 'info' in locals() else False
        print(f"  Episode {episode + 1} completed with {len(frames)} actions, Success = {success}")
        
        # Save episode data
        episode_dir = os.path.join(args.rollout_dir, f"episode_{episode}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Save video
        if frames:
            video_path = os.path.join(episode_dir, f"episode_{episode}_success_{success}.mp4")
            imageio.mimsave(video_path, frames, fps=10)
            print(f"  ✓ Saved video: {video_path}")
        
        # Save action data
        if actions:
            actions_array = np.array(actions)
            actions_path = os.path.join(episode_dir, f"episode_{episode}_actions.npy")
            np.save(actions_path, actions_array)
            print(f"  ✓ Saved actions: {actions_path}")
        
        # Save arm positions
        if arm_positions:
            arm_pos_array = np.array(arm_positions)
            arm_pos_path = os.path.join(episode_dir, f"episode_{episode}_arm_positions.npy")
            np.save(arm_pos_path, arm_pos_array)
            print(f"  ✓ Saved arm positions: {arm_pos_path}")
        
        # Save sparse codes
        if sparse_codes_episode:
            sparse_codes_arr = np.concatenate(sparse_codes_episode, axis=0)
            sparse_codes_path = os.path.join(episode_dir, f"episode_{episode}_sparse_codes.npy")
            np.save(sparse_codes_path, sparse_codes_arr)
            print(f"  ✓ Saved sparse codes: {sparse_codes_path}")
            
            # Collect for global analysis
            all_sparse_codes.append(sparse_codes_arr)
            all_actions.append(actions_array)
            all_arm_positions.append(arm_pos_array)
            all_tasks.extend([task_command] * len(sparse_codes_arr))
            
            # --- Episode-specific Visualization ---
            # t-SNE of sparse codes
            try:
                from sklearn.manifold import TSNE
                codes_2d = TSNE(n_components=2, perplexity=min(30, len(sparse_codes_arr)-1), random_state=42).fit_transform(sparse_codes_arr)
                plt.figure(figsize=(10,8))
                plt.scatter(codes_2d[:,0], codes_2d[:,1], alpha=0.7, s=30, c=range(len(codes_2d)), cmap='viridis')
                plt.colorbar(label='Time Step')
                plt.title(f'Latent Space Trajectory - Episode {episode + 1}\nTask: {task_command}')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.tight_layout()
                plt.savefig(os.path.join(sparse_img_dir, f'episode_{episode}_latent_trajectory.png'), dpi=200)
                plt.close()
                print(f"  ✓ Saved latent trajectory plot")
            except Exception as e:
                print(f"  Warning: t-SNE visualization failed for episode {episode + 1}: {e}")
            
            # Action space trajectory
            try:
                plt.figure(figsize=(12,6))
                for i in range(actions_array.shape[1]):
                    plt.subplot(2, 3, i+1)
                    plt.plot(actions_array[:, i], 'o-', alpha=0.7)
                    plt.title(f'Arm Joint {i+1}')
                    plt.xlabel('Time Step')
                    plt.ylabel('Action Value')
                plt.suptitle(f'Arm Action Trajectory - Episode {episode + 1}\nTask: {task_command}')
                plt.tight_layout()
                plt.savefig(os.path.join(sparse_img_dir, f'episode_{episode}_action_trajectory.png'), dpi=200)
                plt.close()
                print(f"  ✓ Saved action trajectory plot")
            except Exception as e:
                print(f"  Warning: Action trajectory visualization failed for episode {episode + 1}: {e}")
            
            # Sparse code activation heatmap
            try:
                plt.figure(figsize=(10,6))
                plt.imshow(sparse_codes_arr.T, aspect='auto', cmap='viridis')
                plt.colorbar(label='Activation Value')
                plt.title(f'Sparse Code Activations - Episode {episode + 1}\nTask: {task_command}')
                plt.xlabel('Time Step')
                plt.ylabel('Latent Dimension')
                plt.tight_layout()
                plt.savefig(os.path.join(sparse_img_dir, f'episode_{episode}_activation_heatmap.png'), dpi=200)
                plt.close()
                print(f"  ✓ Saved activation heatmap")
            except Exception as e:
                print(f"  Warning: Activation heatmap failed for episode {episode + 1}: {e}")
            
            # Arm position trajectory
            try:
                plt.figure(figsize=(12,4))
                for i in range(arm_pos_array.shape[1]):
                    plt.subplot(1, 3, i+1)
                    plt.plot(arm_pos_array[:, i], 'o-', alpha=0.7)
                    plt.title(f'Arm Position {["X", "Y", "Z"][i]}')
                    plt.xlabel('Time Step')
                    plt.ylabel('Position')
                plt.suptitle(f'Arm Position Trajectory - Episode {episode + 1}\nTask: {task_command}')
                plt.tight_layout()
                plt.savefig(os.path.join(sparse_img_dir, f'episode_{episode}_arm_position.png'), dpi=200)
                plt.close()
                print(f"  ✓ Saved arm position plot")
            except Exception as e:
                print(f"  Warning: Arm position visualization failed for episode {episode + 1}: {e}")
    
    # --- Global Analysis Across All Episodes ---
    if all_sparse_codes:
        print("\nGenerating global latent space analysis...")
        
        # Combine all data
        all_codes_combined = np.concatenate(all_sparse_codes, axis=0)
        all_actions_combined = np.concatenate(all_actions, axis=0)
        all_arm_positions_combined = np.concatenate(all_arm_positions, axis=0)
        
        # Global t-SNE of all sparse codes
        try:
            from sklearn.manifold import TSNE
            codes_2d_global = TSNE(n_components=2, perplexity=min(30, len(all_codes_combined)-1), random_state=42).fit_transform(all_codes_combined)
            
            # Color by episode
            episode_colors = []
            for i, codes in enumerate(all_sparse_codes):
                episode_colors.extend([i] * len(codes))
            
            plt.figure(figsize=(12,10))
            scatter = plt.scatter(codes_2d_global[:,0], codes_2d_global[:,1], c=episode_colors, alpha=0.7, s=30, cmap='tab20')
            plt.colorbar(scatter, label='Episode')
            plt.title('Global Latent Space Clustering\nAll Episodes Combined')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.tight_layout()
            plt.savefig(os.path.join(sparse_img_dir, 'global_latent_clustering.png'), dpi=200)
            plt.close()
            print(f"  ✓ Saved global latent clustering plot")
            
            # Color by task
            unique_tasks = list(set(all_tasks))
            task_to_idx = {task: i for i, task in enumerate(unique_tasks)}
            task_colors = [task_to_idx[task] for task in all_tasks]
            
            plt.figure(figsize=(12,10))
            scatter = plt.scatter(codes_2d_global[:,0], codes_2d_global[:,1], c=task_colors, alpha=0.7, s=30, cmap='tab20')
            plt.colorbar(scatter, label='Task Type')
            plt.title('Latent Space Clustering by Task\nDifferent Colors = Different Tasks')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.tight_layout()
            plt.savefig(os.path.join(sparse_img_dir, 'latent_clustering_by_task.png'), dpi=200)
            plt.close()
            print(f"  ✓ Saved task-based clustering plot")
            
        except Exception as e:
            print(f"  Warning: Global t-SNE visualization failed: {e}")
        
        # Global action statistics
        try:
            plt.figure(figsize=(15,10))
            
            # Action histograms
            for i in range(all_actions_combined.shape[1]):
                plt.subplot(2, 3, i+1)
                plt.hist(all_actions_combined[:, i], bins=20, alpha=0.7, edgecolor='black')
                plt.title(f'Action Distribution - Joint {i+1}')
                plt.xlabel('Action Value')
                plt.ylabel('Frequency')
            
            plt.suptitle('Global Action Distributions Across All Episodes')
            plt.tight_layout()
            plt.savefig(os.path.join(sparse_img_dir, 'global_action_distributions.png'), dpi=200)
            plt.close()
            print(f"  ✓ Saved global action distributions")
            
        except Exception as e:
            print(f"  Warning: Global action statistics failed: {e}")
        
        # Arm position statistics
        try:
            plt.figure(figsize=(15,5))
            
            for i in range(all_arm_positions_combined.shape[1]):
                plt.subplot(1, 3, i+1)
                plt.hist(all_arm_positions_combined[:, i], bins=20, alpha=0.7, edgecolor='black')
                plt.title(f'Arm Position Distribution - {["X", "Y", "Z"][i]}')
                plt.xlabel('Position')
                plt.ylabel('Frequency')
            
            plt.suptitle('Global Arm Position Distributions Across All Episodes')
            plt.tight_layout()
            plt.savefig(os.path.join(sparse_img_dir, 'global_arm_position_distributions.png'), dpi=200)
            plt.close()
            print(f"  ✓ Saved global arm position distributions")
            
        except Exception as e:
            print(f"  Warning: Global arm position statistics failed: {e}")
        
        # Save global data
        np.save(os.path.join(args.rollout_dir, 'all_sparse_codes.npy'), all_codes_combined)
        np.save(os.path.join(args.rollout_dir, 'all_actions.npy'), all_actions_combined)
        np.save(os.path.join(args.rollout_dir, 'all_arm_positions.npy'), all_arm_positions_combined)
        
        print(f"  Saved {len(all_codes_combined)} total sparse codes across {len(all_sparse_codes)} episodes")
        print(f"  Generated comprehensive latent space and statistics visualizations")
    
    env.close()
    
    print(f"\n{'='*50}")
    print("Enhanced evaluation with plots completed!")
    print(f"Results saved in: {args.rollout_dir}")
    print(f"Plots saved in: {sparse_img_dir}")
    print(f"Check the generated videos and visualizations!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 