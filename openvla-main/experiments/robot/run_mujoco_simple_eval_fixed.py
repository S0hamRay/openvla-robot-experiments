import os
import numpy as np
import imageio
from dataclasses import dataclass
from typing import Union, Optional, List
from pathlib import Path
import torch
import torch.nn.functional as F
import gymnasium as gym  # Fixed: use gymnasium instead of gym
import random

# Use headless-compatible environment to fix rendering issues
import sys
# Add parent directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))  # For Lec14a_camera_sensor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))  # For openvla-main

# Import the real MuJoCo environment
try:
    from Lec14a_camera_sensor.mujoco_gym_env_headless import MujocoSimpleEnvHeadless
    print("✓ Successfully imported MujocoSimpleEnvHeadless")
except ImportError as e:
    print(f"❌ Could not import MujocoSimpleEnvHeadless: {e}")
    sys.exit(1)

# Try to import robot utilities, with fallbacks
try:
    from experiments.robot.robot_utils import (
        get_model,
        get_action,
        get_image_resize_size,
        set_seed_everywhere,
        invert_gripper_action,
    )
    from experiments.robot.openvla_utils import get_processor
    print("✓ Successfully imported robot utilities")
except ImportError as e:
    print(f"Warning: Could not import robot utilities: {e}")
    print("Using simplified fallback functions")
    
    def set_seed_everywhere(seed: int):
        """Sets the random seed for Python, NumPy, and PyTorch functions."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
    
    def get_model(cfg):
        """Simplified model loading"""
        print("Using simplified model loading")
        return None
    
    def get_action(cfg, model, obs, task_label, processor=None):
        """Simplified action generation with better heuristics"""
        # Generate action based on task description and current state
        action = np.zeros(7, dtype=np.float32)
        
        # Get current position from proprioceptive state
        current_pos = obs['state'][:3] if len(obs['state']) >= 3 else np.zeros(3)
        
        # Simple heuristic-based action generation
        if "move" in task_label.lower() or "position" in task_label.lower():
            # Move toward target
            target_pos = np.array([0.0, 0.0, 0.5])  # Target position
            direction = target_pos - current_pos
            direction = direction / (np.linalg.norm(direction) + 1e-6)  # Normalize
            
            # Apply movement in the direction of target
            action[:3] = direction * 0.1  # Small movement step
            
        elif "pick" in task_label.lower() or "grasp" in task_label.lower():
            # Gripper action for picking
            action[-1] = -0.5  # Open gripper
            action[:3] = np.random.uniform(-0.05, 0.05, 3)  # Small movement
            
        elif "place" in task_label.lower():
            # Movement action for placing
            action[:3] = np.random.uniform(-0.05, 0.05, 3)  # Small movement
            action[-1] = 0.5  # Close gripper
            
        elif "rotate" in task_label.lower() or "turn" in task_label.lower():
            # Rotational action
            action[3:6] = np.random.uniform(-0.1, 0.1, 3)  # Rotation
            
        else:
            # Default: small random movement toward target
            target_pos = np.array([0.0, 0.0, 0.5])
            direction = target_pos - current_pos
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            action[:3] = direction * 0.05
        
        return action, None
    
    def invert_gripper_action(action):
        """Invert gripper action"""
        action[-1] = -action[-1]
        return action
    
    def get_processor(cfg):
        """Simplified processor"""
        return None

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for cluster
import matplotlib.pyplot as plt

@dataclass
class GenerateConfig:
    # Model parameters
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = False
    unnorm_key: Optional[str] = None
    # Env parameters
    num_episodes: int = 5
    max_steps: int = 50  # Increased from 100 to ensure more than 14 actions per episode
    image_width: int = 128
    image_height: int = 128
    seed: int = 42
    # Output
    rollout_dir: str = "./rollouts_mujoco_simple"


def get_varied_task_commands() -> List[str]:
    """
    Returns a list of varied language commands for the robot.
    These commands provide different instructions to test the model's ability
    to understand and execute various tasks.
    """
    return [
        "move the block to the center",
        "pick up the block and place it on the left",
        "grasp the object and move it to the right side",
        "lift the block and position it in front",
        "take the block and place it behind",
        "manipulate the object to the upper area",
        "move the block downward",
        "rotate the block clockwise",
        "turn the object counterclockwise",
        "push the block forward",
        "pull the object backward",
        "slide the block to the side",
        "position the block at an angle",
        "arrange the object in a specific location",
        "relocate the block to a new position",
        "transport the object to the target area",
        "displace the block from its current location",
        "reposition the object as requested",
        "shift the block to the desired spot",
        "move the object with precision"
    ]


def get_random_task_command() -> str:
    """Returns a random task command from the varied list."""
    commands = get_varied_task_commands()
    return random.choice(commands)


def save_rollout_video(rollout_images, idx, success, rollout_dir):
    os.makedirs(rollout_dir, exist_ok=True)
    mp4_path = os.path.join(rollout_dir, f"episode_{idx}_success_{success}.mp4")
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_checkpoint', type=str, required=True)
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--max_steps', type=int, default=50)  # Increased default
    parser.add_argument('--image_width', type=int, default=128)
    parser.add_argument('--image_height', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rollout_dir', type=str, default="./rollouts_mujoco_simple")
    parser.add_argument('--center_crop', action='store_true')
    parser.add_argument('--load_in_8bit', action='store_true')
    parser.add_argument('--load_in_4bit', action='store_true')
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    args = parser.parse_args()

    cfg = GenerateConfig(
        pretrained_checkpoint=args.pretrained_checkpoint,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        image_width=args.image_width,
        image_height=args.image_height,
        seed=args.seed,
        rollout_dir=args.rollout_dir,
        center_crop=args.center_crop,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
    )
    
    # Force CPU if requested
    if args.force_cpu:
        torch.cuda.is_available = lambda: False
        print("Forcing CPU usage as requested")

    set_seed_everywhere(cfg.seed)

    # CLEAR THE ROLLOUT DIRECTORY BEFORE GENERATING NEW VIDEOS
    print(f"Clearing rollout directory: {cfg.rollout_dir}")
    if os.path.exists(cfg.rollout_dir):
        import shutil
        shutil.rmtree(cfg.rollout_dir)
        print(f"✓ Removed existing directory: {cfg.rollout_dir}")
    os.makedirs(cfg.rollout_dir, exist_ok=True)
    print(f"✓ Created fresh directory: {cfg.rollout_dir}")

    # Load model and processor
    try:
        model = get_model(cfg)
        processor = get_processor(cfg)
        cfg.unnorm_key = "bridge_orig"  # Use bridge_orig dataset statistics for normalization
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
        model = None
        processor = None

    # Create real MuJoCo environment with proper rendering
    print("Creating MuJoCo environment with real rendering...")
    env = MujocoSimpleEnvHeadless(
        image_width=cfg.image_width, 
        image_height=cfg.image_height,
        enable_rendering=True  # Enable real rendering
    )
    
    # Print environment info
    state_info = env.get_state_info()
    print(f"Environment info: {state_info}")

    # === Train Sparse Autoencoder ===
    action_dim = 7  # Default action dimension
    # Use the same device as the model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training sparse autoencoder on {device}")
    sae = train_sparse_autoencoder(env, action_dim, hidden_dim=25, num_actions=1000, epochs=20, batch_size=64, device=device)

    # === Attach SAE to model ===
    if model is not None:
        def patch_predict_action(model_obj, sae_obj):
            model_obj.sparse_autoencoder = sae_obj
            try:
                from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
                model_obj.predict_action = OpenVLAForActionPrediction.predict_action.__get__(model_obj, type(model_obj))
            except ImportError:
                print("Warning: Could not patch predict_action method")

        patch_predict_action(model, sae)
        for attr in ["model", "base_model", "module"]:
            if hasattr(model, attr):
                patch_predict_action(getattr(model, attr), sae)

    print("Sparse autoencoder training and attachment complete")

    # Utility: Recursively attach SAE to all submodules
    def attach_sae_recursive(obj, sae, depth=0):
        if depth > 5:
            return
        for attr in ["model", "base_model", "module"]:
            if hasattr(obj, attr):
                sub = getattr(obj, attr)
                setattr(sub, "sparse_autoencoder", sae)
                attach_sae_recursive(sub, sae, depth+1)
        setattr(obj, "sparse_autoencoder", sae)

    # Attach the trained SAE recursively
    if model is not None:
        attach_sae_recursive(model, sae)

    # After training SAE, store it globally for access in get_vla_action
    import builtins
    builtins.GLOBAL_TRAINED_SAE = sae

    # Directory for sparse code images
    sparse_img_dir = os.path.join(cfg.rollout_dir, "../sparse_code_images")
    os.makedirs(sparse_img_dir, exist_ok=True)

    all_sparse_codes = []
    all_actions = []
    all_tasks = []
    
    for ep in range(cfg.num_episodes):
        print(f"Starting episode {ep + 1}/{cfg.num_episodes}")
        
        # Select a varied task command for this episode
        task_command = get_random_task_command()
        print(f"  Task: '{task_command}'")
        
        obs, _ = env.reset()
        done = False
        t = 0
        rollout_images = []
        # Add the initial observation to the video
        rollout_images.append(obs["image"])
        sparse_codes_episode = []
        actions_episode = []
        success = False
        
        while not done and t < cfg.max_steps:
            # Prepare observation for model using the camera image from the environment
            observation = {
                "full_image": obs["image"],
                "state": obs["proprio"],
            }
            
            # Query model for action using varied task command
            try:
                action, sparse_code = get_action(cfg, model, observation, task_label=task_command, processor=processor)
            except Exception as e:
                print(f"  Warning: Could not get action from model: {e}")
                # Generate action based on task description
                action, sparse_code = get_action(cfg, None, observation, task_label=task_command, processor=None)
            
            if sparse_code is not None:
                sparse_codes_episode.append(sparse_code)
                actions_episode.append(action.copy())
            
            # Invert gripper action if needed (OpenVLA convention)
            action = invert_gripper_action(action)
            
            # Step environment
            obs, reward, done, _, info = env.step(action)
            # Save the current observation's image (which is the real camera image)
            rollout_images.append(obs["image"])  # Save the real camera image
            t += 1
            
            # Print progress
            if t % 10 == 0:
                print(f"    Step {t}: Reward = {reward:.3f}, Done = {done}")
        
        # Check if episode was successful
        success = info.get('success', False) if 'info' in locals() else False
        print(f"  Episode {ep + 1} completed with {t} actions, Success = {success}")
        save_rollout_video(rollout_images, ep, success=success, rollout_dir=cfg.rollout_dir)
        
        if sparse_codes_episode:
            sparse_codes_arr = np.concatenate(sparse_codes_episode, axis=0)
            actions_arr = np.array(actions_episode)
            
            # Save data
            np.save(os.path.join(cfg.rollout_dir, f"episode_{ep}_sparse_codes.npy"), sparse_codes_arr)
            np.save(os.path.join(cfg.rollout_dir, f"episode_{ep}_actions.npy"), actions_arr)
            
            # Collect for global analysis
            all_sparse_codes.append(sparse_codes_arr)
            all_actions.append(actions_arr)
            all_tasks.extend([task_command] * len(sparse_codes_arr))
            
            # --- Episode-specific Visualization ---
            # t-SNE of sparse codes
            try:
                from sklearn.manifold import TSNE
                codes_2d = TSNE(n_components=2, perplexity=min(30, len(sparse_codes_arr)-1), random_state=42).fit_transform(sparse_codes_arr)
                plt.figure(figsize=(10,8))
                plt.scatter(codes_2d[:,0], codes_2d[:,1], alpha=0.7, s=30, c=range(len(codes_2d)), cmap='viridis')
                plt.colorbar(label='Time Step')
                plt.title(f'Latent Space Trajectory - Episode {ep + 1}\nTask: {task_command}')
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.tight_layout()
                plt.savefig(os.path.join(sparse_img_dir, f'episode_{ep}_latent_trajectory.png'), dpi=200)
                plt.close()
            except Exception as e:
                print(f"  Warning: t-SNE visualization failed for episode {ep + 1}: {e}")
            
            # Action space trajectory
            try:
                plt.figure(figsize=(12,6))
                for i in range(actions_arr.shape[1]):
                    plt.subplot(2, 4, i+1)
                    plt.plot(actions_arr[:, i], 'o-', alpha=0.7)
                    plt.title(f'Action Dim {i+1}')
                    plt.xlabel('Time Step')
                    plt.ylabel('Action Value')
                plt.suptitle(f'Action Trajectory - Episode {ep + 1}\nTask: {task_command}')
                plt.tight_layout()
                plt.savefig(os.path.join(sparse_img_dir, f'episode_{ep}_action_trajectory.png'), dpi=200)
                plt.close()
            except Exception as e:
                print(f"  Warning: Action trajectory visualization failed for episode {ep + 1}: {e}")
            
            # Sparse code activation heatmap
            try:
                plt.figure(figsize=(10,6))
                plt.imshow(sparse_codes_arr.T, aspect='auto', cmap='viridis')
                plt.colorbar(label='Activation Value')
                plt.title(f'Sparse Code Activations - Episode {ep + 1}\nTask: {task_command}')
                plt.xlabel('Time Step')
                plt.ylabel('Latent Dimension')
                plt.tight_layout()
                plt.savefig(os.path.join(sparse_img_dir, f'episode_{ep}_activation_heatmap.png'), dpi=200)
                plt.close()
            except Exception as e:
                print(f"  Warning: Activation heatmap failed for episode {ep + 1}: {e}")
    
    # --- Global Analysis Across All Episodes ---
    if all_sparse_codes:
        print("\nGenerating global latent space analysis...")
        
        # Combine all sparse codes
        all_codes_combined = np.concatenate(all_sparse_codes, axis=0)
        all_actions_combined = np.concatenate(all_actions, axis=0)
        
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
            
        except Exception as e:
            print(f"  Warning: Global t-SNE visualization failed: {e}")
        
        # Save global data
        np.save(os.path.join(cfg.rollout_dir, 'all_sparse_codes.npy'), all_codes_combined)
        np.save(os.path.join(cfg.rollout_dir, 'all_actions.npy'), all_actions_combined)
        
        print(f"  Saved {len(all_codes_combined)} total sparse codes across {len(all_sparse_codes)} episodes")
        print(f"  Generated comprehensive latent space visualizations")
    
    print(f"\nEvaluation completed: {cfg.num_episodes} episodes, {cfg.max_steps} max steps per episode")
    env.close()

if __name__ == "__main__":
    main() 