#!/usr/bin/env python3
"""
Enhanced MuJoCo evaluation script with robotic arm and objects
"""

import os
import sys
import argparse
import numpy as np
import imageio
from pathlib import Path

# Add parent directories to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))  # For Lec14a_camera_sensor

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced MuJoCo evaluation with robotic arm')
    parser.add_argument('--pretrained_checkpoint', type=str, required=True,
                       help='Path to pretrained checkpoint')
    parser.add_argument('--num_episodes', type=int, default=3,
                       help='Number of episodes to run')
    parser.add_argument('--max_steps', type=int, default=50,
                       help='Maximum steps per episode')
    parser.add_argument('--rollout_dir', type=str, default='rollouts_enhanced',
                       help='Directory to save rollouts')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Import the enhanced environment
    from Lec14a_camera_sensor.enhanced_mujoco_env import EnhancedMujocoEnv
    
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
    
    # Run episodes
    print(f"\nRunning {args.num_episodes} episodes with enhanced robotic arm environment...")
    
    for episode in range(args.num_episodes):
        print(f"\nEpisode {episode + 1}/{args.num_episodes}")
        
        # Reset environment
        obs, _ = env.reset()
        
        # Store frames for video
        frames = []
        actions = []
        arm_positions = []
        
        # Run episode
        for step in range(args.max_steps):
            # Generate random action for the 5-DOF arm
            action = np.random.uniform(-0.3, 0.3, env.action_space.shape[0])
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            
            # Store data
            frames.append(obs['image'].copy())
            actions.append(action.copy())
            arm_positions.append(obs['arm_position'].copy())
            
            print(f"  Step {step+1}: reward={reward:.3f}, arm_pos={obs['arm_position']}")
            
            if done:
                print(f"  Episode ended after {step+1} steps")
                break
        
        # Save episode data
        episode_dir = os.path.join(args.rollout_dir, f"episode_{episode}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Save video
        if frames:
            video_path = os.path.join(episode_dir, f"episode_{episode}_success_False.mp4")
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
    
    env.close()
    
    print(f"\n{'='*50}")
    print("Enhanced evaluation completed!")
    print(f"Results saved in: {args.rollout_dir}")
    print(f"Check the generated videos to see the robotic arm in action!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 