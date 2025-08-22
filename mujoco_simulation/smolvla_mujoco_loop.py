#!/usr/bin/env python3
"""
SmolVLA + MuJoCo Integration Loop
This script connects SmolVLA policy to MuJoCo simulation for closed-loop control.
"""

import os
import numpy as np
import cv2
import torch
import mujoco
import imageio

# Important: set before importing mujoco on macOS
os.environ.setdefault("MUJOCO_GL", "glfw")

def main():
    print("Setting up SmolVLA + MuJoCo integration...")
    
    # 1) Load SmolVLA policy
    try:
        from lerobot.common.policies.smolvla import SmolVLAPolicy
        print("✓ SmolVLA policy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SmolVLA policy: {e}")
        print("Make sure you have LeRobot installed and are in the smolvla conda environment")
        return
    
    # Load the policy
    try:
        policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").eval()
        print("✓ SmolVLA policy loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load policy: {e}")
        return
    
    # Set device (CPU/MPS/GPU as available)
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    policy.to(device)
    
    # 2) MuJoCo setup
    print("\nSetting up MuJoCo simulation...")
    m = mujoco.MjModel.from_xml_path("manipulator_scene.xml")
    d = mujoco.MjData(m)
    
    # Set initial joint positions
    d.qpos[:] = [0.0, 0.3, 0.5, 0.0]  # Base, shoulder, elbow, wrist
    mujoco.mj_forward(m, d)
    
    # Rendering parameters
    W, H = 320, 240  # Inference resolution (adjust based on policy requirements)
    print(f"Rendering resolution: {W}x{H}")
    
    # Create offscreen GL context
    ctx = mujoco.GLContext(W, H)
    ctx.make_current()
    
    # Setup visualization objects
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    
    # Get camera ID by name
    cam_name = "ee_cam"
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        print(f"✗ Camera '{cam_name}' not found")
        return
    
    cam.fixedcamid = cam_id
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    print(f"✓ Using camera '{cam_name}' with ID {cam_id}")
    
    # Setup rendering context
    scn = mujoco.MjvScene(m, maxgeom=1_000_000)
    rcon = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
    
    # Set offscreen buffer
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, rcon)
    mujoco.mjr_resizeOffscreen(W, H, rcon)
    rect = mujoco.MjrRect(0, 0, W, H)
    
    # Image buffer
    rgb = np.empty((H, W, 3), dtype=np.uint8)
    
    # 3) Helper: render ee_cam to CHW float tensor
    def render_obs():
        """Render observation from ee_cam and convert to tensor format."""
        mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        mujoco.mjr_render(rect, scn, rcon)
        mujoco.mjr_readPixels(rgb, None, rect, rcon)
        
        # Fix OpenGL origin and convert to tensor
        frame = np.flipud(rgb)
        
        # Resize to model's expected size (adjust based on policy requirements)
        target_size = 224  # Common input size for vision models
        img = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        # Convert to tensor: HWC uint8 → CHW float
        img = torch.from_numpy(img).to(device)
        img = img.permute(2, 0, 1).float() / 255.0
        
        return img
    
    # 4) Control loop
    print("\nStarting control loop...")
    instruction = "Look at the red block and reach toward it."
    print(f"Instruction: {instruction}")
    
    # Get control ranges
    ctrl_low = np.array([a.ctrlrange[0] for a in m.actuator_ctrlrange])
    ctrl_high = np.array([a.ctrlrange[1] for a in m.actuator_ctrlrange])
    print(f"Control ranges: {ctrl_low} to {ctrl_high}")
    
    # Simulation parameters
    sim_duration = 30  # seconds
    sim_fps = 30
    total_steps = sim_duration * sim_fps
    
    # Optional: record video
    record_video = True
    frames = []
    
    try:
        for t in range(total_steps):
            # Render observation
            obs_img = render_obs().unsqueeze(0)  # (B=1, C, H, W)
            
            # Get proprioceptive state
            proprio = torch.from_numpy(
                np.concatenate([d.qpos.copy(), d.qvel.copy()])
            ).float().to(device).unsqueeze(0)
            
            # Get action from SmolVLA policy
            with torch.no_grad():
                try:
                    out = policy.predict(
                        images={"wrist": obs_img},  # Key name should match training config
                        text=[instruction],
                        proprio=proprio  # Optional, if checkpoint expects it
                    )
                    action = out["actions"][0].cpu().numpy()
                except Exception as e:
                    print(f"✗ Policy prediction failed at step {t}: {e}")
                    # Use safe fallback action
                    action = np.zeros(len(ctrl_low))
            
            # Map action to controls (clip to safe ranges)
            d.ctrl[:] = np.clip(action, ctrl_low, ctrl_high)
            
            # Step simulation
            mujoco.mj_step(m, d)
            
            # Optional: record frame for video
            if record_video and t % 3 == 0:  # Record every 3rd frame to reduce file size
                # Render at higher resolution for video
                video_W, video_H = 640, 480
                video_rect = mujoco.MjrRect(0, 0, video_W, video_H)
                video_rgb = np.empty((video_H, video_W, 3), dtype=np.uint8)
                
                # Temporarily resize offscreen buffer for video
                mujoco.mjr_resizeOffscreen(video_W, video_H, rcon)
                mujoco.mjr_render(video_rect, scn, rcon)
                mujoco.mjr_readPixels(video_rgb, None, video_rect, rcon)
                
                # Reset to inference resolution
                mujoco.mjr_resizeOffscreen(W, H, rcon)
                
                frames.append(np.flipud(video_rgb))
            
            # Progress indicator
            if t % sim_fps == 0:
                elapsed = t / sim_fps
                print(f"  Step {t}/{total_steps} ({elapsed:.1f}s elapsed)")
        
        print(f"\n✓ Control loop completed successfully!")
        
        # Save video if frames were recorded
        if record_video and frames:
            output_path = "smolvla_control_video.mp4"
            print(f"Saving control video to {output_path}...")
            
            imageio.mimwrite(
                output_path, frames, fps=10, quality=8,
                codec='libx264', pixelformat='yuv420p'
            )
            
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                print(f"✓ Video saved: {output_path} ({file_size / 1024 / 1024:.2f} MB)")
        
    except KeyboardInterrupt:
        print("\nControl loop interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during control loop: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        rcon.free()
        ctx.free()
        print("✓ Resources freed")

if __name__ == "__main__":
    main()
