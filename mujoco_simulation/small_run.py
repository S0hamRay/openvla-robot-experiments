import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time
from pathlib import Path

def record_simulation_video(model, data, output_path="simulation_video.mp4", duration=10.0, fps=30):
    """
    Record a video of the MuJoCo simulation.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        output_path: Path to save the video
        duration: Duration of video in seconds
        fps: Frames per second
    """
    print(f"Recording simulation video for {duration} seconds at {fps} FPS...")
    
    # Get camera dimensions (using the end-effector camera)
    camera_id = model.camera("ee_cam").id
    width, height = 640, 480  # Standard resolution
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Error: Could not open video writer")
        return False
    
    # Calculate total frames
    total_frames = int(duration * fps)
    frame_time = 1.0 / fps
    
    print(f"Recording {total_frames} frames...")
    
    # Record frames
    for frame in range(total_frames):
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Render camera view
        try:
            # Create viewport for rendering
            viewport = mujoco.MjrRect(0, 0, width, height)
            
            # Create camera context
            context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
            
            # Render camera image
            image = np.empty((height, width, 3), dtype=np.uint8)
            mujoco.mjr_render(viewport, model, data, context, mujoco.mjtCamera.mjCAMERA_FIXED, image)
            
            # Convert from RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Write frame to video
            video_writer.write(image_bgr)
            
            # Print progress
            if frame % fps == 0:
                print(f"Recorded frame {frame}/{total_frames} ({(frame/total_frames)*100:.1f}%)")
                
        except Exception as e:
            print(f"Error rendering frame {frame}: {e}")
            # Write a black frame as fallback
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)
            video_writer.write(black_frame)
        
        # Wait for frame time
        time.sleep(frame_time)
    
    # Release video writer
    video_writer.release()
    print(f"Video saved to: {output_path}")
    return True

def run_simulation_with_recording():
    """Run the simulation with video recording."""
    print("Loading MuJoCo model...")
    
    # Load your XML model
    model = mujoco.MjModel.from_xml_path("manipulator_scene.xml")
    data = mujoco.MjData(model)
    
    print(f"Model loaded: {model.model}")
    print(f"Number of bodies: {model.nbody}")
    print(f"Number of joints: {model.njnt}")
    print(f"Number of cameras: {model.ncam}")
    
    # List available cameras
    for i in range(model.ncam):
        cam = model.camera(i)
        print(f"Camera {i}: {cam.name} at position {cam.pos}")
    
    # Set initial joint positions for a nice starting pose
    initial_positions = [0.0, 0.3, 0.5, 0.0]  # [base_yaw, shoulder, elbow, wrist]
    data.qpos[:len(initial_positions)] = initial_positions
    
    # Forward kinematics to update positions
    mujoco.mj_forward(model, data)
    
    print("\nStarting simulation with video recording...")
    
    # Record video first
    video_path = "manipulator_simulation.mp4"
    success = record_simulation_video(
        model, data, 
        output_path=video_path,
        duration=15.0,  # 15 seconds
        fps=30
    )
    
    if success:
        print(f"\nVideo recording completed successfully!")
        print(f"Video saved to: {video_path}")
        
        # Now launch interactive viewer
        print("\nLaunching interactive viewer...")
        print("Press 'q' to quit the viewer")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                time.sleep(0.01)  # Small delay for smooth viewing
    else:
        print("Video recording failed. Launching viewer without recording...")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                time.sleep(0.01)

def test_rendering_system():
    """Test MuJoCo's rendering system with different camera views."""
    print("Testing MuJoCo rendering system...")
    
    # Load model
    model = mujoco.MjModel.from_xml_path("manipulator_scene.xml")
    data = mujoco.MjData(model)
    
    # Test different rendering scenarios
    test_scenarios = [
        {
            "name": "Fixed Camera View",
            "camera_type": mujoco.mjtCamera.mjCAMERA_FIXED,
            "camera_id": 0
        },
        {
            "name": "Tracking Camera View", 
            "camera_type": mujoco.mjtCamera.mjCAMERA_TRACKING,
            "camera_id": 0
        },
        {
            "name": "Free Camera View",
            "camera_type": mujoco.mjtCamera.mjCAMERA_FREE,
            "camera_id": 0
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\nTesting: {scenario['name']}")
        
        # Set some motion in the simulation
        data.qpos[:] = [0.0, 0.5, 0.8, 0.2]
        mujoco.mj_forward(model, data)
        
        # Render a test frame
        try:
            width, height = 640, 480
            viewport = mujoco.MjrRect(0, 0, width, height)
            context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
            
            image = np.empty((height, width, 3), dtype=np.uint8)
            mujoco.mjr_render(viewport, model, data, context, scenario['camera_type'], image)
            
            # Save test frame
            test_image_path = f"test_render_{scenario['name'].replace(' ', '_').lower()}.png"
            cv2.imwrite(test_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"  ✓ Test frame saved: {test_image_path}")
            
        except Exception as e:
            print(f"  ✗ Failed to render: {e}")
    
    print("\nRendering system test completed!")

if __name__ == "__main__":
    print("MuJoCo Simulation with Video Recording")
    print("=" * 50)
    
    # Test rendering system first
    test_rendering_system()
    
    print("\n" + "=" * 50)
    
    # Run main simulation with recording
    run_simulation_with_recording()

