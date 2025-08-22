#!/usr/bin/env python

"""
MuJoCo Simulation with Manipulator, Camera, and Blocks
"""

import mujoco
import numpy as np
import time
import cv2
from pathlib import Path


class ManipulatorSimulation:
    def __init__(self, xml_path):
        """Initialize the MuJoCo simulation."""
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        
        # Get joint names and IDs
        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        self.joint_ids = {name: i for i, name in enumerate(self.joint_names)}
        
        # Get actuator names and IDs
        self.actuator_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        self.actuator_ids = {name: i for i, name in enumerate(self.actuator_names)}
        
        # Get camera name and ID
        self.camera_name = "main_camera"
        self.camera_id = self.model.camera(self.camera_name).id
        
        # Get block body IDs
        self.block_names = ["block", "block2", "block3"]
        self.block_ids = {name: self.model.body(name).id for name in self.block_names}
        
        print(f"Simulation initialized with {self.model.njnt} joints and {self.model.nu} actuators")
        print(f"Joint names: {self.joint_names}")
        print(f"Actuator names: {self.actuator_names}")
        print(f"Block names: {self.block_names}")
    
    def get_joint_positions(self):
        """Get current joint positions."""
        return {name: self.data.qpos[self.joint_ids[name]] for name in self.joint_names}
    
    def get_joint_velocities(self):
        """Get current joint velocities."""
        return {name: self.data.qvel[self.joint_ids[name]] for name in self.joint_names}
    
    def set_joint_positions(self, positions):
        """Set joint positions."""
        for name, pos in positions.items():
            if name in self.joint_ids:
                self.data.qpos[self.joint_ids[name]] = pos
    
    def set_joint_velocities(self, velocities):
        """Set joint velocities."""
        for name, vel in velocities.items():
            if name in self.joint_ids:
                self.data.qvel[self.joint_ids[name]] = vel
    
    def apply_control(self, control_actions):
        """Apply control actions to actuators."""
        for name, action in control_actions.items():
            if name in self.actuator_ids:
                self.data.ctrl[self.actuator_ids[name]] = action
    
    def get_camera_image(self):
        """Get image from the camera."""
        # Render the scene
        mujoco.mj_forward(self.model, self.data)
        
        # Get camera image
        width = self.model.camera(self.camera_id).res[0]
        height = self.model.camera(self.camera_id).res[1]
        
        # Create camera viewport
        viewport = mujoco.MjrRect(0, 0, width, height)
        
        # Create camera context
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        
        # Get camera image
        image = np.empty((height, width, 3), dtype=np.uint8)
        mujoco.mjr_render(viewport, self.model, self.data, context, image)
        
        return image
    
    def get_block_positions(self):
        """Get current positions of all blocks."""
        positions = {}
        for name in self.block_names:
            body_id = self.block_ids[name]
            pos = self.data.xpos[body_id]
            positions[name] = pos.copy()
        return positions
    
    def move_to_target(self, target_pos, duration=2.0, dt=0.01):
        """Move the end effector to a target position using simple IK."""
        print(f"Moving to target position: {target_pos}")
        
        # Simple proportional control for demonstration
        for _ in range(int(duration / dt)):
            # Get current end effector position
            current_pos = self.data.xpos[self.model.body("gripper_base").id]
            
            # Calculate error
            error = target_pos - current_pos
            
            # Simple proportional control (this is a basic implementation)
            # In practice, you'd want proper inverse kinematics
            control_actions = {}
            for i, joint_name in enumerate(self.joint_names[:6]):  # Only arm joints
                if i < 3:  # First 3 joints control position
                    control_actions[f"motor{i+1}"] = error[i] * 100
                else:  # Last 3 joints control orientation
                    control_actions[f"motor{i+1}"] = 0
            
            self.apply_control(control_actions)
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            time.sleep(dt)
    
    def open_gripper(self):
        """Open the gripper."""
        print("Opening gripper")
        self.data.ctrl[self.actuator_ids["gripper_motor"]] = 0.1
    
    def close_gripper(self):
        """Close the gripper."""
        print("Closing gripper")
        self.data.ctrl[self.actuator_ids["gripper_motor"]] = -0.1
    
    def run_demo(self):
        """Run a demonstration of the manipulator capabilities."""
        print("Starting manipulator demonstration...")
        
        # Initial pose
        initial_positions = {
            "joint1": 0.0,      # Base rotation
            "joint2": -0.5,     # Shoulder
            "joint3": 1.0,      # Elbow
            "joint4": 0.0,      # Wrist rotation
            "joint5": -0.5,     # Wrist pitch
            "joint6": 0.0,      # Wrist roll
        }
        
        self.set_joint_positions(initial_positions)
        mujoco.mj_forward(self.model, self.data)
        
        # Demo sequence
        demo_sequence = [
            ("Move to block 1", np.array([0.8, 0, 0.3])),
            ("Move to block 2", np.array([1.0, 0.2, 0.3])),
            ("Move to block 3", np.array([0.9, -0.15, 0.3])),
            ("Return to center", np.array([0.5, 0, 0.5])),
        ]
        
        for step_name, target_pos in demo_sequence:
            print(f"\n{step_name}")
            
            # Move to target
            self.move_to_target(target_pos, duration=1.5)
            
            # Get camera image
            image = self.get_camera_image()
            
            # Display image
            cv2.imshow("Camera View", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(100)
            
            # Show block positions
            block_positions = self.get_block_positions()
            for name, pos in block_positions.items():
                print(f"  {name}: {pos}")
            
            time.sleep(0.5)
        
        cv2.destroyAllWindows()
        print("\nDemonstration completed!")
    
    def run_interactive(self):
        """Run interactive simulation."""
        print("Starting interactive simulation...")
        print("Press 'q' to quit, 'r' to reset, 'c' to capture camera image")
        
        # Reset to initial pose
        self.set_joint_positions({
            "joint1": 0.0, "joint2": -0.5, "joint3": 1.0,
            "joint4": 0.0, "joint5": -0.5, "joint6": 0.0
        })
        
        # Main simulation loop
        while True:
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            
            # Check if viewer is still active
            try:
                # Simple keyboard input handling (basic implementation)
                # In a real application, you'd want more sophisticated input handling
                pass
            except:
                break
            
            time.sleep(0.01)
    
    def close(self):
        """Close the simulation."""
        # The passive viewer will close automatically
        pass


def main():
    """Main function to run the simulation."""
    # Get the path to the XML file
    xml_path = Path(__file__).parent / "manipulator_scene.xml"
    
    if not xml_path.exists():
        print(f"Error: XML file not found at {xml_path}")
        return
    
    # Create and run simulation
    sim = ManipulatorSimulation(str(xml_path))
    
    try:
        # Run demo first
        sim.run_demo()
        
        # Then run interactive mode
        sim.run_interactive()
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
