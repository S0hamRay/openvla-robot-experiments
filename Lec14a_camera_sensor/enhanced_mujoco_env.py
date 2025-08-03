#!/usr/bin/env python3
"""
Enhanced MuJoCo environment with robotic arm and interactive objects
"""

import os
import numpy as np
import mujoco as mj
import gymnasium as gym
from gymnasium import spaces

class EnhancedMujocoEnv(gym.Env):
    """Enhanced MuJoCo environment with robotic arm and objects"""
    
    def __init__(self, enable_rendering=True, xml_path=None):
        super().__init__()
        
        # Camera settings (set these first)
        self.image_height = 128
        self.image_width = 128
        
        # Set up rendering backend
        self.enable_rendering = enable_rendering
        self._setup_rendering_backend()
        
        # Load the enhanced model
        if xml_path is None:
            xml_path = os.path.join(os.path.dirname(__file__), 'enhanced_robot_arm.xml')
        
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        
        # Test rendering capabilities
        self._test_rendering_capabilities()
        
        # Set up action and observation spaces
        self._setup_spaces()
        
        # Initialize episode variables
        self.episode_steps = 0
        self.max_episode_steps = 100
        
        print(f"✓ Enhanced MuJoCo environment initialized")
        print(f"  - Model: {xml_path}")
        print(f"  - Rendering: {'Available' if self.rendering_available else 'Not available'}")
        print(f"  - Action space: {self.action_space.shape}")
        print(f"  - Observation space: {self.observation_space.spaces}")
    
    def _setup_rendering_backend(self):
        """Set up MuJoCo rendering backend"""
        os.environ['MUJOCO_GL'] = 'egl'
        print("Using MuJoCo EGL backend")
        print("  - MUJOCO_GL=egl")
        
        # Test if the rendering backend works
        try:
            test_xml_path = os.path.join(os.path.dirname(__file__), 'enhanced_robot_arm.xml')
            test_model = mj.MjModel.from_xml_path(test_xml_path)
            test_data = mj.MjData(test_model)
            
            # Test only the new renderer API
            test_renderer = mj.Renderer(test_model, height=64, width=64)
            test_renderer.update_scene(test_data, camera=0)
            test_image = test_renderer.render()
            
            print("✓ Rendering backend initialized successfully")
            
        except Exception as e:
            print(f"⚠ Rendering backend failed: {e}")
            print("   Will fall back to dummy image generation if needed")
    
    def _test_rendering_capabilities(self):
        """Test if MuJoCo rendering is available"""
        try:
            # Ensure the scene is properly set up
            mj.mj_forward(self.model, self.data)
            
            # Test only the new renderer API
            renderer = mj.Renderer(self.model, height=self.image_height, width=self.image_width)
            renderer.update_scene(self.data, camera=0)
            image = renderer.render()
            
            # Check if the rendered image is valid
            if image is not None and image.size > 0:
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                
                # Check if image has meaningful content
                if image.max() > 0:
                    self.rendering_available = True
                    self.rendering_backend = "egl_new_api"
                    print(f"✓ MuJoCo rendering available (new API)")
                    print(f"  Image shape: {image.shape}, min/max: {image.min()}/{image.max()}")
                    return
                else:
                    print(f"⚠ Rendered image is all black (max: {image.max()})")
                    raise Exception("Rendered image is all black")
            else:
                raise Exception("Rendered image is None or empty")
                
        except Exception as e:
            self.rendering_available = False
            self.rendering_backend = None
            print(f"⚠ MuJoCo rendering not available: {e}")
            print("   Using dummy images instead")
    
    def _setup_spaces(self):
        """Set up action and observation spaces"""
        # Action space: 4 joint positions for the arm + 1 gripper
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        
        # Observation space: image + joint positions + object positions
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255, shape=(self.image_height, self.image_width, 3), 
                dtype=np.uint8
            ),
            'joint_positions': spaces.Box(
                low=-np.pi, high=np.pi, shape=(4,), dtype=np.float32
            ),
            'object_positions': spaces.Box(
                low=-10, high=10, shape=(15,), dtype=np.float32  # 5 objects * 3 coordinates
            ),
            'arm_position': spaces.Box(
                low=-10, high=10, shape=(3,), dtype=np.float32
            )
        })
    
    def _generate_camera_image(self):
        """Generate camera image using MuJoCo rendering"""
        if not self.enable_rendering or not self.rendering_available:
            print("⚠ Rendering disabled or not available, using dummy image")
            return self._generate_dummy_image()
        
        # Try to find the robot_camera first, then other cameras
        camera_id = 0
        try:
            # Look for the robot_camera first (best view of the arm)
            camera_id = self.model.camera('robot_camera').id
            print(f"✓ Found robot_camera with ID: {camera_id}")
        except Exception as e:
            print(f"⚠ robot_camera not found: {e}")
            try:
                # Look for the close_camera as fallback
                camera_id = self.model.camera('close_camera').id
                print(f"✓ Found close_camera with ID: {camera_id}")
            except Exception as e2:
                print(f"⚠ close_camera not found: {e2}")
                # Fallback to first camera if no named cameras found
                if self.model.ncam > 0:
                    camera_id = 0
                    print(f"⚠ Using first camera (ID: {camera_id})")
                else:
                    print("⚠ No cameras found in model")
                    return self._generate_dummy_image()
        
        # Use new renderer API only
        try:
            renderer = mj.Renderer(self.model, height=self.image_height, width=self.image_width)
            
            # Update the renderer with current state
            renderer.update_scene(self.data, camera=camera_id)
            
            # Render the image
            image = renderer.render()
            print(f"✓ Rendered image: shape={image.shape}, dtype={image.dtype}")
            
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
                print(f"✓ Converted to uint8: min={image.min()}, max={image.max()}")
            
            # Validate the image
            if image is not None and image.size > 0 and image.max() > 0:
                print(f"✓ Image validation passed: min={image.min()}, max={image.max()}, std={image.std():.1f}")
                return image
            else:
                print(f"⚠ Rendered image is invalid: min={image.min()}, max={image.max()}, std={image.std():.1f}")
                return self._generate_dummy_image()
            
        except Exception as e:
            print(f"Rendering failed: {e}, using dummy image")
            return self._generate_dummy_image()
    
    def _generate_dummy_image(self):
        """Generate a dummy image for testing"""
        # Create a more interesting dummy image
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        # Dark gray background
        image[:, :] = [40, 40, 40]
        
        # Add some colored circles based on joint positions
        center_x, center_y = self.image_width // 2, self.image_height // 2
        
        # Draw circles for each joint
        for i in range(4):
            if i < len(self.data.qpos):
                joint_pos = self.data.qpos[i]
                # Map joint position to circle position
                circle_x = int(center_x + joint_pos * 20)
                circle_y = int(center_y + (i * 30))
                
                # Different colors for different joints
                colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
                color = colors[i % len(colors)]
                
                # Draw circle
                for dx in range(-10, 11):
                    for dy in range(-10, 11):
                        if dx*dx + dy*dy <= 100:  # Circle radius
                            x, y = circle_x + dx, circle_y + dy
                            if 0 <= x < self.image_width and 0 <= y < self.image_height:
                                image[y, x] = color
        
        return image
    
    def _get_joint_positions(self):
        """Get current joint positions"""
        # Get the 4 arm joint positions
        joint_positions = []
        for i in range(4):
            if i < len(self.data.qpos):
                joint_positions.append(self.data.qpos[i])
            else:
                joint_positions.append(0.0)
        return np.array(joint_positions, dtype=np.float32)
    
    def _get_object_positions(self):
        """Get positions of all objects"""
        object_positions = []
        
        # Get positions of all bodies (objects)
        for i in range(self.model.nbody):
            body_pos = self.data.xpos[i]
            object_positions.extend(body_pos)
        
        # Pad to fixed size
        while len(object_positions) < 15:
            object_positions.append(0.0)
        
        return np.array(object_positions[:15], dtype=np.float32)
    
    def _get_arm_position(self):
        """Get the end effector position"""
        try:
            # Get the gripper position
            gripper_id = self.model.body('gripper').id
            arm_pos = self.data.xpos[gripper_id]
            return np.array(arm_pos, dtype=np.float32)
        except:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    def reset(self, seed=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        # Reset MuJoCo data
        mj.mj_resetData(self.model, self.data)
        
        # Set initial joint positions
        initial_positions = [0.0, 0.5, -0.3, 0.2]  # Slightly bent arm
        for i in range(min(4, len(initial_positions))):
            if i < len(self.data.qpos):
                self.data.qpos[i] = initial_positions[i]
        
        # Forward the model
        mj.mj_forward(self.model, self.data)
        
        # Reset episode variables
        self.episode_steps = 0
        
        # Generate observation
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action):
        """Take a step in the environment"""
        # Apply action to joints
        for i in range(min(4, len(action))):
            if i < len(self.data.ctrl):
                # Scale action to joint limits
                self.data.ctrl[i] = action[i] * np.pi
        
        # Step the simulation
        mj.mj_step(self.model, self.data)
        
        # Update episode step counter
        self.episode_steps += 1
        
        # Generate observation
        observation = self._get_observation()
        
        # Calculate reward (simple distance-based reward)
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.episode_steps >= self.max_episode_steps
        
        return observation, reward, done, False, {}
    
    def _get_observation(self):
        """Get current observation"""
        # Generate camera image
        image = self._generate_camera_image()
        
        # Get joint positions
        joint_positions = self._get_joint_positions()
        
        # Get object positions
        object_positions = self._get_object_positions()
        
        # Get arm position
        arm_position = self._get_arm_position()
        
        return {
            'image': image,
            'joint_positions': joint_positions,
            'object_positions': object_positions,
            'arm_position': arm_position
        }
    
    def _calculate_reward(self):
        """Calculate reward based on arm-object interactions"""
        # Simple reward: encourage arm movement and object interaction
        reward = 0.0
        
        # Reward for arm movement
        joint_velocities = self.data.qvel[:4]
        movement_reward = np.mean(np.abs(joint_velocities))
        reward += movement_reward * 0.1
        
        # Reward for being close to objects
        arm_pos = self._get_arm_position()
        for i in range(self.model.nbody):
            if i > 0:  # Skip the base body
                object_pos = self.data.xpos[i]
                distance = np.linalg.norm(arm_pos - object_pos)
                if distance < 0.3:  # Close to object
                    reward += 0.5
        
        return reward
    
    def render(self):
        """Render the current state"""
        return self._generate_camera_image()
    
    def close(self):
        """Clean up resources"""
        pass 