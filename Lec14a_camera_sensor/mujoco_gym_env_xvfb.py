import os
import numpy as np
import gymnasium as gym
import mujoco as mj
from gymnasium import spaces
import subprocess
import time
import signal
import atexit

class MujocoSimpleEnvXvfb(gym.Env):
    def __init__(self, xml_path=None, image_width=128, image_height=128, enable_rendering=True):
        super().__init__()
        
        # Set up Xvfb for headless rendering
        self.xvfb_process = None
        self.display_num = None
        self._setup_xvfb()
        
        # Set up MuJoCo rendering backend with software rendering
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
        os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
        
        if xml_path is None:
            dirname = os.path.dirname(__file__)
            xml_path = os.path.join(dirname, 'block_with_camera.xml')
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        
        # Camera settings
        self.image_width = image_width
        self.image_height = image_height
        self.enable_rendering = enable_rendering
        
        # Try to detect camera
        self.camera_id = 0
        self.rendering_available = False
        
        if self.enable_rendering:
            try:
                # Test if rendering works
                self._test_rendering()
                self.rendering_available = True
                print("✓ MuJoCo rendering is available with Xvfb")
            except Exception as e:
                print(f"⚠ MuJoCo rendering not available: {e}")
                print("   Falling back to dummy images")
                self.rendering_available = False
        
        # Action space: joint positions (qpos)
        n_act = self.model.nu if self.model.nu > 0 else self.model.nq
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_act,), dtype=np.float32
        )
        
        # Observation space: proprio + image
        self.observation_space = spaces.Dict({
            'proprio': spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nq,), dtype=np.float32),
            'image': spaces.Box(low=0, high=255, shape=(self.image_height, self.image_width, 3), dtype=np.uint8)
        })
        
        # Track episode state for more realistic simulation
        self.episode_step = 0
        self.max_episode_steps = 100
        self.success_threshold = 0.3  # Distance threshold for success (increased from 0.1)
        self.target_position = np.array([0.0, 0.0, 0.5])  # Target position for the block

    def _setup_xvfb(self):
        """Set up Xvfb virtual display"""
        try:
            # Find an available display number
            for display_num in range(99, 200):
                display = f":{display_num}"
                try:
                    # Test if display is available
                    result = subprocess.run(['xdpyinfo', '-display', display], 
                                          capture_output=True, timeout=1)
                    if result.returncode != 0:
                        self.display_num = display_num
                        break
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    self.display_num = display_num
                    break
            
            if self.display_num is None:
                raise RuntimeError("Could not find available display number")
            
            # Start Xvfb with better OpenGL support
            display = f":{self.display_num}"
            self.xvfb_process = subprocess.Popen([
                'Xvfb', display, '-screen', '0', '1280x1024x24', '-ac', 
                '+extension', 'GLX', '+extension', 'RANDR', '+extension', 'RENDER',
                '-nolisten', 'tcp', '-dpi', '96'
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait a moment for Xvfb to start
            time.sleep(1)
            
            # Set display environment variable
            os.environ['DISPLAY'] = display
            
            # Register cleanup
            atexit.register(self._cleanup_xvfb)
            
            print(f"✓ Xvfb started on display :{self.display_num}")
            
        except Exception as e:
            print(f"⚠ Failed to start Xvfb: {e}")
            print("   Falling back to dummy images")
            self.xvfb_process = None
            self.display_num = None

    def _cleanup_xvfb(self):
        """Clean up Xvfb process"""
        if self.xvfb_process is not None:
            try:
                self.xvfb_process.terminate()
                self.xvfb_process.wait(timeout=5)
                print(f"✓ Xvfb process terminated")
            except subprocess.TimeoutExpired:
                self.xvfb_process.kill()
                print("⚠ Forced Xvfb process termination")
            except Exception as e:
                print(f"⚠ Error terminating Xvfb: {e}")

    def _test_rendering(self):
        """Test if rendering is available by creating a minimal context"""
        try:
            # Try the new Renderer API first
            renderer = mj.Renderer(self.model, height=self.image_height, width=self.image_width)
            renderer.update_scene(self.data, camera=0)
            test_image = renderer.render()
            print("✓ New renderer API works")
            return True
        except Exception as e1:
            print(f"New renderer failed: {e1}")
            try:
                # Try old API
                scene = mj.MjvScene(self.model, maxgeom=1000)
                context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150)
                print("✓ Old renderer API works")
                return True
            except Exception as e2:
                print(f"Old renderer also failed: {e2}")
                raise RuntimeError("OpenGL rendering not available")

    def _generate_dummy_image(self):
        """Generate a dummy image based on current state for headless mode"""
        # Create a more meaningful dummy image that represents the scene
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        # Set background color (dark gray)
        image[:, :, :] = [40, 40, 40]
        
        if self.model.nq > 0:
            # Get joint positions
            qpos = self.data.qpos.copy()
            
            # Create a simple visual representation of the robot state
            center_x, center_y = self.image_width // 2, self.image_height // 2
            
            # Draw a circle representing the robot's position
            radius = min(20, min(self.image_width, self.image_height) // 8)
            
            # Color based on joint positions (normalize to 0-255)
            if len(qpos) >= 3:
                r = int(((qpos[0] + 1.0) / 2.0) * 255)
                g = int(((qpos[1] + 1.0) / 2.0) * 255)
                b = int(((qpos[2] + 1.0) / 2.0) * 255)
                r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
            else:
                r, g, b = 255, 128, 0  # Default orange
            
            # Draw the circle
            y_coords, x_coords = np.ogrid[:self.image_height, :self.image_width]
            mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
            image[mask] = [r, g, b]
            
            # Draw target position as a smaller circle
            target_radius = radius // 2
            target_x = center_x + int((self.target_position[0] * 50))  # Scale target position
            target_y = center_y + int((self.target_position[1] * 50))
            target_x = max(target_radius, min(self.image_width - target_radius, target_x))
            target_y = max(target_radius, min(self.image_height - target_radius, target_y))
            
            target_mask = (x_coords - target_x)**2 + (y_coords - target_y)**2 <= target_radius**2
            image[target_mask] = [0, 255, 0]  # Green target
        
        return image

    def _generate_camera_image(self):
        """Generate camera image using MuJoCo rendering or fallback to dummy image"""
        if not self.enable_rendering or not self.rendering_available:
            print("⚠ Rendering disabled or not available, using dummy image")
            return self._generate_dummy_image()
        
        # Try to find the scene_camera first, then robot_camera
        camera_id = 0
        try:
            # Look for the scene_camera first (better view)
            camera_id = self.model.camera('scene_camera').id
            print(f"✓ Found scene_camera with ID: {camera_id}")
        except Exception as e:
            print(f"⚠ scene_camera not found: {e}")
            try:
                # Look for the robot_camera by name
                camera_id = self.model.camera('robot_camera').id
                print(f"✓ Found robot_camera with ID: {camera_id}")
            except Exception as e2:
                print(f"⚠ robot_camera not found: {e2}")
                # Fallback to first camera if no named cameras found
                if self.model.ncam > 0:
                    camera_id = 0
                    print(f"⚠ Using first camera (ID: {camera_id})")
                else:
                    print("⚠ No cameras found in model")
                    return self._generate_dummy_image()
        
        # Use new renderer API only (avoid old API that causes segfaults on macOS)
        try:
            renderer = mj.Renderer(self.model, height=self.image_height, width=self.image_width)
            
            # Update the renderer with current state
            renderer.update_scene(self.data, camera=camera_id)
            
            # Render the image
            image = renderer.render()
            
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            
            # Validate the image (be less strict)
            if image is not None and image.size > 0 and image.max() > 0:
                print(f"✓ Successfully rendered camera image with shape: {image.shape}, std: {np.std(image):.1f}")
                return image
            else:
                print("⚠ Rendered image is invalid, using dummy image")
                return self._generate_dummy_image()
            
        except Exception as e:
            # If rendering fails, use dummy image
            print(f"Rendering failed: {e}, using dummy image")
            return self._generate_dummy_image()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.data = mj.MjData(self.model)  # Reset data
        mj.mj_forward(self.model, self.data)
        
        # Reset episode tracking
        self.episode_step = 0
        
        # Randomize initial block position for variety
        if self.model.nq > 0:
            # Add some randomness to initial joint positions
            self.data.qpos[:] += np.random.uniform(-0.1, 0.1, self.model.nq)
            mj.mj_forward(self.model, self.data)
        
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        
        # Apply action
        if self.model.nu > 0:
            mj.mj_set_control(self.model, self.data, action)
        else:
            # For position control, apply action directly to joint positions
            self.data.qpos[:len(action)] = action
        
        # Step simulation
        mj.mj_step(self.model, self.data)
        
        # Get observation
        obs = self._get_obs()
        
        # Calculate reward and done conditions
        reward = self._calculate_reward()
        done = self._is_done()
        
        # Update episode step
        self.episode_step += 1
        
        info = {
            'episode_step': self.episode_step,
            'success': done and self.episode_step < self.max_episode_steps
        }
        
        return obs, reward, done, False, info

    def _get_obs(self):
        # Get proprioceptive state
        proprio = self.data.qpos.copy()
        
        # Generate camera image using MuJoCo rendering
        image = self._generate_camera_image()
        
        return {
            'proprio': proprio,
            'image': image
        }

    def _calculate_reward(self):
        """Calculate reward based on current state."""
        if self.model.nq < 3:
            return 0.0
        
        # Get current end-effector position (approximated from joint positions)
        current_pos = self.data.qpos[:3]
        
        # Calculate distance to target
        distance = np.linalg.norm(current_pos - self.target_position)
        
        # Reward is negative distance (closer is better)
        reward = -distance
        
        # Bonus for being very close to target
        if distance < self.success_threshold:
            reward += 10.0
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.01
        
        return reward

    def _is_done(self):
        """Determine if episode is done."""
        # Episode ends if:
        # 1. Maximum steps reached
        # 2. Success condition met
        # 3. Robot goes out of bounds
        
        if self.episode_step >= self.max_episode_steps:
            return True
        
        if self.model.nq >= 3:
            current_pos = self.data.qpos[:3]
            distance = np.linalg.norm(current_pos - self.target_position)
            
            # Success condition
            if distance < self.success_threshold:
                return True
            
            # Out of bounds condition
            if np.any(np.abs(current_pos) > 2.0):
                return True
        
        return False

    def render(self, mode='rgb_array'):
        """Return the camera image."""
        if mode == 'rgb_array':
            return self._generate_camera_image()
        return None

    def close(self):
        """Clean up resources"""
        self._cleanup_xvfb()

    def get_state_info(self):
        """Get additional state information for debugging"""
        return {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'episode_step': self.episode_step,
            'target_position': self.target_position.copy(),
            'enable_rendering': self.enable_rendering,
            'rendering_available': self.rendering_available,
            'camera_id': self.camera_id,
            'xvfb_display': f":{self.display_num}" if self.display_num else None,
        } 