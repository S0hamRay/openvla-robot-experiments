import os
import numpy as np
import gymnasium as gym
import mujoco as mj
from gymnasium import spaces

class MujocoSimpleEnvHeadless(gym.Env):
    def __init__(self, xml_path=None, image_width=128, image_height=128, enable_rendering=True):
        super().__init__()
        
        # Set up headless rendering backend - try multiple options
        self._setup_rendering_backend()
        
        if xml_path is None:
            dirname = os.path.dirname(__file__)
            xml_path = os.path.join(dirname, 'block_with_camera.xml')
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        
        # Camera settings for synthetic image generation
        self.image_width = image_width
        self.image_height = image_height
        self.enable_rendering = enable_rendering
        
        # Try to detect camera
        self.camera_id = 0
        self.rendering_available = False
        self.rendering_backend = None
        
        if self.enable_rendering:
            self._test_rendering_capabilities()
        
        # Action space: joint positions (qpos)
        n_act = self.model.nu if self.model.nu > 0 else self.model.nq
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_act,), dtype=np.float32
        )
        
        # Observation space: proprio + synthetic image
        self.observation_space = spaces.Dict({
            'proprio': spaces.Box(low=-np.inf, high=np.inf, shape=(self.model.nq,), dtype=np.float32),
            'image': spaces.Box(low=0, high=255, shape=(self.image_height, self.image_width, 3), dtype=np.uint8)
        })
        
        # Track episode state for more realistic simulation
        self.episode_step = 0
        self.max_episode_steps = 100
        self.success_threshold = 0.1  # Distance threshold for success
        self.target_position = np.array([0.0, 0.0, 0.5])  # Target position for the block

    def _setup_rendering_backend(self):
        """Set up software rendering backend for headless environments"""
        # Set EGL backend (avoid problematic environment variables on macOS)
        os.environ['MUJOCO_GL'] = 'egl'
        # Don't set LIBGL_ALWAYS_SOFTWARE or MESA_GL_VERSION_OVERRIDE on macOS
        # as they can cause segmentation faults
        
        print("Using MuJoCo EGL backend")
        print("  - MUJOCO_GL=egl")
        
        # Test if the rendering backend works
        try:
            # Test with a minimal model to verify rendering works
            test_xml_path = os.path.join(os.path.dirname(__file__), 'block_with_camera.xml')
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
        """Test if software rendering is available and working"""
        try:
            # Ensure the scene is properly set up
            mj.mj_forward(self.model, self.data)
            
            # Test only the new renderer API (avoid old API that causes segfaults on macOS)
            renderer = mj.Renderer(self.model, height=self.image_height, width=self.image_width)
            renderer.update_scene(self.data, camera=self.camera_id)
            image = renderer.render()
            
            # Check if the rendered image is valid
            if image is not None and image.size > 0:
                # Convert to uint8 if needed
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                
                # Check if image has meaningful content (be less strict)
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
            # If rendering fails, disable it and use dummy image
            self.rendering_available = False
            self.rendering_backend = None
            print(f"⚠ MuJoCo rendering not available: {e}")
            print("   Using dummy images instead")

    def _test_rendering(self):
        """Test if rendering is available by creating a minimal context"""
        try:
            scene = mj.MjvScene(self.model, maxgeom=1000)
            context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150)
            # If we get here, rendering works
            return True
        except Exception:
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
            
            # Add some text-like indicators
            # Draw a line from robot to target
            if len(qpos) >= 3:
                current_pos = qpos[:3]
                distance = np.linalg.norm(current_pos - self.target_position)
                
                # Draw distance indicator
                line_length = int(min(distance * 30, 50))
                if line_length > 5:
                    # Calculate direction vector
                    dx = target_x - center_x
                    dy = target_y - center_y
                    distance_pixels = np.sqrt(dx**2 + dy**2)
                    
                    # Only draw line if there's a meaningful direction
                    if distance_pixels > 1:
                        # Normalize direction vector
                        dx_norm = dx / distance_pixels
                        dy_norm = dy / distance_pixels
                        
                        # Calculate end point
                        end_x = center_x + int(line_length * dx_norm)
                        end_y = center_y + int(line_length * dy_norm)
                        
                        # Simple line drawing (Bresenham-like)
                        if abs(end_x - center_x) > abs(end_y - center_y):
                            # Horizontal line
                            if center_x <= end_x:
                                x_range = range(center_x, min(end_x + 1, self.image_width))
                            else:
                                x_range = range(max(0, end_x), center_x + 1)
                            
                            for x in x_range:
                                if end_x != center_x:
                                    y = center_y + int((x - center_x) * (end_y - center_y) / (end_x - center_x))
                                    if 0 <= y < self.image_height:
                                        image[y, x] = [255, 255, 0]  # Yellow line
                        else:
                            # Vertical line
                            if center_y <= end_y:
                                y_range = range(center_y, min(end_y + 1, self.image_height))
                            else:
                                y_range = range(max(0, end_y), center_y + 1)
                            
                            for y in y_range:
                                if end_y != center_y:
                                    x = center_x + int((y - center_y) * (end_x - center_x) / (end_y - center_y))
                                    if 0 <= x < self.image_width:
                                        image[y, x] = [255, 255, 0]  # Yellow line
        
        return image

    def _generate_camera_image(self):
        """Generate camera image using MuJoCo rendering or fallback to dummy image"""
        if not self.enable_rendering or not self.rendering_available:
            print("⚠ Rendering disabled or not available, using dummy image")
            return self._generate_dummy_image()
        
        # Try to find the robot_camera first, then scene_camera
        camera_id = 0
        try:
            # Look for the robot_camera first (closer to scene)
            camera_id = self.model.camera('robot_camera').id
            print(f"✓ Found robot_camera with ID: {camera_id}")
        except Exception as e:
            print(f"⚠ robot_camera not found: {e}")
            try:
                # Look for the scene_camera as fallback
                camera_id = self.model.camera('scene_camera').id
                print(f"✓ Found scene_camera with ID: {camera_id}")
            except Exception as e2:
                print(f"⚠ scene_camera not found: {e2}")
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
        pass

    def get_state_info(self):
        """Get additional state information for debugging"""
        return {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy(),
            'episode_step': self.episode_step,
            'target_position': self.target_position.copy(),
            'enable_rendering': self.enable_rendering,
            'rendering_available': self.rendering_available,
            'rendering_backend': self.rendering_backend,
            'camera_id': self.camera_id,
        } 