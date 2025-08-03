import mujoco as mj
import numpy as np
import gym
from gym import spaces
import os

class MujocoSimpleEnv(gym.Env):
    def __init__(self, xml_path=None, image_width=128, image_height=128):
        super().__init__()
        if xml_path is None:
            dirname = os.path.dirname(__file__)
            xml_path = os.path.join(dirname, 'block_with_camera.xml')
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.data = mj.MjData(self.model)
        
        # Camera settings
        self.image_width = image_width
        self.image_height = image_height
        
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

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.data = mj.MjData(self.model)  # Reset data
        mj.mj_forward(self.model, self.data)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if self.model.nu > 0:
            mj.mj_set_control(self.model, self.data, action)
        else:
            self.data.qpos[:len(action)] = action
        mj.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = 0.0  # Placeholder
        done = False  # Placeholder
        info = {}
        return obs, reward, done, False, info

    def _get_obs(self):
        # Get proprioceptive state
        proprio = self.data.qpos.copy()
        
        # Return dummy image (no rendering to avoid OpenGL issues)
        image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        
        return {
            'proprio': proprio,
            'image': image
        }

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            return np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        return None

    def close(self):
        pass 
