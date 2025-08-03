# OpenVLA + MuJoCo Integration Guide

## Overview

The integration between OpenVLA (Vision-Language-Action model) and MuJoCo is designed to enable robotic control using natural language instructions and visual observations. Here's how the integration works:

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MuJoCo Env    │───▶│   OpenVLA Model │───▶│   Action Exec   │
│                 │    │                 │    │                 │
│ • Visual Obs    │    │ • Vision Encoder│    │ • 7D Actions    │
│ • Proprio State │    │ • LLM Decoder   │    │ • Gripper Ctrl  │
│ • Physics Sim   │    │ • Action Head   │    │ • Joint Control │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Components

### 1. MuJoCo Environment (`mujoco_gym_env.py`)

**Purpose**: Provides a Gym-compatible interface to MuJoCo simulation with camera rendering.

**Key Features**:
- **Visual Observations**: RGB images from robot camera
- **Proprioceptive State**: Joint positions (qpos)
- **Action Space**: 7-dimensional actions (6 DOF + gripper)
- **Physics Simulation**: Realistic dynamics and collision detection

```python
class MujocoSimpleEnv(gym.Env):
    def __init__(self, xml_path=None, image_width=128, image_height=128):
        # Load MuJoCo model and setup camera
        self.model = mj.MjModel.from_xml_path(xml_path)
        self.camera_id = self.model.camera('robot_camera').id
        
    def _get_obs(self):
        # Render RGB image from camera
        rgb = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        # ... rendering code ...
        return {
            'image': rgb.copy(),
            'proprio': self.data.qpos.copy(),
        }
```

### 2. OpenVLA Model Integration (`openvla_utils.py`)

**Purpose**: Loads and manages the OpenVLA vision-language-action model.

**Key Functions**:
- **Model Loading**: Loads pre-trained OpenVLA checkpoints
- **Action Prediction**: Generates 7D actions from observations
- **Image Processing**: Handles image preprocessing and augmentation

```python
def get_vla(cfg):
    """Loads OpenVLA model from checkpoint"""
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        # ... other params ...
    )
    return vla

def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key):
    """Generates action from observation and task description"""
    image = Image.fromarray(obs["full_image"])
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
    inputs = processor(prompt, image)
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key)
    return action
```

### 3. Robot Utilities (`robot_utils.py`)

**Purpose**: Provides helper functions for action processing and environment management.

**Key Functions**:
- **Action Normalization**: Converts between different action formats
- **Gripper Control**: Handles gripper action conventions
- **Model Management**: Loads and configures models

```python
def get_action(cfg, model, obs, task_label, processor=None):
    """Main interface for getting actions from model"""
    if cfg.model_family == "openvla":
        action = get_vla_action(model, processor, cfg.pretrained_checkpoint, 
                              obs, task_label, cfg.unnorm_key)
    return action

def invert_gripper_action(action):
    """Handles gripper action sign conventions"""
    action[..., -1] = action[..., -1] * -1.0
    return action
```

## Integration Flow

### 1. Environment Setup
```python
# Create MuJoCo environment
env = MujocoSimpleEnv(image_width=128, image_height=128)

# Load OpenVLA model
model = get_model(cfg)
processor = get_processor(cfg)
```

### 2. Observation Processing
```python
# Get observation from MuJoCo
obs, _ = env.reset()

# Format for OpenVLA
observation = {
    "full_image": obs["image"],      # RGB image from camera
    "state": obs["proprio"],         # Joint positions
}
```

### 3. Action Generation
```python
# Generate action using OpenVLA
action = get_action(cfg, model, observation, task_label="move block", processor=processor)

# Apply gripper action convention
action = invert_gripper_action(action)
```

### 4. Environment Step
```python
# Execute action in MuJoCo
obs, reward, done, _, info = env.step(action)
```

## Data Flow

### Input to OpenVLA:
1. **Visual Observation**: RGB image (128x128x3) from robot camera
2. **Proprioceptive State**: Joint positions (7D vector)
3. **Task Description**: Natural language instruction (e.g., "move block")

### Output from OpenVLA:
1. **7D Action Vector**: [dx, dy, dz, drx, dry, drz, gripper]
   - First 6 dimensions: End-effector velocity in world frame
   - Last dimension: Gripper control (-1 = open, +1 = close)

## Action Conventions

### Gripper Actions:
- **OpenVLA Convention**: 0 = close, 1 = open
- **MuJoCo Convention**: -1 = open, +1 = close
- **Solution**: Use `invert_gripper_action()` to flip the sign

### Action Normalization:
- Actions are normalized to [-1, +1] range
- Dataset statistics are used for un-normalization
- Gripper actions may be binarized for discrete control

## File Structure

```
openvla-main/
├── experiments/robot/
│   ├── run_mujoco_simple_eval.py          # Main evaluation script
│   ├── run_mujoco_simple_eval_mock.py     # Mock evaluation (no model)
│   ├── robot_utils.py                     # Robot utilities
│   └── openvla_utils.py                   # OpenVLA utilities
└── Lec14a_camera_sensor/
    ├── mujoco_gym_env.py                  # MuJoCo environment
    ├── mujoco_gym_env_headless.py         # Headless version
    └── block_with_camera.xml              # MuJoCo model
```

## Usage Examples

### 1. Real Evaluation (with checkpoint)
```bash
python run_mujoco_simple_eval.py \
    --pretrained_checkpoint /path/to/openvla/checkpoint \
    --num_episodes 5 \
    --max_steps 100
```

### 2. Mock Evaluation (for testing)
```bash
python run_mujoco_simple_eval_mock.py \
    --num_episodes 3 \
    --max_steps 50
```

### 3. Custom Integration
```python
from mujoco_gym_env import MujocoSimpleEnv
from robot_utils import get_model, get_action, invert_gripper_action

# Setup
env = MujocoSimpleEnv()
model = get_model(cfg)
processor = get_processor(cfg)

# Run episode
obs, _ = env.reset()
for step in range(max_steps):
    observation = {"full_image": obs["image"], "state": obs["proprio"]}
    action = get_action(cfg, model, observation, "move block", processor)
    action = invert_gripper_action(action)
    obs, reward, done, _, info = env.step(action)
```

## Key Design Decisions

### 1. Modular Architecture
- **Separation of Concerns**: Environment, model, and utilities are separate
- **Easy Testing**: Mock evaluation allows testing without real model
- **Flexible**: Can easily swap different models or environments

### 2. Standardized Interfaces
- **Gym Compatibility**: Uses standard Gym environment interface
- **Consistent Observations**: Standardized observation format
- **Action Conventions**: Handles different action space conventions

### 3. Robust Error Handling
- **Headless Support**: Works without display for server deployment
- **Graceful Fallbacks**: Handles rendering failures gracefully
- **Action Validation**: Ensures actions are in correct format

## Extensions and Customization

### 1. Different Tasks
- Modify task descriptions in `get_vla_action()`
- Add task-specific reward functions
- Implement task success detection

### 2. Different Robots
- Create new MuJoCo XML models
- Adjust action space dimensions
- Modify camera configurations

### 3. Different Models
- Support other VLA models (Prismatic, RT-X, etc.)
- Add model-specific preprocessing
- Implement ensemble methods

This integration provides a complete pipeline from natural language instructions to robotic actions in a simulated environment, enabling research in vision-language robotics. 