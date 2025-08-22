# MuJoCo Simulation with Manipulator, Camera, and Blocks

This directory contains a MuJoCo simulation featuring a 6-DOF manipulator robot arm, a camera, and manipulable blocks.

## Features

- **6-DOF Manipulator**: 6-joint robotic arm with gripper
- **Camera**: Positioned to view the manipulation workspace
- **Blocks**: Three different sized blocks for manipulation tasks
- **Interactive Control**: Real-time simulation with keyboard controls
- **Camera Feed**: Real-time camera view with image capture capability

## File Structure

```
mujoco_simulation/
├── manipulator_scene.xml    # MuJoCo scene definition
├── run_simulation.py        # Main simulation script
├── test_simulation.py       # Test script to verify setup
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python test_simulation.py
   ```

## Usage

### Running the Simulation

```bash
python run_simulation.py
```

The simulation will:
1. Run a demonstration sequence showing the manipulator moving to different blocks
2. Enter interactive mode where you can control the simulation

### Interactive Controls

- **`q`**: Quit simulation
- **`r`**: Reset simulation to initial pose
- **`c`**: Capture and save current camera image

### Simulation Components

#### Manipulator Robot
- **6 joints**: Base rotation, shoulder, elbow, wrist rotation/pitch/roll
- **Gripper**: Parallel jaw gripper for object manipulation
- **Actuators**: Motors for each joint and gripper

#### Camera
- **Position**: Mounted at (2.5, 0, 2.5) with 45° field of view
- **Resolution**: 640x480 pixels
- **View**: Overhead view of the manipulation workspace

#### Blocks
- **Block 1**: 10cm cube at (0.8, 0, 0.1)
- **Block 2**: 8cm cube at (1.0, 0.2, 0.1)
- **Block 3**: 6cm cube at (0.9, -0.15, 0.1)

## Customization

### Modifying the Scene

Edit `manipulator_scene.xml` to:
- Change block positions and sizes
- Adjust camera position and parameters
- Modify robot joint limits and actuator properties
- Add new objects or sensors

### Adding New Functionality

Modify `run_simulation.py` to:
- Implement more sophisticated control algorithms
- Add new interaction modes
- Integrate with external controllers
- Implement machine learning tasks

## Technical Details

### Robot Specifications
- **Degrees of Freedom**: 6 (3 for position, 3 for orientation)
- **Joint Types**: Hinge joints for rotation, slide joint for gripper
- **Actuator Types**: Motors with configurable gear ratios
- **Control**: Position and velocity control available

### Physics Engine
- **Engine**: MuJoCo 2.3+
- **Timestep**: Configurable (default: 0.01s)
- **Collision Detection**: Automatic between all bodies
- **Gravity**: Standard Earth gravity (9.81 m/s²)

### Camera System
- **Type**: Perspective camera
- **Rendering**: OpenGL-based rendering
- **Image Format**: RGB (640x480)
- **Frame Rate**: Real-time (limited by simulation timestep)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Display Issues**: MuJoCo viewer requires OpenGL support
   - On macOS: Should work out of the box
   - On Linux: Install OpenGL drivers
   - On Windows: Install OpenGL drivers

3. **Performance Issues**: 
   - Reduce camera resolution in XML
   - Increase simulation timestep
   - Close other applications

### Getting Help

- Check MuJoCo documentation: https://mujoco.readthedocs.io/
- Verify package versions match requirements
- Test with `test_simulation.py` first

## Examples

### Basic Movement
```python
from run_simulation import ManipulatorSimulation

sim = ManipulatorSimulation("manipulator_scene.xml")
sim.move_to_target([0.8, 0, 0.3])  # Move to block 1
sim.close_gripper()                  # Close gripper
```

### Getting Sensor Data
```python
# Get joint positions
joint_pos = sim.get_joint_positions()
print(f"Joint 1 position: {joint_pos['joint1']}")

# Get camera image
image = sim.get_camera_image()
cv2.imshow("Camera", image)
```

## Future Enhancements

- [ ] Inverse kinematics solver
- [ ] Trajectory planning
- [ ] Force feedback control
- [ ] Multiple camera views
- [ ] Object recognition
- [ ] Task automation
- [ ] Integration with ROS
- [ ] Machine learning interfaces
