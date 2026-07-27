# Quick start
This quick start allows you to collect data in the MuJoCo simulation and train and rollout the ACT policy.

## Install
Install RoboManipBaselines:
```console
$ git clone git@github.com:isri-aist/RoboManipBaselines.git --recursive
$ cd RoboManipBaselines
$ pip install -e .[act]
```

Install ACT from a third party:
```console
$ cd third_party/act/detr
$ pip install -e .
```

## Data collection by teleoperation
> [!TIP]
> Instead of collecting data by teleoperation, you can download the public dataset `TeleopMujocoUR5eCable_Dataset30` from [here](./dataset_list.md#Demonstrations-in-MuJoCo-environments).

Operate the robot in the simulation and save the data:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ # Connect a SpaceMouse to your PC
$ python ./bin/Teleop.py MujocoUR5eCable --world_idx_list 0 5 --input_device keyboard
```

### Keyboard Control Guide
When using keyboard teleoperation, the following keys control the robot end-effector:

| Movement | Key | Direction |
|----------|-----|-----------|
| **Position (Cartesian space)** | | |
| X-axis (forward/backward) | **W** / **S** | Forward / Backward |
| Y-axis (left/right) | **A** / **D** | Left / Right |
| Z-axis (up/down) | **Q** / **E** | Up / Down |
| **Rotation (Euler angles: Roll/Pitch/Yaw)** | | |
| Roll | **J** / **L** | Counter-clockwise / Clockwise |
| Pitch | **I** / **K** | Up / Down |
| Yaw | **U** / **O** | Counter-clockwise / Clockwise |
| **Gripper** | | |
| Gripper control | **Z** / **X** | Close / Open |

**Note:** Hold keys for continuous motion. You can combine multiple keys (e.g., press W+I to move forward and pitch up simultaneously).

**Keyboard Layout Diagram:**
```
┌─────────────────────────────────────────────┐
│ POSITION CONTROL (Left side of keyboard)   │
│                                             │
│   Q (Z+)                                   │
│   W (X+)   I (Pitch+)                      │
│ A S D    J L  U O  Z X                     │
│ Y- X-   Roll- Roll+  Yaw  Gripper          │
│   E (Z-)   K (Pitch-)                      │
│                                             │
│ W, A, S, D = X, Y position movement       │
│ Q, E = Z position (height)                │
│ I, K = Pitch (forward/backward tilt)      │
│ J, L = Roll (left/right tilt)             │
│ U, O = Yaw (rotation)                     │
│ Z = Close gripper                         │
│ X = Open gripper                          │
└─────────────────────────────────────────────┘
```

### Test command transmission (dry-run mode)
Before running on the real robot, you can test the entire teleoperation pipeline without connecting to hardware. This validates keyboard input → inverse kinematics → command generation:
```console
$ python ./bin/Teleop.py RealFairino3Demo --config ./envs/configs/RealFairino3DemoEnv_DryRun.yaml --world_idx_list 0 5 --input_device keyboard
```
In dry-run mode, ServoJ commands will be printed to the console instead of transmitted to the robot. This allows you to:
- Verify the keyboard input device works
- Check that inverse kinematics computes successfully
- See the generated joint commands before hardware execution
- **Test without requiring the LinkerHand gripper module** (useful for development/testing)

> [!NOTE]
> **LinkerHand Gripper Dependency:** The dry-run mode works without the LinkerHand module. Hardware mode requires LinkerHand for gripper control.
> 
> **Installation:** To install LinkerHand dependencies:
> ```console
> $ pip install -r ./third_party/linkerhand-python-sdk/requirements.txt
> ```
> After installation, the LinkerHand module is automatically available for hardware control.

### Run on the real Fairino FR3 arm
To teleoperate the real Fairino FR3 arm instead, edit `robot_ip` (and, if available, `camera_ids` / `gelsight_ids`) in [RealFairino3DemoEnv.yaml](../robo_manip_baselines/envs/configs/RealFairino3DemoEnv.yaml), then run:
```console
$ python ./bin/Teleop.py RealFairino3Demo --config ./envs/configs/RealFairino3DemoEnv.yaml --world_idx_list 0 5 --input_device keyboard
$ # A Vive tracker can be used instead of a keyboard (requires --input_device_config, e.g. ./teleop/configs/Vive.yaml)
$ python ./bin/Teleop.py RealFairino3Demo --config ./envs/configs/RealFairino3DemoEnv.yaml --world_idx_list 0 5 --input_device vive --input_device_config ./teleop/configs/Vive.yaml
```

> [!TIP]
> A teleoperation input device such as a 3D mouse can be used instead of a keyboard. See [here](../robo_manip_baselines/teleop/README.md).

### Run on the real Fairino FR5 arm
The FR5 arm shares the same joint ranges and control interface as FR3, just a different kinematic chain. Test command transmission first:
```console
$ python ./bin/Teleop.py RealFairino5Demo --config ./envs/configs/RealFairino5DemoEnv_DryRun.yaml --world_idx_list 0 5 --input_device keyboard
```
To teleoperate the real Fairino FR5 arm, edit `robot_ip` (and, if available, `camera_ids` / `gelsight_ids`) in [RealFairino5DemoEnv.yaml](../robo_manip_baselines/envs/configs/RealFairino5DemoEnv.yaml), then run:
```console
$ python ./bin/Teleop.py RealFairino5Demo --config ./envs/configs/RealFairino5DemoEnv.yaml --world_idx_list 0 5 --input_device keyboard
$ # A Vive tracker can be used instead of a keyboard (requires --input_device_config, e.g. ./teleop/configs/Vive.yaml)
$ python ./bin/Teleop.py RealFairino5Demo --config ./envs/configs/RealFairino5DemoEnv.yaml --world_idx_list 0 5 --input_device vive --input_device_config ./teleop/configs/Vive.yaml
```

### Run on the dual-arm Fairino FR3 setup
For the two-arm rig (two FR3 arms, two LinkerHand grippers, one head camera, two wrist cameras), edit `robot_ip_left` / `robot_ip_right` / `gripper_modbus_port_left` / `gripper_modbus_port_right` and the Orbbec camera serial numbers in [RealFairinoDualDemoEnv.yaml](../robo_manip_baselines/envs/configs/RealFairinoDualDemoEnv.yaml), then run:
```console
$ python ./bin/Teleop.py RealFairinoDualDemo --config ./envs/configs/RealFairinoDualDemoEnv.yaml --world_idx_list 0 5 --input_device keyboard
```
Both arms are driven from a single physical keyboard with non-overlapping key sets:

| Arm | XY | Z | Roll/Pitch | Yaw | Gripper |
|---|---|---|---|---|---|
| Left | WASD | Q/E | J/L, I/K | U/O | Z/X |
| Right | 4/6/8/2 (numpad-style) | 9/3 | F/H, T/G | Y/B | N/M |

A dry-run config is also available for testing the pipeline without hardware:
```console
$ python ./bin/Teleop.py RealFairinoDualDemo --config ./envs/configs/RealFairinoDualDemoEnv_DryRun.yaml --world_idx_list 0 5 --input_device keyboard
```

In our experience, models can be trained stably with roughly 30 data sets.
The teleoperation data is saved in the `robo_manip_baselines/dataset/MujocoUR5eCable_<date_suffix>` directory (e.g., `MujocoUR5eCable_20240101_120000`).

## Model training
Train the ACT:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py Act --dataset_dir ./dataset/MujocoUR5eCable_20240101_120000
```
The learned parameters are saved in the `robo_manip_baselines/checkpoint/Act/<dataset_name>_Act_<date_suffix>` directory (e.g., `MujocoUR5eCable_20240101_120000_Act_20240101_130000`).

> [!NOTE]
> The following error will occur if the chunk_size is larger than the time series length of the training data.
> In such a case, either set the `--skip` option to a small value, or set the `--chunk_size` option to a small value.
> ```console
> RuntimeError: The size of tensor a (70) must match the size of tensor b (102) at non-singleton dimension 0
> ```

## Policy rollout
Rollout the ACT in the simulation:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Rollout.py Act MujocoUR5eCable \
--checkpoint ./checkpoint/Act/MujocoUR5eCable_20240101_120000_Act_20240101_130000/policy_last.ckpt \
--world_idx 0
```
