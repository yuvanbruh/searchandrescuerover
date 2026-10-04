# Search and Rescue Rover: hand-designed vs. learned exploration

This repository contains the simulation and ROS 2 code used to study high-level exploration for an autonomous search-and-rescue rover. The main experiment compares a hand-designed frontier exploration policy ("Mission 2") with a PPO policy, both choosing from the same candidate goals and both executed by the same Nav2 stack.

**Status: work in progress.** The simulation pipeline and the Gazebo deployment are working. The PPO policy trained so far does not clearly outperform the hand-designed policy in the 2D simulator and has not yet been evaluated in Gazebo. See [Results](#results) and [Limitations](#limitations-and-current-work).

<!-- ADD: main demo video (Gazebo + RViz, rover exploring and inspecting people) -->
<!-- ADD: Gazebo / RViz screenshots -->
<!-- ADD: YOLO detection image -->

## Demo

*Media to be added.*

## Research question

Can a learned high-level policy (PPO) choose exploration goals better than a hand-designed frontier-utility rule, when both are given the same feasible candidate goals and the same low-level navigation?

"Better" here means finding and inspecting more of the people in the environment, sooner, with less driving and fewer stuck events. Only the goal-selection step differs between the two policies.

## System overview

```
 LiDAR + RGB-D + IMU
        |
   SLAM Toolbox  +  EKF  ->  occupancy map, robot pose
        |
   YOLO detector -> semantic database (person entries with map coordinates)
        |
   Candidate generator (shared)
        |   8 frontier candidates + up to 8 target candidates
        |
   +----+---------------------+
   |                          |
 Mission 2 rule             PPO policy
 (hand-designed)            (learned)
   |                          |
   +----------+---------------+
              |
        goal pose -> Nav2 (MPPI controller) -> rover
```

Everything below the goal selection (Nav2, stuck watchdog, LiDAR-directed recovery, perception, target inspection, result logging) is shared and unchanged between the two policies.

## Robot and software stack

| Component | Details |
|---|---|
| Platform | 4-wheel differential-drive rover, URDF/xacro in `src/minibot` |
| OS / middleware | Ubuntu 24.04, ROS 2 Jazzy |
| Simulator | Gazebo (the default pairing for Jazzy), `ros_gz` bridge |
| Drive | `ros2_control` `diff_drive_controller`, `twist_mux` |
| Sensors | 2D LiDAR (360 samples, 10 m), RGB-D camera on a mast, IMU, two mast-mounted spot lights |
| State estimation | `robot_localization` EKF (wheel odometry + IMU) |
| Mapping | SLAM Toolbox, online asynchronous mapping, 0.05 m resolution |
| Navigation | Nav2 with the MPPI controller |
| Perception | Ultralytics YOLO (OpenVINO export) with tracking and depth-based 3D localisation, `semantic_perception` package |
| Learning | PyTorch (training), NumPy (inference inside ROS) |

## Exploration methods

### Mission 2 (hand-designed baseline)

Implemented in `src/frontier_explorer/frontier_explorer/mission2.py`.

1. If there are known, uninspected, non-abandoned `person` targets above a low confidence floor (0.10), go to the one with the shortest Nav2 path and inspect it.
2. Otherwise explore: score each frontier candidate with

   `U = alpha * IG_norm - beta_eff * Cost_norm`

   with `alpha = 3.0`, `beta = 0.25` (scaled up as the battery model drains), `IG_norm` the visible-unknown-cell count from 180 LiDAR-like rays (10 m range) divided by the ray count, and `Cost_norm` the Nav2 path length divided by the longest candidate path. Go to the highest-scoring frontier.
3. A target that cannot be reached or confirmed after 2 attempts is abandoned. A frontier that causes a stuck event is blacklisted for 30 s. The mission ends after 3 consecutive cycles with no frontier or target left.

### Shared candidate generation

Both policies receive the same candidate set, built as in Mission 2 (with the target list capped at 8):

- Frontier cells are clustered; visited and blacklisted clusters are dropped.
- The 15 largest clusters get a safe goal cell (obstacle margin) and an information-gain estimate.
- The 8 with the highest information gain become the frontier candidates.
- Every available target gets a stand-off point near it; the 8 with the shortest Nav2 path become the target candidates.

This gives a 16-slot action space (8 frontier + 8 target, masked when empty).

### PPO policy

Implemented in `sar_training/sar_rl/policy.py` and trained with `sar_training/train_ppo.py`.

- A shared encoder scores every candidate; two self-attention layers let the candidates see each other; a masked softmax picks one.
- Input per candidate (13 numbers): target flag, Nav2 path length (absolute and relative), visible information gain, cluster size, offset to the robot (x, y), target confidence, failed attempts, clearance at the goal, minimum clearance along the path, target offset from free space, distance to previously chosen goals.
- Global input (10 numbers): known free area, number of frontier candidates, battery, elapsed time, available targets, inspected / abandoned counts, stuck events, decisions so far, empty cycles.
- The reward is task-based: positive for each person inspected (more if early) and for map coverage gained; penalties for time, distance, stuck events and abandoned targets. No Mission 2 utility formula is used anywhere in training.
- The checkpoint evaluated below (`run16_baseline`) was trained from scratch for 800,000 decisions with the default reward weights.

For deployment the policy is exported to a `.npz` file and run with NumPy inside the ROS node (`sar_training/gazebo_deploy/`), so PyTorch is not needed on the robot.

## 2D simulation environment

Training in Gazebo would take months (one mission takes tens of minutes), so the policy is trained in a fast 2D simulator, `sar_training/sar_rl/`. One simulator step is one goal decision.

- **Copied from the Mission 2 node:** frontier detection and clustering, goal-cell safety filter, information-gain ray casting, visited/blacklist logic, target availability rules, stand-off point, completion rule.
- **Approximated:** Nav2 (Dijkstra on the known map), stuck events (probability grows with path/goal narrowness), YOLO (range, field of view, line of sight, localisation noise, duplicate and false detections, snapshot failures), driving speed and timing. These are randomised each episode.
- **Maps:** an exact replica of the Gazebo building (`sar_building_v2`) plus procedurally generated corridor-and-room layouts. Training uses the generated layouts; the replica is kept for testing.
- **Calibration:** `CALIBRATED` in `sar_env.py` tunes the physics to one real Gazebo Mission 2 run (about 2000 s, about 13 stuck events, 15 database entries for 6 people). Note that the results below were produced with the default (uncalibrated, easier) simulator configuration, not `CALIBRATED`.

<p align="center">
  <img src="sar_training/sim_v2_mission2_like.png" width="48%">
  <img src="sar_training/sim_random_mission2_like.png" width="48%">
</p>
<p align="center"><em>Simulator: the Gazebo building replica (left) and a generated layout (right).</em></p>

**Metrics** (same names as the ROS result files where possible): mission time, distance traveled, final coverage, people inspected (ground truth), stuck events, abandoned database entries, completion.

## Results

### Simulator: Mission 2 rule vs. PPO (`run16_baseline`)

50 paired episodes per setting (same seeds for both policies), default (uncalibrated) simulator configuration. Values are mean ± standard deviation. People inspected is out of 6 in the building layout (6.18 on average in the generated layouts).

**Building replica, nominal physics**

| | Mission 2 rule | PPO |
|---|---|---|
| Mission time (s) | 613 ± 116 | 730 ± 223 |
| Distance (m) | 104 ± 20 | 135 ± 39 |
| Coverage (%) | 99.99 | 99.96 |
| People inspected | 4.98 ± 0.8 | 5.44 ± 0.8 |
| Stuck events | 4.44 ± 1.9 | 4.52 ± 2.7 |
| Abandoned entries | 2.14 ± 1.3 | 2.58 ± 1.9 |

**Building replica, randomised physics**

| | Mission 2 rule | PPO |
|---|---|---|
| Mission time (s) | 677 ± 211 | 759 ± 243 |
| Distance (m) | 117 ± 29 | 138 ± 35 |
| Coverage (%) | 99.81 | 100.00 |
| People inspected | 4.90 ± 1.0 | 5.22 ± 1.0 |
| Stuck events | 4.34 ± 3.1 | 4.60 ± 3.2 |
| Abandoned entries | 3.24 ± 2.7 | 3.72 ± 2.5 |

**Generated layouts, randomised physics**

| | Mission 2 rule | PPO |
|---|---|---|
| Mission time (s) | 862 ± 314 | 1076 ± 389 |
| Distance (m) | 168 ± 59 | 216 ± 79 |
| Coverage (%) | 97.44 | 97.57 |
| People inspected | 5.20 ± 2.4 | 5.44 ± 2.3 |
| Stuck events | 4.54 ± 2.4 | 6.14 ± 3.3 |
| Abandoned entries | 4.14 ± 2.8 | 4.96 ± 3.0 |

In the simulator this PPO policy inspects slightly more people than the Mission 2 rule, but takes longer and drives farther, and it has the same or more stuck events. It does not outperform the hand-designed policy overall. The training reward at that stage charged very little for time and distance, which is consistent with this behaviour. A second run with a stronger time/distance/stuck penalty on the calibrated simulator is planned.

### Gazebo: Mission 2 reference run

One run of the original Mission 2 node in Gazebo (single run, so treat it as an example, not an average):

| Metric | Value |
|---|---|
| Runtime | 1999.8 s (33.3 min) |
| Distance traveled (sum of planned path lengths of reached goals) | 93.5 m |
| Final map coverage | 97.35 % |
| Target entries detected / inspected / abandoned | 15 / 11 / 4 |
| First detection / first inspection | 24.8 s / 51.8 s |
| Stuck events (watchdog) | 13 |
| Average decision latency | 186 ms |

"Targets" are semantic-database entries, which can include duplicate and false detections. The Gazebo worlds contain 6 people.

### Gazebo: PPO vs. Mission 2

*Not yet available.* Matched runs (same world, same start) of `mission2`, `mission2c` and `rl` are planned; see below.

## Gazebo experiments

`sar_training/gazebo_deploy/explorer_rl_node.py` inherits from the Mission 2 node and adds a switch:

| `policy_mode` | What runs |
|---|---|
| `mission2` | The original Mission 2 node, unchanged (reference) |
| `mission2c` | The Mission 2 rule applied to the shared candidate set (fair baseline) |
| `rl` | The PPO policy choosing from the same candidate set |

Each run writes to `~/ares_results/run_NNN/`: `experiment_summary.csv`, `frontier_decisions.csv`, `target_milestones.csv`, and for `mission2c`/`rl` also `rl_decisions.csv`, `rl_candidates.csv` (every candidate, its features and the chosen one) and `rl_globals.csv`.

*Representative runs and plots: to be added.*

## Environments

Worlds are in `src/minibot/worlds/`. The two main worlds share the same building: a west entrance, an east-west corridor and six rooms, with doorways of different widths (including one narrow doorway), loop doors between rooms, clutter, and six person models.

| World | Description |
|---|---|
| `baselinebest.sdf` (same as `baseline4.sdf`) | Clean version of the building |
| `gazebo1.sdf` (same as `underground.sdf`) | Same building with floor zones, puddles, debris, grime, fog layers and smoke (low visibility) |
| `baseline1.sdf` to `baseline3.sdf` | Earlier worlds (empty, generated maze, first building version) |

## Repository structure

```
src/
  minibot/             Robot description, Gazebo worlds, launch files, Nav2/SLAM/EKF configs
  frontier_explorer/   Mission 2 explorer node (mission2.py) and earlier versions
  semantic_perception/ YOLO detector node and semantic database
  semantic_interfaces/ Custom messages and services for targets
sar_training/
  sar_rl/              2D simulator, layouts, PPO policy network, vectorised env
  train_ppo.py         PPO training
  eval_sim.py          Compare a trained policy with the Mission 2 rule in the simulator
  check_sim.py         Simulator sanity check
  gazebo_deploy/       Export script, NumPy policy, RL explorer node, launch file
  policy16.npz         Exported PPO policy (run16_baseline)
yolo_ros/              Third-party ROS 2 YOLO wrapper (see Credits)
behavior_trees/        Nav2 behavior tree
yolo26n_openvino_model/ , *.pt   YOLO weights
```

## Running the system

### Requirements

Ubuntu 24.04 and ROS 2 Jazzy with Gazebo, `ros_gz`, `ros2_control`, Nav2, `slam_toolbox`, `robot_localization`, `twist_mux` and `twist_stamper`. The minibot installation notes in `src/minibot/Package_Installation_Instruction.md` cover most of these. YOLO needs `ultralytics` (and `openvino` for the exported model).

### Build

```bash
git clone https://github.com/<your-username>/searchandrescuerover.git
cd searchandrescuerover
colcon build --symlink-install
source install/setup.bash
```

### Mission 2 in Gazebo

Each command in its own terminal, with the workspace sourced:

```bash
ros2 launch minibot cam.py        # Gazebo world (gazebo1.sdf), robot, bridge, RViz
ros2 launch minibot ek.py         # EKF, SLAM Toolbox, Nav2
ros2 launch minibot semantic.py   # Mission 2 explorer, YOLO detector, semantic database
```

### PPO (or the fair baseline) in Gazebo

Start the first two launch files as above, then, from the repository root, instead of `semantic.py`:

```bash
ros2 launch sar_training/gazebo_deploy/rl_explorer_launch.py \
  base_module:=frontier_explorer.mission2 \
  policy_mode:=rl \
  model_path:=$PWD/sar_training/policy16.npz \
  script_path:=$PWD/sar_training/gazebo_deploy/explorer_rl_node.py \
  experiment_name:=rl16_01
```

Use `policy_mode:=mission2c` (and drop `model_path`) for the Mission 2 rule on the shared candidates. This launch file also starts the YOLO detector and the semantic database, so do not run `semantic.py` at the same time. Run it from a terminal that does not have a Python virtual environment active.

### Training and evaluating in the simulator

No ROS is needed:

```bash
cd sar_training
pip install numpy scipy matplotlib torch gymnasium

python check_sim.py --episodes 5 --png                 # simulator sanity check
python train_ppo.py --name run1 --total-steps 800000   # PPO training, writes runs/run1/
python eval_sim.py --ckpt runs/run1/final.pt --episodes 50
```

To use a trained policy in Gazebo, export it first:

```bash
python gazebo_deploy/export_policy.py --ckpt runs/run1/final.pt --out policy.npz
```

The export script also checks that the NumPy version reproduces the PyTorch policy.

## Limitations and current work

- The PPO policy has not been evaluated in Gazebo yet, and in the simulator it is about on par with, not better than, the hand-designed policy.
- `run16_baseline` was trained on the default simulator configuration with the default reward, which barely penalises time and distance. A second run on the calibrated simulator with stronger penalties is planned.
- The simulator replaces Nav2 with a shortest-path planner and models stuck events, YOLO errors and timing statistically. It is calibrated to a single real Gazebo run, so the sim-to-Gazebo gap is not yet measured.
- The Gazebo reference numbers come from one run; repeated runs are needed for a comparison.
- Several launch files and configs still contain hard-coded paths and older versions kept as comments (see the notes below).
- Hardware deployment is planned but has not been done; everything here is simulation.

### Known issues to fix

- `sim.launch.py` and `camd.py` load `playground.sdf`, which is not in `worlds/`. The experiments use `cam.py`.
- The YOLO detector's default `model_path` is an absolute path on the author's machine; set it to `yolo26n_openvino_model/` in this repository.
- `rl_explorer_launch.py` defaults to paths under `~/last/`; pass `model_path` and `script_path` explicitly as shown above.
- `setup.py` in `frontier_explorer` and `semantic_perception` lists entry points for modules that do not exist (`logs`, `best`).
- `mission2withc.py` is an earlier version of `mission2.py` (inspection radius 0.8 m, obstacle margin 3 cells, fixed goal heading).

## Research status

This is an ongoing student research project. Planned next steps: train the second PPO run on the calibrated simulator, run matched Gazebo experiments (`mission2`, `mission2c`, `rl`, several runs each) in the clean and the low-visibility world, then report coverage, distance, time, inspections, abandonments and stuck events with their spread.

## Credits and licenses

- The robot description, launch structure and Nav2/SLAM configuration in `src/minibot` are adapted from [YJ0528/minibot](https://github.com/YJ0528/minibot) (MIT; earlier history Apache-2.0, see `src/minibot/LICENSE` and `LICENSE.previous.md`), which follows the [Articulated Robotics](https://articulatedrobotics.xyz/) tutorials. The images and GIFs in `src/minibot/visual_demos/` come from that project, not from this one.
- `yolo_ros/` is [mgonzs13/yolo_ros](https://github.com/mgonzs13/yolo_ros), licensed GPL-3.0.
- Object detection uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics); check its license terms before redistributing weights.
- `twist_stamper` is by Josh Newans.
- The license for the original code in this repository (`frontier_explorer`, `semantic_perception`, `sar_training`) has not been chosen yet.
