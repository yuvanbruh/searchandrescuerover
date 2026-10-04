# Search and Rescue Rover

An autonomous search-and-rescue rover for exploration, target detection, navigation and high-level goal selection.

The project combines ROS 2, SLAM, EKF state estimation, Nav2, LiDAR, RGB-D perception and YOLO-based target detection. The main research experiment studies whether a learned PPO policy can select exploration goals more effectively than a hand-designed frontier exploration policy.

**Status: ongoing research project.**

The complete navigation and perception stack is operational in Gazebo. PPO has been trained in a fast 2D simulator and deployed into the ROS 2/Gazebo system. Current simulator experiments do not show an overall advantage for PPO over the hand-designed policy, while representative Gazebo runs show differences in exploration behaviour that are being investigated with repeated matched experiments.

---

## Demo

### Autonomous exploration in a low-visibility environment

<p align="center">
  <video src="docs/media/sar_demo.mp4" controls width="90%">
    Your browser does not support embedded video.
  </video>
</p>

The rover uses LiDAR for mapping and navigation, RGB-D perception for target localisation, and Nav2 for local and global motion control. The low-visibility environment is used as a robustness and demonstration scenario; the clean building is used for the main controlled experiments.

### System view

<p align="center">
  <img src="docs/media/gazebo_rviz.png" width="90%">
</p>

Gazebo simulation and the corresponding SLAM/navigation map in RViz.

### Exploration result

<p align="center">
  <img src="docs/media/ppo_09696_map.png" width="90%">
</p>

Representative PPO Gazebo run at 0.7 m/s: 96.96% final map coverage, 10 target detections and 5 inspected targets.

---

## Research question

The main question is:

> **Can a learned high-level policy choose exploration goals better than a hand-designed frontier-utility rule when both are given the same feasible candidate goals and the same low-level navigation stack?**

The comparison is deliberately limited to the goal-selection layer.

Both policies receive the same candidate set and use the same:

- SLAM map
- robot state estimate
- target database
- Nav2 planner/controller
- stuck watchdog
- recovery behaviour
- target inspection pipeline
- result logging

The main measures are target inspection, time, distance and recovery behaviour. Coverage is reported as a secondary measure rather than treating complete map coverage as the objective by itself.

---

## System overview

```text
                 LiDAR + RGB-D + IMU
                          |
                          v
                SLAM Toolbox + EKF
                          |
                occupancy map + pose
                          |
             +------------+-------------+
             |                          |
             v                          v
       Frontier generation        YOLO perception
             |                          |
             |                    semantic database
             |                          |
             +------------+-------------+
                          |
                  Shared candidate set
                          |
                8 frontier candidates
                + up to 8 target candidates
                          |
             +------------+-------------+
             |                          |
             v                          v
        Mission 2 rule             PPO policy
       hand-designed                learned
             |                          |
             +------------+-------------+
                          |
                       goal pose
                          |
                          v
                  Nav2 / MPPI
                          |
                          v
                       Rover

The candidate-generation and navigation layers are shared. The experiment changes the policy that chooses the next goal.
Robot and software stack
Component	Implementation
Platform	4-wheel differential-drive rover
OS	Ubuntu 24.04
Middleware	ROS 2 Jazzy
Simulator	Gazebo
Drive	ros2_control, differential-drive controller, twist_mux
LiDAR	2D LiDAR, 360 samples, 10 m range
Camera	Mast-mounted RGB-D camera
IMU	Wheel odometry + IMU fusion
State estimation	robot_localization EKF
Mapping	SLAM Toolbox, 0.05 m resolution
Navigation	Nav2 with MPPI
Perception	Ultralytics YOLO with tracking and depth-based localisation
Target representation	Semantic target database
RL training	PyTorch
RL deployment	NumPy policy inside ROS 2


The mast-mounted RGB-D camera was introduced after early experiments showed that a low-mounted camera was poorly suited to observing low-lying objects and scene structure.
Exploration methods
Mission 2
The original hand-designed exploration policy is implemented in:
src/frontier_explorer/frontier_explorer/mission2.py

The policy first handles known, uninspected targets and otherwise selects a frontier using an information-gain/travel-cost utility.
For frontier exploration:
U = alpha * IG_norm - beta_eff * Cost_norm

with:
alpha = 3.0
beta  = 0.25

The effective travel-cost weight increases as the simulated battery model drains.
Information gain is estimated from visible unknown cells along LiDAR-like rays, while travel cost comes from the Nav2 path.
The policy also includes:
- visited frontier filtering
- temporary frontier blacklisting
- target retry limits
- unreachable-target abandonment
- stuck detection
- recovery behaviour
- mission completion logic
Shared candidate generation
The PPO policy does not receive an entirely different exploration problem.
Both policies receive the same candidate set.
The candidate generator:
1. clusters frontier cells
2. removes visited and temporarily blacklisted clusters
3. selects the largest frontier clusters
4. computes safe goal cells
5. estimates information gain
6. keeps the eight highest-information frontier candidates
7. generates stand-off points for available targets
8. keeps up to eight target candidates with the shortest Nav2 paths
This produces a maximum 16-action space:
8 frontier candidates
+
8 target candidates
=
16 possible actions

Invalid candidates are masked.
This makes the comparison:
Same candidates
      |
      +---- Mission 2 utility rule
      |
      +---- PPO policy
      |
      v
Same Nav2 stack

rather than comparing two completely different exploration systems.
PPO policy
The PPO implementation is in:
sar_training/sar_rl/policy.py
sar_training/train_ppo.py

The policy uses a shared candidate encoder followed by self-attention layers so that candidate decisions can be made in the context of the other available options.
Each candidate contains 13 features including:
- target/frontier type
- Nav2 path length
- relative path cost
- visible information gain
- frontier cluster size
- robot-relative position
- target confidence
- failed attempts
- goal clearance
- minimum path clearance
- target offset
- distance from previously selected goals
Global state contains information such as:
- known free area
- number of frontier candidates
- battery state
- elapsed time
- available targets
- inspected targets
- abandoned targets
- stuck events
- decision count
- empty cycles
The reward is task-oriented. It rewards target inspection and useful exploration while penalising time, distance, stuck events and abandoned targets.
The main checkpoint used in the current experiments:
run16_baseline
800,000 training decisions

The trained PyTorch policy is exported to:
sar_training/policy16.npz

and the ROS deployment uses the NumPy representation rather than requiring PyTorch at runtime.
2D simulation
Training directly in Gazebo is too slow for large numbers of PPO episodes. A fast simulator was therefore built under:
sar_training/sar_rl/

One simulator step corresponds to one high-level exploration decision.
What is shared with Mission 2
The simulator reproduces the main decision-making logic:
- frontier detection
- frontier clustering
- safe goal-cell selection
- information-gain estimation
- visited/blacklisted frontiers
- target availability
- target stand-off points
- completion logic
What is approximated
The following Gazebo components are represented statistically:
- Nav2 path planning
- driving time
- stuck events
- YOLO detection
- field of view
- line of sight
- localisation noise
- duplicate detections
- false detections
- snapshot failures
The simulator contains both a replica of the main Gazebo building and procedurally generated corridor/room layouts.
<p align="center">
  <img src="sar_training/sim_v2_mission2_like.png" width="48%">
  <img src="sar_training/sim_random_mission2_like.png" width="48%">
</p>

<p align="center">
  <em>Example simulator environments: building replica and generated layout.</em>
</p>

The simulator also contains a calibrated configuration based on a real Gazebo Mission 2 run. The results below, however, were produced with the default simulator configuration rather than the calibrated configuration.
Simulator results
The main PPO evaluation used 50 paired episodes per condition, with the same random seeds for both policies.
Values are mean ± standard deviation.
Building replica — nominal physics
Metric	Mission 2	PPO
Mission time (s)	613 ± 116	730 ± 223
Distance (m)	104 ± 20	135 ± 39
Coverage (%)	99.99	99.96
People inspected	4.98 ± 0.8	5.44 ± 0.8
Stuck events	4.44 ± 1.9	4.52 ± 2.7
Abandoned entries	2.14 ± 1.3	2.58 ± 1.9


PPO inspected slightly more people, but required more time and distance.
Building replica — randomized physics
Metric	Mission 2	PPO
Mission time (s)	677 ± 211	759 ± 243
Distance (m)	117 ± 29	138 ± 35
Coverage (%)	99.81	100.00
People inspected	4.90 ± 1.0	5.22 ± 1.0
Stuck events	4.34 ± 3.1	4.60 ± 3.2
Abandoned entries	3.24 ± 2.7	3.72 ± 2.5


Again, PPO achieved slightly higher inspection and coverage, but at a higher time and distance cost.
Generated layouts — randomized physics
Metric	Mission 2	PPO
Mission time (s)	862 ± 314	1076 ± 389
Distance (m)	168 ± 59	216 ± 79
Coverage (%)	97.44	97.57
People inspected	5.20 ± 2.4	5.44 ± 2.3
Stuck events	4.54 ± 2.4	6.14 ± 3.3
Abandoned entries	4.14 ± 2.8	4.96 ± 3.0


Current interpretation
The current PPO policy does not outperform Mission 2 overall.
It tends to inspect slightly more targets, but takes longer, drives farther and can experience more stuck events.
This behaviour is consistent with the reward configuration used for run16_baseline, which placed relatively little penalty on time and distance.
That result is kept as part of the experiment rather than treating it as a failure to be hidden. The next experiment is to retrain on the calibrated simulator with stronger penalties for inefficient travel and recovery events.
Gazebo experiments
The PPO policy was subsequently deployed into the full ROS 2/Gazebo system.
The deployment node supports three modes:
Mode	Description
mission2	Original Mission 2 node
mission2c	Mission 2 decision rule using the shared candidate set
rl	PPO selecting from the shared candidate set


The deployment architecture therefore allows the same perception, navigation and recovery stack to be used for all three modes.
Each experiment records:
experiment_summary.csv
frontier_decisions.csv
target_milestones.csv
rl_decisions.csv
rl_candidates.csv
rl_globals.csv

The RL-specific logs allow the selected candidate to be examined together with the alternatives that were available at each decision.
Representative Gazebo results
These are individual runs, not statistical averages.
They are useful for showing system behaviour but should not be interpreted as the final quantitative comparison.
PPO — 0.7 m/s
Metric	Result
Runtime	1212.77 s
Distance	49.91 m
Final coverage	96.96%
Targets detected	10
Targets inspected	5
Targets abandoned	2
First detection	74.84 s
First inspection	109.71 s
High-level decisions	33


<p align="center">
  <img src="docs/media/ppo_09696_map.png" width="90%">
</p>

Mission 2 — 0.7 m/s
A representative Mission 2 run under the same nominal speed:
Metric	Mission 2
Runtime	1211.96 s
Distance	48.39 m
Final coverage	85.69%
Targets detected	9
Targets inspected	6
Targets abandoned	3
First detection	38.87 s
First inspection	79.45 s


PPO vs Mission 2 — representative 0.7 m/s runs
Metric	PPO	Mission 2
Runtime	1212.77 s	1211.96 s
Distance	49.91 m	48.39 m
Coverage	96.96%	85.69%
Targets detected	10	9
Targets inspected	5	6
Abandoned	2	3
First detection	74.84 s	38.87 s
First inspection	109.71 s	79.45 s


This single comparison is interesting but not sufficient to establish that either policy is better.
PPO explored more of the map in this run and abandoned fewer targets, while Mission 2 detected and inspected a target earlier.
Repeated matched runs are required before drawing a general conclusion.
Earlier Gazebo Mission 2 reference
An earlier longer Mission 2 run produced:
Metric	Result
Runtime	1999.8 s
Distance	93.5 m
Final coverage	97.35%
Target entries detected	15
Targets inspected	11
Targets abandoned	4
First detection	24.8 s
First inspection	51.8 s
Stuck events	13
Average decision latency	186 ms


The target count here refers to semantic-database entries. Multiple entries can correspond to the same physical person because of duplicate detections.
The Gazebo worlds contain six person models.
Perception and navigation engineering
The project also involved several system-level changes that were necessary before the exploration experiments were useful.
RGB-D camera placement
Early versions used a low-mounted camera. This caused problems with low-lying objects and scene visibility.
The camera was moved to a mast-mounted configuration and combined with depth for 3D target localisation.
<p align="center">
  <img src="docs/media/mast_rgbd.png" width="80%">
</p>

This separated semantic target detection from the 2D LiDAR used primarily for navigation and mapping.
Opportunistic target capture
Target evidence is captured when the perception pipeline has a sufficiently confident detection within the configured distance range, rather than introducing a separate blocking rotate-and-capture action.
The reasoning is practical: during a search mission, a target can move or disappear while the robot is performing an additional alignment manoeuvre.
The approach therefore favours:
detect
  ↓
localise
  ↓
capture evidence when conditions are satisfied
  ↓
continue navigation

rather than:
detect
  ↓
stop
  ↓
rotate to target
  ↓
verify orientation
  ↓
capture
  ↓
resume navigation

The latter introduces additional control and verification dependencies.
Runtime measurements
The perception stack was also profiled to quantify its effect on the exploration loop.
One PyTorch CPU + YOLO configuration produced:
Component	Measured value
Average decision latency	329.1 ms
Frontier detection	73.2 ms
Information-gain evaluation	88.9 ms
Goal-cell selection	~8–22 ms
Async path planning	108.7 ms


The main measurable increase came from CPU contention during perception and information-gain evaluation.
These measurements are implementation diagnostics rather than the main exploration result.
Navigation planner comparison
A separate diagnostic experiment compared NavFn and SMAC 2D under a perception-heavy configuration.
Metric	NavFn + YOLO	SMAC 2D + YOLO
Runtime	21:23.9	21:36.5
Distance	140.02 m	188.67 m
Coverage	59.66%	73.55%
Frontier decisions	18	47
Targets detected	9	4
Targets inspected	6	2
Abandoned	2	2
First detection	413.38 s	58.78 s
First inspection	481.72 s	138.24 s
Average decision latency	251.96 ms	278.00 ms
Frontier processing	81.62 ms	86.40 ms
Path planning	99.67 ms	105.69 ms


This was a diagnostic comparison rather than part of the primary PPO-vs-Mission2 experiment.
Environments
The main building contains:
- west entrance
- east-west corridor
- six rooms
- multiple doorway widths
- loop doors between rooms
- clutter
- six person models
The repository contains several versions of the environment.
World	Description
baselinebest.sdf	Clean version of the main building
baseline4.sdf	Clean main-building variant
gazebo1.sdf	Low-visibility / cluttered version
underground.sdf	Low-visibility building variant
baseline1.sdf–baseline3.sdf	Earlier experimental worlds


The clean environment is used for controlled quantitative comparisons.
The darker environment adds visual clutter, fog, debris and lighting changes and is primarily useful for robustness experiments and demonstrations.
Repository structure
src/
├── minibot/
│   ├── description/        Robot URDF/Xacro
│   ├── launch/              Camera, EKF/SLAM/Nav2 and semantic launch files
│   ├── config/              Nav2 and motion configuration
│   └── worlds/              Gazebo environments
│
├── frontier_explorer/
│   └── frontier_explorer/
│       ├── mission2.py      Main hand-designed exploration policy
│       └── mission2withc.py Earlier exploration variant
│
├── semantic_perception/
│   └── semantic_perception/
│       ├── yolo_detector.py
│       └── semantic_database.py
│
└── semantic_interfaces/
    └── msg/                 Semantic target messages

sar_training/
├── sar_rl/                  2D simulator and PPO implementation
├── train_ppo.py             PPO training
├── eval_sim.py              Simulator evaluation
├── check_sim.py             Simulator sanity checks
├── gazebo_deploy/
│   ├── explorer_rl_node.py
│   ├── rl_explorer_launch.py
│   ├── export_policy.py
│   └── rl_policy_np.py
├── runs/                    Training checkpoints and logs
└── policy16.npz             Exported run16 PPO policy

behavior_trees/
└── navigate_to_pose_w_smoothing.xml

yolo_ros/
└── ROS 2 YOLO integration

Running the system
Requirements
The main development environment is:
Ubuntu 24.04
ROS 2 Jazzy
Gazebo
Nav2
SLAM Toolbox
robot_localization
ros2_control
twist_mux
ros_gz
Ultralytics YOLO
OpenVINO

Build the workspace:
git clone https://github.com/yuvanbruh/searchandrescuerover.git
cd searchandrescuerover

colcon build --symlink-install
source install/setup.bash

Mission 2 in Gazebo
Start the Gazebo/robot/camera system:
ros2 launch minibot cam.py

Start EKF, SLAM and Nav2 using the current launch file in the repository:
ros2 launch minibot ekf.py

Start the semantic perception and Mission 2 stack:
ros2 launch minibot semantic.py

The exact launch arguments can be adjusted for the selected Gazebo world.
PPO in Gazebo
Start the robot, camera, EKF, SLAM and Nav2 stack first.
Then run the PPO deployment:
cd ~/last
source install/setup.bash

ros2 launch sar_training/gazebo_deploy/rl_explorer_launch.py \
  base_module:=frontier_explorer.mission2 \
  policy_mode:=rl \
  model_path:=$PWD/sar_training/policy16.npz \
  script_path:=$PWD/sar_training/gazebo_deploy/explorer_rl_node.py \
  experiment_name:=rl16_01

For the fair hand-designed baseline using the same candidate generator:
ros2 launch sar_training/gazebo_deploy/rl_explorer_launch.py \
  base_module:=frontier_explorer.mission2 \
  policy_mode:=mission2c \
  script_path:=$PWD/sar_training/gazebo_deploy/explorer_rl_node.py \
  experiment_name:=mission2c_01

The RL deployment launch also starts the relevant perception components, so duplicate semantic launch processes should not be started simultaneously.
Training the PPO policy
The simulator does not require ROS.
cd sar_training

pip install numpy scipy matplotlib torch gymnasium

Run a sanity check:
python check_sim.py --episodes 5 --png

Train PPO:
python train_ppo.py \
  --name run1 \
  --total-steps 800000

Evaluate a trained policy:
python eval_sim.py \
  --ckpt runs/run1/final.pt \
  --episodes 50

Export a trained PyTorch checkpoint for ROS deployment:
python gazebo_deploy/export_policy.py \
  --ckpt runs/run1/final.pt \
  --out policy.npz

The export script checks the NumPy implementation against the PyTorch policy before writing the deployment file.
Current limitations
The project is still under active development.
1. PPO does not currently outperform Mission 2 overall
The 50-episode simulator evaluation shows a small improvement in target inspection but higher time and travel cost.
2. The current PPO reward needs another training iteration
The run16_baseline policy was trained using the default reward configuration. Time and distance penalties were relatively weak, which is consistent with the learned policy travelling farther and taking longer.
The next training condition will use the calibrated simulator and stronger efficiency/recovery penalties.
3. Gazebo statistics are not yet sufficient
The current Gazebo results include representative runs and one matched 0.7 m/s PPO/Mission 2 comparison.
Repeated matched runs are still required before making a statistical claim about the real ROS 2/Gazebo system.
4. Simulation is an approximation
The 2D environment approximates:
- Nav2 planning
- perception timing
- YOLO errors
- localisation noise
- stuck behaviour
- driving time
The simulator is calibrated against Gazebo, but the sim-to-Gazebo gap has not yet been fully quantified.
5. Semantic exploration is a later experiment
The current main comparison uses the geometric Mission 2 candidate-generation pipeline.
A semantic-aware exploration branch is being developed separately so that structure-aware and learned exploration can be evaluated without changing the primary baseline retrospectively.
Research status
The current project has three layers of evaluation:
                 ROS 2 / Gazebo system
                         ^
                         |
                 deployment tests
                         ^
                         |
                  2D simulator
                         ^
                         |
                PPO training/evaluation

The immediate research direction is:
1. retrain PPO with the calibrated simulator and stronger efficiency penalties
2. run repeated matched Gazebo experiments
3. compare Mission 2, Mission 2 with the shared candidate generator, and PPO
4. evaluate the clean and low-visibility environments separately
5. report target inspection, time, distance, coverage, abandonment and stuck-event distributions
6. investigate semantic structure cues as a separate extension
The goal is not to assume that PPO is better than the hand-designed method. The experiment is intended to determine when, where and under what conditions a learned high-level policy is useful for autonomous search and rescue exploration.
Credits and licenses
Parts of the robot description and simulation structure were adapted from:
- YJ0528/minibot
- Articulated Robotics
The yolo_ros/ directory is based on:
- mgonzs13/yolo_ros
Object detection uses:
- Ultralytics
twist_stamper is by Josh Newans.
Please refer to the original repositories and included license files for the licensing terms of third-party components.
The license for the original research code in:
frontier_explorer
semantic_perception
sar_training

will be specified separately.

### The media files I'd use

Before committing the README, make this directory:

```bash
mkdir -p docs/media

Then put only these in it:
docs/media/sar_demo.mp4
docs/media/gazebo_rviz.png
docs/media/ppo_09696_map.png
docs/media/mast_rgbd.png
