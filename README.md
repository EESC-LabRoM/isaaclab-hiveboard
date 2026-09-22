# Isaac Lab - HiveBoard Multi-Robot Manipulation Suite

[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3-bffdce9-blue.svg)](https://github.com/isaac-sim/IsaacLab/commit/bffdce9d7467f349bfc8ab111fe633a0bb234851)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

An Isaac Lab extension package for simulating, controlling, and benchmarking various robot platforms (**Boston Dynamics Spot with Arm**, **Franka Emika Panda**, and **ANYmal C/D**) performing manipulation tasks on the **[HiveBoard Benchmark](https://github.com/EESC-LabRoM/HiveBoard)**.

---

## 🚀 Features

- **Multi-Robot Support**:
  - 🐕 **Boston Dynamics Spot with Arm**: Differential IK, CuRobo collision avoidance, and RMPFlow control for gate valves, lever valves, and circuit breakers.
  - 🦾 **Franka Emika Panda**: Precision end-effector tracking with orientation alignments and real-time pose diagnostics.
- **HiveBoard Submodule Integration**: Directly loads CAD/URDF/USD models from `dependencies/HiveBoard` (ball valves, high torque gate valves, small valves, circuit breakers, drawers, keys, buttons).
- **Demonstration Collection**: Fixed-base relative TCP demonstration collector exporting HDF5 datasets compatible with Diffusion Policy.
- **Pose Diagnostics**: Automated frame error decomposition (IK tracking vs facing/jaw alignment).
- **`uv` Ready**: Seamless dependency management and script execution.

---

## 📦 Installation

Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/EESC-LabRoM/isaaclab-hiveboard.git
cd isaaclab-hiveboard
```

If already cloned without submodules, initialize HiveBoard:

```bash
git submodule update --init --recursive
```

Create the Python 3.12 environment and install the local Isaac Lab 3 packages:

```bash
uv sync --python 3.12
```

The default install is kitless: Isaac Sim, camera support, and CuRobo are not
required. Development extras remain available as `--extra cameras`, `--extra
curobo`, and `--extra data`.

---

## 🕹️ Available Environments

| Task ID | Robot | Target Object | Controller / Action |
| --- | --- | --- | --- |
| `Isaac-HiveBoard-Spot-BallValve-v0` | Spot + Arm | Ball (Lever) Valve | Sequential Absolute / Relative IK |
| `Isaac-HiveBoard-Spot-BenchValve-Play-v0` | Spot + Arm | Ball (Lever) Valve | Fixed 50 Hz joint trajectory (website clip) |
| `Isaac-HiveBoard-Spot-Gains-Play-v0` | Spot + Arm | none (robot only) | Same joint clip; PD gain eval / optimize |
| `Isaac-HiveBoard-Spot-CircuitBreaker-v0` | Spot + Arm | Circuit Breaker | Sequential Pose IK |
| `Isaac-HiveBoard-Spot-HighTorqueValve-v0` | Spot + Arm | Gate Valve | Multi-revolution IK |
| `Isaac-HiveBoard-Spot-SmallValve-v0` | Spot + Arm | Small Gate Valve | Multi-revolution IK |
| `Isaac-HiveBoard-Spot-Lamp-v0` | Spot + Arm | Screw-in Lamp | TCP pose IK with screw coupling |
| `Isaac-HiveBoard-Franka-Lamp-v0` | Franka FR3 | Screw-in Lamp | TCP pose IK with screw coupling |
| `Isaac-HiveBoard-Franka-LeverValve-v0` | Franka Panda | Ball (Lever) Valve | Operational Space / Differential IK |
| `Isaac-HiveBoard-Franka-CircuitBreaker-v0` | Franka Panda | Circuit Breaker | Differential IK with facing alignment |

List all available tasks:

```bash
uv run python scripts/list_envs.py
```

---

## 🎮 Running Simulations

### Kitless Spot Ball-Valve Play

The validated task uses Newton MJWarp at 600 Hz, a 20 Hz controller, a fixed
Spot base, committed USD assets, and a simple local floor (no warehouse or
Nucleus assets). Interactive play uses the Newton visualizer and does not
launch Isaac Sim:

```bash
uv run --python 3.12 python scripts/play.py \
  --task Isaac-HiveBoard-Spot-BallValve-Play-v0 \
  --pose-debug --contact-debug \
  physics=newton_mjwarp --visualizer newton
```

`--contact-debug` prints finger/jaw net force and valve-filtered force every
`--pose-debug-interval` steps. HDF5 episode traces (including named
`evaluation` contact terms) are written under `logs/recorded_datasets/`.

The valve reset uses Isaac Lab's standard `reset_root_state_uniform` and
`reset_joints_by_offset` terms. Set a fixed custom start in
`tasks/scenes/lever_valve.py` (`ball_valve.init_state.pos`, `.rot`, and
`.joint_pos`), or edit the corresponding ranges in
`tasks/spot/ball_valve/configs/events.py` for per-episode sampling.

Run without a window:

```bash
uv run --python 3.12 python scripts/play.py \
  --task Isaac-HiveBoard-Spot-BallValve-Play-v0 \
  physics=newton_mjwarp --visualizer none
```

`scripts/play_spot_ball_valve.py` is the pass/fail demo runner: it exits
nonzero unless the physical valve joint reaches the sampled endpoint within
the 15-degree success tolerance.

### Newton lamp tasks

Generate the lamp asset once, then run either robot with the kitless player:

```bash
uv run python scripts/generate_newton_usd.py --assets lamp
uv run python scripts/play.py --task Isaac-HiveBoard-Spot-Lamp-v0 physics=newton_mjwarp --visualizer newton
uv run python scripts/play.py --task Isaac-HiveBoard-Franka-Lamp-v0 physics=newton_mjwarp --visualizer newton
```

The original HiveBoard lamp USD remains in `dependencies/HiveBoard/Simulation/Lamp/`.
Newton uses the generated USD with the URDF's primitive bulb colliders, plus
an environment-side 6 mm/revolution screw coupling. The default sequence uses
sixteen quarter turns. Franka's public FR3 USD is downloaded on first use.
Pass `--device cpu --visualizer none --max-steps 5` for a short headless check.

Saved Franka setups can use `CuroboPlannedGoToFrameCfg` and
`CuroboPlannedRotateFrameCfg` with `robot_joint_names` set to `fr3_joint1`
through `fr3_joint7` (requires CUDA). The arm executes their planned joint
waypoints to preserve cuRobo's elbow configuration. Plain `GoToFrameCfg` and
`RotateFrameCfg` use differential IK; `ScrewFrameCfg` also uses differential
IK to retain the lamp's rotation/translation coupling. Franka uses implicit
joint drives to keep the arm and gripper stable at the lamp task's timestep.

### Website Spot valve playback

`Isaac-HiveBoard-Spot-BenchValve-Play-v0` replays the HiveBoard website Spot
clip: the same 50 Hz joint trajectory, home pose, upright board at chest
height, and arm PD (`kp=500`, `kd=40`) as
`dependencies/hiveboard-bench.github.io`. This is not the IK collection task.

```bash
uv run --python 3.12 python scripts/play.py \
  --task Isaac-HiveBoard-Spot-BenchValve-Play-v0 \
  physics=newton_mjwarp --visualizer newton
```

Headless playback writes commanded vs measured arm joints to
`logs/joint_tracking/<timestamp>/` (`joint_traj.csv`, error plots, RMS summary).
The same log includes valve-filtered contact force and a hit flag for the
finger, lower jaw, `wr1`, `wr0`, `el1`, and `el0`. Pass `--joint-log DIR` to
choose the directory, or `--no-joint-log` to skip.

### Editing the website clip

`scripts/traj_edit.py` is the Newton / Isaac Lab port of the HiveBoard website
`tools/traj_edit.py`. It loads `spot_bench_valve.json`, draws the same TCP
beads the playback markers use, and re-solves damped-least-squares IK on this
Spot instead of MuJoCo.

```bash
uv run python scripts/traj_edit.py physics=newton_mjwarp --visualizer none
# or: just edit-spot-traj
```

This opens Newton's own ViewerGL (not the Isaac Lab visualizer wrapper). The
editor is under **Example Options → Trajectory Editor** in the left panel
(press **H** if the HUD is hidden). Pause, select a bead (cyan), nudge with
**I/K J/L U/O** (Shift = 1 mm) or the xyz fields, then **Solve** / Enter.
**Save** writes `q` and `keys` back to the JSON. Re-IK only a file without the
viewer:

```bash
just retarget-spot-traj
```

### Single-position command validation

`validate_command_spot` contains Spot, ground, and light. It continuously commands
the TCP to `(0.9182, -0.0727, 0.8261)` metres relative to the environment origin,
using the bench-valve sequential command and position IK. There is no automatic
success termination or timeout. The wrist orientation is unconstrained.
CUDA graph caching is disabled so the solver applies the gravity-compensation
settings written at startup.

```bash
just validate-command-spot
# Headless tracking check (prints TCP position error in metres):
just validate-command-spot --headless --visualizer none --max-steps 350
```

Change `TARGET_POSITION_ENV` in
`source/isaaclab_hiveboard/isaaclab_hiveboard/tasks/spot/validate_command_spot/env.py`
to test another position. The target stays active after arrival.

### Robot-only gain baseline

`Isaac-HiveBoard-Spot-Gains-Play-v0` is the same clip with no HiveBoard: ground,
light, and Spot. Use it to tune PD without valve contact. CUDA graphs are off
so live `kp`/`kd` writes take effect.

```bash
uv run --python 3.12 python scripts/play.py \
  --task Isaac-HiveBoard-Spot-Gains-Play-v0 \
  physics=newton_mjwarp --visualizer newton --no-joint-log
```

Score the current bench gains, or search `kp`/`kd`. `--num-envs N` runs N
copies in one physics step and scores a different gain set in each
(`--optimize` uses a (1+λ) log-space search with λ = N):

```bash
uv run --python 3.12 python scripts/optimize_spot_gains.py \
  --num-envs 16 physics=newton_mjwarp --visualizer none

uv run --python 3.12 python scripts/optimize_spot_gains.py --optimize \
  --joints all --num-envs 16 --max-evals 80 \
  physics=newton_mjwarp --visualizer none
```

`--joints` can be `gripper`, `arm`, or `all`. Results go to `logs/gain_opt/`.

```bash
uv run --python 3.12 python scripts/play_spot_ball_valve.py \
  --task Isaac-HiveBoard-Spot-BallValve-Play-v0 \
  physics=newton_mjwarp --visualizer newton
```

### Regenerating Newton USD assets

The `*.usd*` files the Newton tasks load are gitignored build outputs, so a
fresh clone must generate them once. `scripts/generate_newton_usd.py` converts
the committed URDFs with [urdf-usd-converter](https://github.com/newton-physics/urdf-usd-converter)
(kitless; needs a Python that can `import urdf_usd_converter`) and bakes the
valve CoACD overlay. Check what is missing:

```bash
uv run python scripts/generate_newton_usd.py --verify-only
```

Rewrite the valve overlay only (needs `coacd` + `trimesh` in the project venv):

```bash
uv run python scripts/generate_newton_usd.py --skip-conversion
# or: just generate-newton-usd --skip-conversion
```

Full regeneration including URDF conversion:

```bash
uv run python scripts/generate_newton_usd.py --uuc-python /path/to/uuc-venv/bin/python
```

UUC rejects OBJ meshes that list vertices no face uses (leftover CAD
polylines). Strip those before converting Spot:

```bash
just strip-obj-unused-verts
```

To change the assets, edit that script — never hand-edit the generated USD —
then re-run and commit the script.

> [!WARNING]
> Only `Isaac-HiveBoard-Spot-BallValve-v0` and its `-Play-v0` variant are
> validated with Isaac Lab 3 and Newton. The remaining HiveBoard environments,
> cameras, and training configurations still require migration validation.

### Collecting Demonstrations

Record 10 successful demonstrations to HDF5:

```bash
uv run python scripts/collect_demos.py \
  --headless --device cuda:0 \
  --num_demos 10
```

---

## 🧠 Imitation Learning (BC + DAgger)

`scripts/imitation/` trains an MLP policy on scripted-expert demonstrations and
then improves it with DAgger. Datasets are written in robomimic's layout
(`data/demo_<i>/obs/<key>` plus `actions`), so they load both in this repo's
scripts and in Isaac Lab's
`scripts/imitation_learning/robomimic/train.py`.

Install the trainer once:

```bash
uv sync --extra imitation     # adds robomimic v0.4.0
```

**1. Collect expert demonstrations.** The expert is the task's own
`pose_command` sequence, so nothing is learned here - a known-good solution is
transcribed into a trainable dataset.

```bash
uv run python scripts/imitation/collect_demos.py \
  --task Isaac-HiveBoard-Spot-BallValve-v0 \
  --num_demos 50 --num_envs 1
```

The expert adapts to whichever action space a task uses: tasks whose
`pose_command` sets `output_joint_positions` (the ball valve) get the cuRobo
joint waypoints `[q_arm, gripper]` directly, while pose-IK tasks (the lamp) get
`[pos, quat, gripper]`.

> Keep `--num_envs 1` on cuRobo-planned tasks. Their motion planner is built
> with `max_batch_size=1`, so extra environments make every plan throw and fall
> back to direct servoing, which silently degrades the demonstrations.

**2. Behaviour cloning.**

```bash
uv run python scripts/imitation/train_bc.py \
  --dataset logs/imitation/datasets/expert_<stamp>.hdf5
```

**3. DAgger.** Each round rolls the current policy out, asks the expert what it
would have done at every state the policy actually visited, appends those
corrections and retrains. `--beta` is the chance of deferring to the expert on
any step and decays by `--beta_decay` each round.

```bash
uv run python scripts/imitation/dagger.py \
  --initial_dataset logs/imitation/datasets/expert_<stamp>.hdf5 \
  --rounds 5 --episodes_per_round 40 --num_envs 8
```

**4. Evaluate** against the task's own `success` termination term, with
`--expert` giving the ceiling the policy is chasing:

```bash
uv run python scripts/imitation/eval_policy.py --checkpoint <run>/models/model_epoch_600.pth
uv run python scripts/imitation/eval_policy.py --expert     # baseline
```

### Observations and privileged information

The policy reads the task's `bc` observation group. Every term is reproducible
on hardware, in three tiers:

| Tier | Ball valve | Source on hardware |
| --- | --- | --- |
| Proprioception | `eef_pos`, `eef_quat`, `arm_joint_pos`, `arm_joint_vel`, `gripper_pos` | joint encoders + FK |
| Privileged object state | `object_pos`, `object_quat`, `valve_current_angle` | AprilTag on the valve |
| Task specification | `valve_goal_angle`, `valve_task_direction` | commanded by the operator |

The privileged tier is what the AprilTags supply, so no state estimator is
needed at deployment. The task-specification tier matters on the ball valve
because episodes sample both open and close goals - without it a policy cannot
know which way to turn. Contact forces are excluded even though the task
records them, since Spot's gripper has no force sensing on hardware.

The Spot lamp group follows the same shape, with `object_pos`, `object_quat`
and `lamp_joint_pos` as its privileged tier.

To use this pipeline on another task, give that task a `bc` observation group
with `concatenate_terms = False` and register a
`robomimic_bc_cfg_entry_point` pointing at a config JSON - see
`tasks/spot/lamp/configs/observations.py` and
`tasks/spot/lamp/agents/robomimic/bc.json`.

> **Task status:** the scripted expert must actually solve the task before any
> of this produces data. `Isaac-HiveBoard-Spot-Lamp-v0` does not currently
> succeed within its configured `episode_length_s = 10.0`: its sixteen-quarter-turn
> sequence is still at command index 1 when the episode times out, so collection
> writes an empty dataset. Validate a task with `scripts/play.py` (and
> `scripts/imitation/eval_policy.py --expert`) before collecting on it.

---

## 📁 Repository Structure

```
isaaclab-hiveboard/
├── dependencies/
│   └── HiveBoard/               # Git submodule (URDF/USD models & meshes)
├── scripts/                     # Standalone CLI tools (play.py, collect_demos.py, etc.)
│   └── imitation/               # BC + DAgger pipeline (collect, train, dagger, eval)
└── source/
    └── isaaclab_hiveboard/
        ├── config/
        │   └── extension.toml   # Omniverse extension configuration
        ├── setup.py
        └── isaaclab_hiveboard/
            ├── assets/          # Dynamic HiveBoard & robot asset resolvers
            ├── imitation/       # Expert, rollout loop, dataset & policy helpers
            ├── mdp/             # Custom actions, commands, events, observations
            ├── tasks/           # Robot tasks (spot/, franka/, anymal/)
            └── utils/           # Diagnostics & metrics
```

## Interactive command setup

Use the Newton Viser editor to author a `SequentialPoseCommand` task in your browser:

```bash
just edit-commands
# Open http://localhost:8080

# Choose another task or reopen saved settings:
just edit-commands --task Isaac-HiveBoard-Spot-BenchValve-Play-v0
just edit-commands --setup logs/command_setup.json

# Optionally use the CUDA device:
just edit-commands --device cuda:0
```

The editor defaults to CPU and does not require CUDA. The default task is
`Isaac-HiveBoard-Spot-BallValve-Play-v0`. Supply the same `--task`
when loading settings authored for another task. `--port` changes the browser port;
`--out` chooses the save file. Settings are saved only when you click **Save setup**.

- **GoTo:** select a bead or command, then drag its position and orientation. Choose
  an object frame in **Reference** to keep the goal relative to that object, or use
  a fixed environment pose. Edit speeds, tolerances and the gripper state below it.
- **Rotate / Screw:** drag the reference pivot, set the axis in that reference frame,
  and adjust the signed angle, angular speed and screw travel. The arc is shown in
  3D. **Use remaining valve angle** preserves valve-task behavior; turn it off to
  use the entered angle.
- **Open / Close gripper:** choose the state and hold duration. Duplicate, insert,
  reorder or delete commands with the sequence controls.
- **TCP offset:** enable **Drag TCP offset** to hold the robot still while placing
  the tool center point relative to its end-effector body. Numeric translation and
  XYZ Euler rotation controls are also available. Turn calibration off to see the
  arm match the selected goal using that offset.
- **Preview:** scrub a command or play the sequence. Position and orientation
  residuals show whether IK reached the goal. Unreachable goals remain editable.

The editor loads the task's scene once and uses its robot Jacobian for subsequent
edits. This is a **kinematic preview**: objects stay at their reset poses, and IK
does not check collisions, forces or grasp success. cuRobo command settings survive
save/load, but preview does not run the planner. Editing a goal with a dense cuRobo
reference clears that reference so the edited endpoint is used on replay.

Validate the saved setup with the task's normal physics and command handlers:

```bash
uv run python scripts/play.py \
  --task Isaac-HiveBoard-Spot-BallValve-Play-v0 \
  --setup logs/command_setup.json
```

Replay with `--setup` automatically shows the active command path: yellow waypoint
spheres, a green next-waypoint marker, RGB orientation frames along the path, and
the current/target TCP poses. GoTo shows the remaining motion; Rotate/Screw shows
the signed arc; cuRobo commands show their actual planned waypoints. The path
disappears during gripper holds and when the sequence finishes. Use
`--no-show-command-path` to hide the path, or `--show-command-path` to enable it
without a setup file.

GoTo's `canonicalize_upward` selects the upright grasp when the command starts
and keeps that choice while following the reference. This prevents 180° target
flips when a moving handle crosses a horizontal pose.

The JSON stores command parameters and a body-to-TCP transform, with quaternions in
`(x, y, z, w)` order. `play.py --setup` applies the offset to the command term, matching
IK action and `ee_tcp` sensor before creating the environment. Existing task Python
configs are not rewritten. For programmatic use, call
`isaaclab_hiveboard.utils.command_setup.apply_setup(env_cfg, data, task=task_id)`
before `gym.make()`. Joint-trajectory-only tasks continue to use `scripts/traj_edit.py`.

Checks: `just check-command-setup`; scene and Viser startup:
`just edit-commands --device cpu --smoke-test`.
