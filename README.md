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

### Record every environment at real-time playback speed

```bash
just record-all
# Equivalent:
uv run python scripts/record_all_envs.py
```

The recorder discovers all environments registered in
`source/isaaclab_hiveboard/isaaclab_hiveboard/tasks/__init__.py`, including
Play variants, aliases, and `validate_command_spot`. Each runs in a separate
process with one environment, recording up to 10 simulated seconds or the first
episode end. HDF5 and joint-tracking logs are disabled for this video batch.

Each video's FPS is calculated **after resolving the task configuration** as
`1 / (sim.dt * decimation)`. For example, `dt=1/200` with decimation `15`
records at `40/3` (approximately 13.33) FPS; the 50 Hz bench tasks record at
50 FPS. One frame is captured per environment step, preserving simulated time
even when rendering runs slowly. Fractional rates are preserved; the duration
limit rounds up to a whole environment step.

The default runs without a window. It captures `scene_cam` when available and
otherwise uses Isaac Lab's perspective video recorder with the task's configured
view. `--viewer` also opens the live Newton viewer. Both `ffmpeg` and `ffprobe`
must be on PATH, and the tasks require their usual GPU/runtime and assets.

```bash
just record-all --list                           # Preview all registered IDs
just record-all --duration 30                    # Up to 30 simulated seconds each
just record-all --match Spot --duration 5         # Only IDs containing Spot
just record-all --task validate_command_spot      # One exact ID; --task can repeat
just record-all --viewer --match BenchValve       # Watch while recording
just record-all --dry-run                        # Print commands without running
```

Videos and per-environment logs are saved under a new dated folder in
`videos/environments/` (change the parent with `--output`). `summary.json`
records each outcome, actual FPS, frame count, video duration, and log path.
Failures do not stop the batch; the command exits nonzero if any recording
fails. `--timeout` sets the wall-time limit per task (default 900 seconds,
including startup). Interrupted or failed encodes may leave `.partial.mp4`
files; only finalized, probed MP4s count as successful recordings.

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

> cuRobo-planned tasks run with `--num_envs > 1`. Each environment owns its
> plan and is solved on its own as it enters a segment, since the cached cuRobo
> solvers take one problem at a time. Physics is parallel but planning is not:
> a segment start costs about 0.8 s of solve per environment entering it.

**2. Behaviour cloning.**

```bash
uv run python scripts/imitation/train_bc.py \
  --task Isaac-HiveBoard-Spot-BallValve-v0 \
  --dataset logs/imitation/datasets/expert_<stamp>.hdf5
```

Actions are rescaled per dimension into `[-1, 1]` before training, and the
stats are written to `action_norm.json` beside the checkpoint so
`RobomimicPolicy` can invert them. This is not optional book-keeping:
robomimic's actor ends in a `tanh` and physically cannot emit anything outside
`[-1, 1]`, so a joint-position task commanding radians trains to a plateau
instead of converging. On the ball valve the difference is a final L2 of
2e-05 with normalization against 0.246 without.

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
records them, since neither Spot's gripper nor the 2F-140 has force sensing on
hardware.

`Isaac-HiveBoard-Anymal-BallValve-v0` carries the same group, keyed
identically, so the pipeline runs on either robot with no change beyond
`--task`. Only the underlying joints differ: six DynaArm joints in
`arm_joint_pos`/`arm_joint_vel` against Spot's seven, and `gripper_pos` reading
the 2F-140's `finger_joint` motor (the rest of the parallel linkage is driven
by USD constraints, so it carries no independent information). Datasets from
the two robots are *not* interchangeable - the key shapes differ.

The Spot lamp group follows the same shape, with `object_pos`, `object_quat`
and `lamp_joint_pos` as its privileged tier.

To use this pipeline on another task, give that task a `bc` observation group
with `concatenate_terms = False` and register a
`robomimic_bc_cfg_entry_point` pointing at a config JSON - see
`tasks/spot/lamp/configs/observations.py` and
`tasks/spot/lamp/agents/robomimic/bc.json`.

### Task status

The scripted expert must actually solve a task before any of this produces
data. Always check the ceiling first:

```bash
uv run python scripts/imitation/eval_policy.py --expert --task <task> --num_envs 1
```

**`Isaac-HiveBoard-Spot-BallValve-v0`** needed two task settings corrected
before it could ever report success: `episode_length_s` 5.0 -> 25.0 (the
-90 degree turn alone takes ~5.2 s at 0.3 rad/s, so episodes timed out
mid-turn) and the success tolerance, which was `math.radians(0.010)` - 0.01
degrees - against the ~0.9 degrees the expert actually achieves.

With those fixed the expert scores 100% on the deterministic `-Play-v0`
variant, but two *independent* reset settings still defeat it on the randomized
task, each measured by toggling it alone:

| Setting | Expert success |
| --- | --- |
| Deterministic reset, no randomization | 4/4 |
| `valve_joint_parameters` enabled | 0/8 |
| `valve_joint_parameters` disabled | 3/3 |
| Full pose randomization, friction term disabled | 0/8 |

`valve_joint_parameters` sets the valve's revolute joint friction to 0.01-0.10
(`operation="abs"`), which resists the gripper across its whole sampled range.
The `reset_valve_root` pose ranges (+-0.20 m x, +-0.30 m y and z, +-30 degrees
roll/pitch, +-36 degrees yaw) put the valve outside what cuRobo plans to
reliably; plans fail and the expert falls back to direct servoing.

Both need retuning against the gripper's achievable torque and the arm's
reachable workspace. Until then, collect with the friction term switched off:

```bash
uv run python scripts/imitation/collect_demos.py \
  --task Isaac-HiveBoard-Spot-BallValve-v0 --num_envs 1 \
  --disable_events valve_joint_parameters
```

**`Isaac-HiveBoard-Anymal-BallValve-v0`** needed three corrections before its
expert could produce anything. Two were the settings the Spot task had already
needed - `episode_length_s` 5.0 -> 25.0 and the `math.radians(0.010)` success
tolerance -> 0.035 rad - and they were necessary but not sufficient.

The third was an actuator, and it is worth recording how it presented, because
it looked exactly like a bad grasp pose. The sequence stalled at command index
1 (`lever_pivot`) with the TCP 9.4 cm high and 16.6 degrees off, cuRobo
reporting a good plan tracked to its last waypoint:

```
[SEQ]   env=0 seg 0 done in 2.40s -> seg 1
[STALL] env=0 stuck on seg 1/6 for >=4.0s
[STALL] target_pos_b=[0.970, -0.024, 0.204] ee_pos_b=[0.956, -0.022, 0.297]
        pos_err_m=0.0940 ori_err_deg=16.63
[STALL] curobo: waypoint 21/21
```

Authoring a tuned `configs/Isaac-HiveBoard-Anymal-BallValve-v0.json` moved the
target but not the error, which ruled out the command sequence. Comparing
commanded against measured joints found the cause: five of the six arm joints
tracked to 0.002 rad, while `dynaarm_wrist_flexion` held a steady **0.29 rad**
offset with its actuator pinned at the 40 N.m ceiling - against a 0.63 kg
gripper assembly whose gravity load is ~1.3 N.m. That 0.29 rad *is* the
16.6 degrees of orientation error and most of the 9.4 cm.

It was the discrete PD fighting itself, the same failure the comment above
`dynaarm_forearm` in `assets/anymal/bench.py` describes for the roll joints:
wrist flexion was the last light joint still on the flat 200/20 gains with no
armature. Raising the effort ceiling makes it *worse* (the ringing gets more
authority - the sequence then fails to finish even segment 0). Soft gains plus
armature, matching its neighbours, fix it:

| `dynaarm_wrist_flex` | stiffness | damping | armature | Expert on `-Play-v0` |
| --- | --- | --- | --- | --- |
| Before | 200.0 | 20.0 | none | 0/3 |
| Raised effort ceiling to 200 N.m | 200.0 | 20.0 | none | 0/2, worse |
| After | 40.0 | 1.5 | 0.01 | **2/2** |

The full sequence now runs: approach, grasp, a 5.3 s rotate, release, retreat.

On the randomized task the blocker is the opposite of Spot's. Toggling each
randomization term alone:

| Setting | Expert success |
| --- | --- |
| Full randomization | 0/8 |
| `valve_joint_parameters` disabled (friction) | 0/6 |
| `reset_valve_root` disabled (valve pose) | 6/6 |

Valve friction, which defeats Spot's gripper, does not bother the 2F-140; the
`reset_valve_root` pose ranges (+-0.20 m x, +-0.30 m y and z, +-30 degrees
roll/pitch, +-36 degrees yaw) do. Until those ranges are narrowed to what the
DynaArm reaches, collect with the pose term switched off - friction, material
and valve-actuator randomization all stay on:

```bash
uv run python scripts/imitation/collect_demos.py \
  --task Isaac-HiveBoard-Anymal-BallValve-v0 --num_envs 1 \
  --disable_events reset_valve_root
```

That dataset has no variety in valve pose, so a policy trained on it will not
generalize across valve placements - it teaches the task, not the reach.

**`Isaac-HiveBoard-Spot-Lamp-v0`** does not currently succeed at all. Its
sixteen-quarter-turn sequence is still at command index 1 when the episode
times out, and it fails even with a 150 s episode, so collection writes an
empty dataset.

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

Without `--setup`, the editor loads `configs/<task>.json` if it exists, falling back
to the base task's file for a `-Play-v0` variant. If neither exists, it starts from
the task's built-in settings. An explicit `--setup` takes precedence. By default,
saving updates the loaded file, or creates `configs/<task>.json` if none was loaded.

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

#### Which setup a task runs

With no `--setup`, a task runs `configs/<task>.json` when that file exists.
**A `-Play-v0` variant falls back to its base task's file**: the two differ only
in events and command sampling, so they share one tuned command sequence. Save
a `configs/<task>-Play-v0.json` if a variant ever needs its own - a task's own
file always wins - and pass `--no-setup` to run the sequence as coded in Python.

This fallback applies everywhere a setup is loaded, `scripts/imitation/`
included: the scripted expert *is* the command sequence, so collecting without
the setup would transcribe a different expert from the one you tuned in the
editor and watched in `play.py`.

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
