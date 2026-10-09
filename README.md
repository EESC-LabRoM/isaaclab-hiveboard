# Isaac Lab - HiveBoard Multi-Robot Manipulation Suite

[![Isaac Lab](https://img.shields.io/badge/IsaacLab-78b12aed-blue.svg)](https://github.com/isaac-sim/IsaacLab/commit/78b12aed1d2a56439493b8ea6d3538e6e379e2e5)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

Isaac Lab tasks for **Boston Dynamics Spot with Arm**, **Franka FR3** and
**ANYmal + DynaArm** operating the mechanisms of the
**[HiveBoard Benchmark](https://github.com/hiveboard-bench/HiveBoard)**, simulated
with Newton MJWarp (no Isaac Sim needed).

---

## Installation

```bash
git clone --recurse-submodules https://github.com/hiveboard-bench/isaaclab-hiveboard.git
cd isaaclab-hiveboard
# already cloned? git submodule update --init --recursive

uv sync --python 3.12
```

Optional extras: `--extra curobo` (cuRobo-planned commands, needs CUDA),
`--extra cameras`, `--extra data`.

The Newton USD assets are build outputs and must be generated once on a fresh
clone:

```bash
uv run python scripts/generate_newton_usd.py --verify-only   # what is missing
uv run python scripts/generate_newton_usd.py \
  --uuc-python /path/to/uuc-venv/bin/python                  # generate
just generate-anymal-usd                                     # ANYmal assets
```

URDF conversion uses
[urdf-usd-converter](https://github.com/newton-physics/urdf-usd-converter);
`--uuc-python` points at a Python that can `import urdf_usd_converter`.

---

## Available tasks

Each task is `Isaac-HiveBoard-<Robot>-<Tool>-v0`, with a `-Play-v0` variant
(fixed reset, for watching and evaluating). In the table below, ✓ marks the
robot/tool pairs that have a working heuristic and ✗ marks those that don't
have one yet.

| Tool | Spot | Franka | ANYmal |
| --- | :-: | :-: | :-: |
| `BallValve` | ✓ | ✓ | ✓ |
| `SmallValve` | ✓ | ✓ | ✓ |
| `HighTorqueValve` | ✓ | ✓ | ✓ |
| `CircuitBreaker` | ✓ | ✓ | ✓ |
| `Button` | ✓ | ✓ | ✓ |
| `Key` | ✓ | ✓ | ✓ |
| `Drawer` | ✗ | ✓| ✓|
| `M8Thread` / `M30Thread` | ✗ | ✓ | ✓ |
| `PegInsertion` | ✗ | ✗ | ✗ |
| `ShockAbsorber` | ✗ | ✗ | ✗ |
| `Lamp` | ✓ | ✓ | ✓ |

List everything registered:

```bash
just list-envs
```

---

## Running the samples

The quickest way is the `play` recipe, which runs a task's `-Play-v0` variant
in the Newton viewer:

```bash
just play Spot BallValve
just play Franka CircuitBreaker
just play Anymal SmallValve
```

Or call the player directly for full control:

```bash
uv run python scripts/play.py \
  --task Isaac-HiveBoard-Spot-BallValve-Play-v0 \
  physics=newton_mjwarp --visualizer newton
```

Useful options:

| Option | Effect |
| --- | --- |
| `--visualizer none` | Run without a window |
| `--device cpu` | Run on CPU |
| `--num_envs N` | Spawn N environments |
| `--duration S` / `--max-steps N` | Stop after S simulated seconds / N steps |
| `--pose-debug` | Print TCP pose error each interval |
| `--contact-debug` | Print finger and object contact forces |
| `--setup FILE` / `--no-setup` | Use a saved command setup / the sequence as coded |
| `--video` | Save an MP4 under `videos/` (`--video-renderer rtx` for path tracing) |
| `--no-dataset` | Skip HDF5 episode recording under `logs/recorded_datasets/` |

Lamp tasks need their asset generated first:

```bash
uv run python scripts/generate_newton_usd.py --assets lamp
uv run python scripts/play.py --task Isaac-HiveBoard-Franka-Lamp-v0 physics=newton_mjwarp --visualizer newton
```

### Record every task

```bash
just record-all                           # all tasks, headless, RTX
just record-all --list                    # preview the task IDs
just record-all --match Spot --duration 5 # only IDs containing "Spot"
just record-all --renderer newton         # fast rasterized videos
```

Videos and a `summary.json` go to a dated folder under `videos/environments/`.
`ffmpeg` and `ffprobe` must be on `PATH`.

---

## Editing commands

Each task drives the robot through a sequence of commands (go to a pose,
rotate, close/open the gripper, ...). Edit it in the browser with the Viser
command editor:

```bash
just edit-commands                                          # default: Spot ball valve
just edit-commands --task Isaac-HiveBoard-Franka-BallValve-Play-v0
just edit-commands --setup logs/command_setup.json          # reopen a saved file
# then open http://localhost:8080
```

In the editor:

- **GoTo:** select a bead or command and drag its position and orientation.
  Pick an object frame in **Reference** to keep the goal relative to it. Edit
  speeds, tolerances and the gripper state below.
- **Rotate / Screw:** drag the pivot, set the axis, angle, angular speed and
  screw travel.
- **Open / Close gripper:** choose the state and hold time. Duplicate, insert,
  reorder or delete commands with the sequence controls.
- **Phase:** tag each command `approach`, `engage`, `grip`, `actuate`,
  `release` or `retreat`.
- **TCP offset:** enable **Drag TCP offset** to place the tool center point
  relative to the end-effector body.
- **Preview:** scrub or play the sequence; residuals show whether IK reached
  each goal.

The preview is kinematic only: objects stay put and no collisions, forces or
cuRobo plans are simulated. Click **Save setup** to write the file.

### Where setups live

- With no `--setup`, the editor and `play.py` load `configs/<task>.json`.
- A `-Play-v0` variant falls back to its base task's file, so both share one
  sequence. A task's own file always wins.
- Saving updates the loaded file, or creates `configs/<task>.json`.
- `--out FILE` saves elsewhere; `--port` changes the browser port.

Check a setup with real physics:

```bash
uv run python scripts/play.py \
  --task Isaac-HiveBoard-Spot-BallValve-Play-v0 \
  --setup configs/Isaac-HiveBoard-Spot-BallValve-v0.json
```

Replay draws the active command path (waypoints, next target and TCP frames);
hide it with `--no-show-command-path`.

### Spot website clip

The `BenchValve` tasks replay a fixed joint trajectory instead of commands.
Edit it with:

```bash
just edit-spot-traj        # Newton viewer: Example Options → Trajectory Editor
just retarget-spot-traj    # re-solve IK without the viewer
```

---

## Repository structure

```
isaaclab-hiveboard/
├── configs/                 # Saved command setups, one per task
├── dependencies/HiveBoard/  # Git submodule: HiveBoard URDF/USD models
├── scripts/                 # play.py, command_edit.py, record_all_envs.py, ...
└── source/isaaclab_hiveboard/isaaclab_hiveboard/
    ├── assets/              # Robot and HiveBoard asset configs
    ├── mdp/                 # Actions, commands, events, observations
    ├── tasks/               # spot/, franka/, anymal/
    └── utils/
```
