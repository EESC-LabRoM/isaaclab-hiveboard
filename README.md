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
| `Isaac-HiveBoard-Spot-CircuitBreaker-v0` | Spot + Arm | Circuit Breaker | Sequential Pose IK |
| `Isaac-HiveBoard-Spot-HighTorqueValve-v0` | Spot + Arm | Gate Valve | Multi-revolution IK |
| `Isaac-HiveBoard-Spot-SmallValve-v0` | Spot + Arm | Small Gate Valve | Multi-revolution IK |
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
Nucleus assets). Run the deterministic opening demo with the Newton visualizer:

```bash
uv run --python 3.12 python scripts/play_spot_ball_valve.py \
  --task Isaac-HiveBoard-Spot-BallValve-Play-v0 \
  physics=newton_mjwarp --visualizer newton
```

Run without a window, or select the randomized task and seed:

```bash
uv run --python 3.12 python scripts/play_spot_ball_valve.py \
  --task Isaac-HiveBoard-Spot-BallValve-v0 --seed 7 \
  physics=newton_mjwarp --visualizer none
```

The runner exits nonzero unless the physical valve joint reaches -90 degrees
within the 15-degree success tolerance.

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

To change the assets, edit that script — never hand-edit the generated USD —
then re-run and commit the script.

### Legacy Isaac Sim Play

Play Spot ball valve with camera orbit:

```bash
uv run python scripts/play.py --task "Isaac-HiveBoard-Spot-BallValve-v0" --orbit
```

> [!WARNING]
> Only `Isaac-HiveBoard-Spot-BallValve-v0` and its `-Play-v0` variant are
> validated with Isaac Lab 3 and Newton. The remaining HiveBoard environments,
> legacy player, cameras, recording, and training configurations still require
> migration validation and may require the optional Isaac Sim dependencies.

Play Franka lever valve with pose diagnostics overlay:

```bash
uv run python scripts/play.py --task "Isaac-HiveBoard-Franka-LeverValve-v0" --pose-debug
```

Play Franka circuit breaker:

```bash
uv run python scripts/play.py --task "Isaac-HiveBoard-Franka-CircuitBreaker-v0" --pose-debug
```

### Collecting Demonstrations

Precompute reachable reset cache:

```bash
uv run python scripts/precompute_reset_states.py \
  --headless --device cuda:0 \
  --output_path logs/spot_reset_states.pt
```

Record 10 successful demonstrations to HDF5:

```bash
uv run python scripts/collect_demos.py \
  --headless --device cuda:0 \
  --reset_state_cache_path logs/spot_reset_states.pt \
  --num_demos 10
```

---

## 📁 Repository Structure

```
isaaclab-hiveboard/
├── dependencies/
│   └── HiveBoard/               # Git submodule (URDF/USD models & meshes)
├── scripts/                     # Standalone CLI tools (play.py, collect_demos.py, etc.)
└── source/
    └── isaaclab_hiveboard/
        ├── config/
        │   └── extension.toml   # Omniverse extension configuration
        ├── setup.py
        └── isaaclab_hiveboard/
            ├── assets/          # Dynamic HiveBoard & robot asset resolvers
            ├── mdp/             # Custom actions, commands, events, observations
            ├── tasks/           # Robot tasks (spot/, franka/, anymal/)
            └── utils/           # Diagnostics & metrics
```
