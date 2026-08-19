# Isaac Lab - HiveBoard Multi-Robot Manipulation Suite

[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-blue.svg)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green.svg)](LICENSE)

An Isaac Lab extension package for simulating, controlling, and benchmarking various robot platforms (**Boston Dynamics Spot with Arm**, **Franka Emika Panda**, and **ANYmal C/D**) performing manipulation tasks on the **[HiveBoard Benchmark](https://github.com/EESC-LabRoM/HiveBoard)**.

---

## 🚀 Features

- **Multi-Robot Support**:
  - 🐕 **Boston Dynamics Spot with Arm**: Differential IK, CuRobo collision avoidance, and RMPFlow control for gate valves, lever valves, and circuit breakers.
  - 🦾 **Franka Emika Panda**: Precision end-effector tracking with orientation alignments and real-time pose diagnostics.
  - 🐾 **ANYmal C/D**: Manipulation & locomotion co-simulation on HiveBoard panels.
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

Install editable package using `uv`:

```bash
uv sync
```

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
| `Isaac-HiveBoard-Anymal-BallValve-v0` | ANYmal C | Ball (Lever) Valve | Locomanipulation baseline |

List all available tasks:

```bash
uv run python scripts/list_envs.py
```

---

## 🎮 Running Simulations

### Interactive Play

Play Spot ball valve with camera orbit:

```bash
uv run python scripts/play.py --task "Isaac-HiveBoard-Spot-BallValve-v0" --orbit
```

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