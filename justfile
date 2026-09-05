# Justfile for Isaac Lab HiveBoard Multi-Robot Manipulation

default:
    @just --list

# List all available registered HiveBoard environments
list-envs:
    uv run python scripts/list_envs.py

# Play Spot Ball Valve task
play-spot-ball-valve:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Spot-BallValve-v0"

# Play Spot hidden-button task
play-spot-button:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Spot-Button-v0"

# Play Spot Circuit Breaker task
play-spot-breaker:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Spot-CircuitBreaker-v0"

# Play Spot High-Torque Valve task
play-spot-high-torque:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Spot-HighTorqueValve-v0"

# Play Spot Small Valve task
play-spot-small-valve:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Spot-SmallValve-v0"

# Play Franka Lever Valve with pose diagnostics
play-franka-lever:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Franka-LeverValve-v0" --pose-debug

# Play Franka Circuit Breaker with pose diagnostics
play-franka-breaker:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Franka-CircuitBreaker-v0" --pose-debug

# Inspect ANYmal-D + DynaArm + Robotiq 2F-140 TCP axes
play-anymal-only:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Anymal-OnlyRobot-v0"

# Sweep the standalone 2F-140 and print mimic-joint signs
play-anymal-gripper:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Anymal-OnlyGripper-v0"

# Play ANYmal-D opening the HiveBoard ball valve
play-anymal-ball-valve:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Anymal-BallValve-v0"

# Precompute Spot reachable reset state cache
precompute-cache:
    uv run python scripts/precompute_reset_states.py --headless --device cuda:0 --output_path logs/spot_reset_states.pt

# Collect Spot demonstration dataset
collect-demos num_demos="10":
    uv run python scripts/collect_demos.py --headless --device cuda:0 --num_demos {{num_demos}}

# Record a validation video for every HiveBoard example
record-all *args:
    ./scripts/record_all_examples.sh {{args}}
