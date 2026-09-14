# Justfile for Isaac Lab HiveBoard Multi-Robot Manipulation

default:
    @just --list

# Hold a known TCP position with only Spot in the scene
validate-command-spot *args:
    uv run python scripts/play.py --task validate_command_spot --pose-debug {{args}}

# List all available registered HiveBoard environments
list-envs:
    uv run python scripts/list_envs.py

# Play Spot Ball Valve task
play-spot-ball-valve:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Spot-BallValve-v0"

# Play website Spot bench-valve joint clip
play-spot-bench-valve:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Spot-BenchValve-Play-v0"

# Play Spot bench valve with each Cartesian leg planned by cuRobo
play-spot-curobo-valve:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Spot-CuroboValve-Play-v0" --pose-debug

# Play ANYmal bench valve with each Cartesian leg planned by cuRobo (DynaArm)
play-anymal-curobo-valve:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Anymal-CuroboValve-Play-v0" --pose-debug

# Play robot-only Spot clip (PD gain eval)
play-spot-gains:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Spot-Gains-Play-v0"

# Rebuild bench Spot q from traj_edit beads using this robot's Jacobian IK
retarget-spot-traj *args:
    uv run python scripts/retarget_spot_traj.py {{args}} physics=newton_mjwarp --visualizer none

# Edit bench Spot beads in Newton ViewerGL and re-solve Isaac Lab IK
edit-spot-traj *args:
    uv run python scripts/traj_edit.py {{args}} physics=newton_mjwarp --visualizer none

# Search Spot arm/gripper PD gains on the robot-only clip
optimize-spot-gains joints="all" num_envs="16" max_evals="80":
    uv run python scripts/optimize_spot_gains.py --optimize \
        --joints {{joints}} --num-envs {{num_envs}} --max-evals {{max_evals}} \
        physics=newton_mjwarp --visualizer none

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

# Drop unused OBJ vertices (and CAD polylines) so urdf-usd-converter can import Spot meshes
strip-obj-unused-verts:
    uv run python scripts/generate_newton_usd.py --strip-obj

# Regenerate Newton USD assets (UUC conversion + valve CoACD overlay)
generate-newton-usd *args:
    uv run python scripts/generate_newton_usd.py {{args}}

# Bake kitless ANYmal-D + DynaArm + Robotiq USD (fetch Nucleus, author arm, weld)
generate-anymal-usd *args:
    uv run python scripts/generate_anymal_newton_usd.py {{args}}

# Precompute Spot reachable reset state cache
precompute-cache:
    uv run python scripts/precompute_reset_states.py --headless --device cuda:0 --output_path logs/spot_reset_states.pt

# Collect Spot demonstration dataset
collect-demos num_demos="10":
    uv run python scripts/collect_demos.py --headless --device cuda:0 --num_demos {{num_demos}}

# Record a validation video for every HiveBoard example
record-all *args:
    ./scripts/record_all_examples.sh {{args}}
