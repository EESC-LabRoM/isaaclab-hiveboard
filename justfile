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

# Drag GoTo goals, rotation references and the TCP offset in a browser
edit-commands *args:
    uv run python scripts/command_edit.py {{args}}

# CPU checks for command setup files, frame transforms and offset IK
check-command-setup:
    uv run python scripts/check_command_setup.py

# Build Spot ball-valve EE paths: Newton beads + terminal goal editor
build-valve-path *args:
    uv run python scripts/valve_path_tui.py {{args}} physics=newton_mjwarp --visualizer newton

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

# Play Franka Ball Valve with pose diagnostics
play-franka-ball-valve:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Franka-BallValve-Play-v0" --pose-debug

# Play Franka Circuit Breaker with pose diagnostics
play-franka-breaker:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Franka-CircuitBreaker-Play-v0" --pose-debug

# Play Franka High-Torque Valve with pose diagnostics
play-franka-high-torque:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Franka-HighTorqueValve-Play-v0" --pose-debug

# Play Franka Small Valve with pose diagnostics
play-franka-small-valve:
    uv run python scripts/play.py --task "Isaac-HiveBoard-Franka-SmallValve-Play-v0" --pose-debug

# Play any robot/tool Play task, e.g. `just play Franka M30Thread` (Spot|Anymal|Franka x tool)
play robot tool *args:
    uv run python scripts/play.py --task "Isaac-HiveBoard-{{robot}}-{{tool}}-Play-v0" --pose-debug {{args}}

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

# Record every registered HiveBoard environment at its simulation-time FPS (RTX path-traced)
record-all *args:
    uv run python scripts/record_all_envs.py {{args}}

# Collect scripted-expert demos for imitation learning (robomimic layout)
il-collect num_demos="50" num_envs="8" *args:
    uv run python scripts/imitation/collect_demos.py --num_demos {{num_demos}} --num_envs {{num_envs}} {{args}}

# Behaviour-clone an MLP policy from a collected dataset
il-train dataset *args:
    uv run python scripts/imitation/train_bc.py --dataset {{dataset}} {{args}}

# Improve a behaviour-cloned policy with DAgger rounds
il-dagger dataset rounds="5" *args:
    uv run python scripts/imitation/dagger.py --initial_dataset {{dataset}} --rounds {{rounds}} {{args}}

# Measure a checkpoint's success rate (pass --expert for the scripted ceiling)
il-eval *args:
    uv run python scripts/imitation/eval_policy.py {{args}}

# Part of the rl-* recipes: BallValve (default), SmallValve or M30Thread, e.g. `RL_TOOL=SmallValve just rl-teacher`
rl_tool := env("RL_TOOL", "BallValve")

# Build the cuRobo expert bank the RL task resets from and tracks
rl-bank num_envs="512" num_trajectories="5000" *args:
    uv run python scripts/rl/build_expert_bank.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-v0 \
        --num_envs {{num_envs}} --num_trajectories {{num_trajectories}} {{args}}

# RL teacher: PPO on privileged state (ANYmal ball valve by default)
rl-teacher num_envs="4096" *args:
    uv run python scripts/rl/train.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-v0 --num_envs {{num_envs}} {{args}}

# RL student: distil the teacher checkpoint onto proprioception + registered valve pose
rl-student teacher num_envs="4096" *args:
    uv run python scripts/rl/train.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-v0 --num_envs {{num_envs}} \
        --agent rsl_rl_distillation_cfg_entry_point --checkpoint {{teacher}} \
        env.terminations.expert_drift=null env.terminations.expert_valve_lag=null {{args}}

# RL student trained directly with PPO: deployable actor, privileged critic
rl-student-ppo num_envs="4096" *args:
    uv run python scripts/rl/train.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-v0 --num_envs {{num_envs}} \
        --agent rsl_rl_student_ppo_cfg_entry_point {{args}}

# Replay expert-bank trajectories in simulation, shown in Viser (http://localhost:9080)
rl-bank-replay *args:
    uv run python scripts/rl/replay_expert_bank.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-Play-v0 {{args}}

# Success rate / reliability / stage metrics of a teacher or student checkpoint
rl-eval checkpoint *args:
    uv run python scripts/rl/evaluate.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-Play-v0 --checkpoint {{checkpoint}} {{args}}

# Watch a checkpoint in the Newton viewer and export it (TorchScript + ONNX)
rl-play checkpoint *args:
    uv run python scripts/rl/play.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-Play-v0 --checkpoint {{checkpoint}} --viz newton {{args}}

# Record one 12 s episode of a checkpoint to <run>/videos/play/ (student: --agent rsl_rl_distillation_cfg_entry_point)
rl-video checkpoint *args:
    uv run python scripts/rl/play.py --task Isaac-HiveBoard-Anymal-{{rl_tool}}-RL-Play-v0 --checkpoint {{checkpoint}} \
        --num_envs 1 --video --video_length 240 --viz newton {{args}}
