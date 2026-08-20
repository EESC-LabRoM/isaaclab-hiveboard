from collections.abc import Sequence
import hashlib
import json
from pathlib import Path

import isaaclab.utils.math as math_utils
import numpy as np
import pytorch_kinematics as pk
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from tqdm import tqdm

from isaaclab_hiveboard.assets import ASSET_DIR

HAS_CUROBO = False
CUROBO_IMPORT_ERROR: ImportError | None = None
try:
    from isaaclab_hiveboard.mdp.curobo_warp import (
        curobo_compatible_warp,
    )

    with curobo_compatible_warp():
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

    from isaaclab_hiveboard.mdp.curobo_robot_cfg import (
        load_spot_robot_cfg,
    )

    HAS_CUROBO = True
except ImportError as err:
    CUROBO_IMPORT_ERROR = err
    InverseKinematics = None
    InverseKinematicsCfg = None
    DeviceCfg = None
    GoalToolPose = None
    JointState = None
    Pose = None
    load_spot_robot_cfg = None


def _assert_get_item(obj, key_name, default: int) -> int:
    if key_name not in obj:
        return default
    if not isinstance(obj[key_name], int):
        raise TypeError(f"Key '{key_name}' should be an integer.")
    return obj[key_name]


def _assert_positive_float(obj, key_name, default: float) -> float:
    if key_name not in obj:
        return default
    if not isinstance(obj[key_name], (float, int)) or obj[key_name] < 0.0:
        raise TypeError(f"Key '{key_name}' should be a positive float.")
    return float(obj[key_name])


def canonicalize_ee_orientation_upward(quat_b: torch.Tensor) -> torch.Tensor:
    """Ensure the TCP local +Z (up) vector has a non-negative projection on base +Z.

    If local +Z points downwards in the base frame (z < 0), rotate 180° around
    the TCP local +X axis (approach axis) to keep the gripper right-side up.
    """
    is_1d = quat_b.ndim == 1
    q = quat_b.unsqueeze(0) if is_1d else quat_b

    z_local = torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], 3)
    z_in_base = math_utils.quat_apply(q, z_local)
    inverted = z_in_base[:, 2] < 0.0

    if not torch.any(inverted):
        return quat_b

    flip_x = torch.tensor([0.0, 1.0, 0.0, 0.0], device=q.device).expand(q.shape[0], 4)
    flipped_quat = math_utils.quat_mul(q, flip_x)
    result = torch.where(inverted.unsqueeze(-1), flipped_quat, q)
    return result.squeeze(0) if is_1d else result


def get_pose_grid(
    n_x: int,
    n_y: int,
    n_z: int,
    n_roll: int,
    n_pitch: int,
    n_yaw: int,
    max_x: float,
    max_y: float,
    max_z: float,
    max_roll: float,
    max_pitch: float,
    max_yaw: float,
):
    """Return Cartesian offsets and world-frame roll-pitch-yaw rotations.

    The rotation order is ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``. The returned
    rotations are pre-multiplied with each nominal pose, so all three sampled
    angles perturb the valve in the fixed robot-base frame.
    """
    x = np.linspace(-max_x, max_x, n_x)
    y = np.linspace(-max_y, max_y, n_y)
    z = np.linspace(-max_z, max_z, n_z)
    roll = np.linspace(-max_roll, max_roll, n_roll)
    pitch = np.linspace(-max_pitch, max_pitch, n_pitch)
    yaw = np.linspace(-max_yaw, max_yaw, n_yaw)
    x, y, z, roll, pitch, yaw = np.meshgrid(x, y, z, roll, pitch, yaw, indexing="ij")

    n_items = n_x * n_y * n_z * n_roll * n_pitch * n_yaw
    position_arr = np.zeros((n_items, 3))
    position_arr[:, 0] = x.flatten()
    position_arr[:, 1] = y.flatten()
    position_arr[:, 2] = z.flatten()

    cos_roll, sin_roll = np.cos(roll.flatten()), np.sin(roll.flatten())
    cos_pitch, sin_pitch = np.cos(pitch.flatten()), np.sin(pitch.flatten())
    cos_yaw, sin_yaw = np.cos(yaw.flatten()), np.sin(yaw.flatten())

    rotations = np.empty((n_items, 3, 3))
    rotations[:, 0, 0] = cos_yaw * cos_pitch
    rotations[:, 0, 1] = cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll
    rotations[:, 0, 2] = cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll
    rotations[:, 1, 0] = sin_yaw * cos_pitch
    rotations[:, 1, 1] = sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll
    rotations[:, 1, 2] = sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll
    rotations[:, 2, 0] = -sin_pitch
    rotations[:, 2, 1] = cos_pitch * sin_roll
    rotations[:, 2, 2] = cos_pitch * cos_roll

    return position_arr, rotations


def _squeeze_seed_dim(js: JointState) -> JointState:
    """Drop the IK return-seed axis so position is ``[batch, dof]``."""
    position = js.position
    if position.ndim == 3:
        position = position.squeeze(1)
    return JointState.from_position(position, joint_names=js.joint_names)


def _as_offset_cfg(offset) -> OffsetCfg:
    """Copy ``pos``/``rot`` into Isaac Lab's frame-transformer ``OffsetCfg``."""
    return OffsetCfg(pos=tuple(offset.pos), rot=tuple(offset.rot))


def _resolve_ee_offset(cfg: EventTermCfg, env: ManagerBasedEnv) -> OffsetCfg:
    """TCP offset from ``ee_offset``, else the pose command's ``body_offset``."""
    ee_offset = cfg.params.get("ee_offset")
    if ee_offset is not None:
        return _as_offset_cfg(ee_offset)
    command_name = cfg.params.get("command_name", "pose_command")
    commands_cfg = getattr(env.cfg, "commands", None)
    command_cfg = (
        getattr(commands_cfg, command_name, None)
        if commands_cfg is not None
        else None
    )
    body_offset = getattr(command_cfg, "body_offset", None)
    if body_offset is None:
        raise ValueError(
            "RandomizeValveHandlePoseEvent needs ee_offset, or a command "
            f"'{command_name}' with body_offset (the same TCP offset used "
            "by SequentialPoseCommand)."
        )
    return _as_offset_cfg(body_offset)


def _resolve_valve_offset(cfg: EventTermCfg, env: ManagerBasedEnv) -> OffsetCfg:
    """Spawn offset from ``valve_offset``, else a named ``FrameCfg``."""
    target_frame_name = cfg.params.get("target_frame_name")
    if target_frame_name:
        frame_name = cfg.params.get("frame_name", "target_frame")
        try:
            transformer = env.scene[frame_name]
        except KeyError as err:
            raise ValueError(
                f"Frame transformer '{frame_name}' is not in the scene."
            ) from err
        for frame in transformer.cfg.target_frames:
            if frame.name == target_frame_name:
                return _as_offset_cfg(frame.offset)
        available = [frame.name for frame in transformer.cfg.target_frames]
        raise ValueError(
            f"FrameCfg '{target_frame_name}' not found on '{frame_name}'. "
            f"Available: {available}."
        )
    valve_offset = cfg.params.get("valve_offset")
    if valve_offset is not None:
        return _as_offset_cfg(valve_offset)
    raise ValueError(
        "RandomizeValveHandlePoseEvent needs target_frame_name (a "
        "FrameTransformerCfg.FrameCfg name) or valve_offset."
    )


class RandomizeValveHandlePoseEvent(ManagerTermBase):
    """Reset the arm so the TCP sits on a named valve frame.

    ``ee_offset`` defaults to the pose command's ``body_offset``. The spawn
    pose defaults to ``FrameTransformerCfg.FrameCfg`` named
    ``target_frame_name`` on the scene transformer ``frame_name``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._valve_cfg: SceneEntityCfg = cfg.params["valve_cfg"]
        self._valve_offset: OffsetCfg = _resolve_valve_offset(cfg, env)
        self._ee_offset: OffsetCfg = _resolve_ee_offset(cfg, env)
        self._valve: Articulation = env.scene[self._valve_cfg.name]
        self._valve_urdf: str = cfg.params.get(
            "valve_urdf", f"{ASSET_DIR}/ball_valve/ball_valve.urdf"
        )
        self._valve_root_link: str = cfg.params.get("valve_root_link", "body")
        self._valve_ee_link: str = cfg.params.get("valve_ee_link", "lever_pivot")
        self._valve_root_pose: OffsetCfg | None = cfg.params.get("valve_root_pose")

        assert self._valve_cfg.body_names is not None, (
            "Body names must be specified in the valve_cfg for RandomizeValveHandlePoseEvent."
        )

        valve_body_idxs, _ = self._valve.find_bodies(
            name_keys=self._valve_cfg.body_names
        )
        assert len(valve_body_idxs) == 1, "Expected exactly one valve body to be found."
        self.valve_body_idx = valve_body_idxs[0]

        if not isinstance(self._asset, Articulation):
            raise ValueError(
                f"Randomization term 'RandomizeValveHandlePoseEvent' not supported for asset: '{asset_cfg.name}'"
                f" with type: '{type(self._asset)}'."
            )

        assert asset_cfg.joint_names is not None, (
            "Joint names must be specified in the asset_cfg for RandomizeValveHandlePoseEvent."
        )
        self._joint_ids, joint_names = self._asset.find_joints(
            name_keys=asset_cfg.joint_names,
            preserve_order=asset_cfg.preserve_order,
        )
        self._joint_names = joint_names
        ee_body_ids, _ = self._asset.find_bodies("arm_link_wr1")
        if not ee_body_ids:
            raise ValueError("Body 'arm_link_wr1' not found on the robot.")
        self._ee_body_idx = ee_body_ids[0]
        self._ee_offset_pos = torch.tensor(
            self._ee_offset.pos, device=env.device, dtype=torch.float32
        )
        self._ee_offset_quat = torch.tensor(
            self._ee_offset.rot, device=env.device, dtype=torch.float32
        )

        self._robot_joint_states = None
        self._num_valid_states = 0
        self._num_valid_valve_states = 0

        if self._valve_root_pose is not None:
            self._setup_ik_solver(env)
            self._on_the_fly = bool(cfg.params.get("on_the_fly", False)) or (
                self._get_reset_state_cache_path() is None
            )
            if self._on_the_fly:
                print(
                    "[INFO] RandomizeValveHandlePoseEvent configured for on-the-fly "
                    "reset IK generation."
                )
                return

            cache_path = self._get_reset_state_cache_path()
            force_regenerate = bool(
                cfg.params.get("reset_state_cache_force_regenerate", False)
            )
            print("Cache: ", cache_path, " | Force regenerate: ", force_regenerate)
            if cache_path is not None and cache_path.is_file() and not force_regenerate:
                print("Try to load cache")
                self._load_valve_first_states(cache_path, env)
            else:
                print("Regenerate")
                self._generate_valve_first_states(env)
                if cache_path is not None:
                    self._save_valve_first_states(cache_path)
            return

        if HAS_CUROBO:
            try:
                self._robot_joint_states = self._generate_valid_ee_poses(
                    env, joint_names
                )
                self._num_valid_states = self._robot_joint_states.shape[0]
                valve_joint_range = cfg.params.get("valve_joint_range", (0.0, 0.0001))
                n_valve_states = cfg.params.get("n_valve_states", 2)
                if (
                    isinstance(valve_joint_range, (tuple, list))
                    and len(valve_joint_range) >= 2
                ):
                    min_j, max_j = (
                        float(valve_joint_range[0]),
                        float(valve_joint_range[1]),
                    )
                    n_steps = (
                        int(valve_joint_range[2])
                        if len(valve_joint_range) > 2
                        else n_valve_states
                    )
                    self._valve_joint_states = torch.linspace(
                        min_j, max_j, n_steps, device=env.device
                    )
                else:
                    self._valve_joint_states = torch.linspace(
                        0, 0.0001, 2, device=env.device
                    )
                align_rotation = cfg.params.get("align_rotation", True)
                (
                    self._valve_inv_pos,
                    self._valve_inv_quat,
                ) = self._generate_valid_valve_poses(
                    env,
                    urdf_name=self._valve_urdf,
                    offset=self._valve_offset,
                    th=self._valve_joint_states,
                    root_link_name=self._valve_root_link,
                    end_link_name=self._valve_ee_link,
                    align_rotation=align_rotation,
                )
                self._num_valid_valve_states = self._valve_joint_states.shape[0]

                print("There are ", self._num_valid_states, "valid end-effector poses.")
                print(
                    "There are ",
                    self._num_valid_valve_states,
                    "valid valve handle poses.",
                )
            except Exception as err:
                print(
                    "[WARN] Failed to precompute cuRobo valve reset poses "
                    f"({err}). Falling back to default joint poses."
                )
                self._robot_joint_states = None
                self._num_valid_states = 0
                self._num_valid_valve_states = 0
        else:
            print(
                "[INFO] CuRobo is not available. RandomizeValveHandlePoseEvent will use default reset poses. "
                f"Import error: {CUROBO_IMPORT_ERROR}"
            )

    def _get_valve_joint_states(self, env: ManagerBasedEnv) -> torch.Tensor:
        """Create the configured scalar valve-joint samples."""
        valve_joint_range = self.cfg.params.get("valve_joint_range", (0.0, 0.0001))
        n_valve_states = self.cfg.params.get("n_valve_states", 2)
        if isinstance(valve_joint_range, (tuple, list)) and len(valve_joint_range) >= 2:
            min_j, max_j = float(valve_joint_range[0]), float(valve_joint_range[1])
            n_steps = (
                int(valve_joint_range[2])
                if len(valve_joint_range) > 2
                else int(n_valve_states)
            )
        else:
            min_j, max_j, n_steps = 0.0, 0.0001, 2
        if n_steps < 1:
            raise ValueError("n_valve_states must be greater than zero")
        return torch.linspace(min_j, max_j, n_steps, device=env.device)

    def _get_valve_tcp_transforms(
        self, env: ManagerBasedEnv, valve_joint_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return valve-root-to-approach-TCP transforms for each joint sample."""
        chain = pk.build_serial_chain_from_urdf(
            open(self._valve_urdf, mode="rb").read(),
            end_link_name=self._valve_ee_link,
            root_link_name=self._valve_root_link,
        )
        th_cpu = valve_joint_states.detach().to(device="cpu")
        if th_cpu.ndim == 1:
            th_cpu = th_cpu[:, None]
        n_dof = getattr(chain, "n_joints", th_cpu.shape[-1])
        if th_cpu.shape[-1] < n_dof:
            th_cpu = torch.cat(
                [th_cpu, torch.zeros(th_cpu.shape[0], n_dof - th_cpu.shape[-1])],
                dim=-1,
            )
        root_to_handle = chain.forward_kinematics(th_cpu, end_only=True)
        handle_to_tcp = pk.Transform3d(
            pos=self._valve_offset.pos, rot=self._valve_offset.rot
        )
        root_to_tcp = root_to_handle.compose(handle_to_tcp)
        pos, rot = math_utils.unmake_pose(root_to_tcp.get_matrix().to(env.device))
        return pos, math_utils.quat_from_matrix(rot)

    def _get_reset_state_cache_path(self) -> Path | None:
        """Return the optional valve-first reset cache location."""
        value = self.cfg.params.get("reset_state_cache_path")
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise TypeError("reset_state_cache_path must be a non-empty string")
        return Path(value).expanduser().resolve()

    def _reset_state_cache_metadata(self) -> dict[str, int | str]:
        """Create an identity for cache inputs that affect reset validity."""
        robot_urdf = self.cfg.params.get(
            "robot_urdf", f"{ASSET_DIR}/spot/spot_with_arm.urdf"
        )
        cache_params = {
            key: self.cfg.params.get(key)
            for key in (
                "n_x",
                "n_y",
                "n_z",
                "n_roll",
                "n_pitch",
                "n_yaw",
                "max_x",
                "max_y",
                "max_z",
                "max_roll",
                "max_pitch",
                "max_yaw",
                "valve_joint_range",
                "n_valve_states",
                "num_ik_retries",
                "ik_position_tolerance",
                "ik_rotation_tolerance",
                "ik_max_iterations",
                "robot_root_link",
                "robot_ee_link",
                "valve_root_link",
                "valve_ee_link",
            )
        }
        cache_inputs = {
            "cache_params": cache_params,
            "robot_arm_joint_names": self._joint_names,
            "robot_urdf_sha256": hashlib.sha256(
                Path(robot_urdf).read_bytes()
            ).hexdigest(),
            "valve_root_pose": {
                "pos": self._valve_root_pose.pos,
                "rot": self._valve_root_pose.rot,
            },
            "valve_urdf_sha256": hashlib.sha256(
                Path(self._valve_urdf).read_bytes()
            ).hexdigest(),
            "valve_offset": {
                "pos": self._valve_offset.pos,
                "rot": self._valve_offset.rot,
            },
            "ee_offset": {"pos": self._ee_offset.pos, "rot": self._ee_offset.rot},
        }
        fingerprint = hashlib.sha256(
            json.dumps(cache_inputs, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {"schema_version": 1, "fingerprint": fingerprint}

    def _load_valve_first_states(self, cache_path: Path, env: ManagerBasedEnv) -> None:
        """Load cached reachable valve/arm reset pairs after validating metadata."""
        print("Loading valve")
        try:
            payload = torch.load(cache_path, map_location=env.device, weights_only=True)
        except (OSError, RuntimeError) as error:
            raise RuntimeError(
                f"Could not load reset-state cache: {cache_path}"
            ) from error
        if (
            not isinstance(payload, dict)
            or payload.get("metadata") != self._reset_state_cache_metadata()
        ):
            raise ValueError(
                f"Reset-state cache is stale or incompatible: {cache_path}. "
                "Regenerate it with scripts/precompute_spot_reset_states.py."
            )
        state_keys = (
            "robot_joint_states",
            "valve_root_pos_b",
            "valve_root_quat_b",
            "paired_valve_joint_states",
        )
        if any(not isinstance(payload.get(key), torch.Tensor) for key in state_keys):
            raise ValueError(f"Reset-state cache has invalid tensor data: {cache_path}")
        robot_joint_states = payload["robot_joint_states"]
        valve_root_pos_b = payload["valve_root_pos_b"]
        valve_root_quat_b = payload["valve_root_quat_b"]
        paired_valve_joint_states = payload["paired_valve_joint_states"]
        num_states = robot_joint_states.shape[0]
        if (
            num_states == 0
            or robot_joint_states.ndim != 2
            or robot_joint_states.shape[1] != len(self._joint_ids)
            or valve_root_pos_b.shape != (num_states, 3)
            or valve_root_quat_b.shape != (num_states, 4)
            or paired_valve_joint_states.shape != (num_states, 1)
        ):
            raise ValueError(
                f"Reset-state cache has incompatible tensor shapes: {cache_path}"
            )
        self._robot_joint_states = robot_joint_states
        self._valve_root_pos_b = valve_root_pos_b
        self._valve_root_quat_b = valve_root_quat_b
        self._paired_valve_joint_states = paired_valve_joint_states[:, 0]
        self._num_valid_states = num_states
        self._num_valid_valve_states = num_states
        print(
            f"[INFO] Loaded {num_states} reachable valve-first reset states "
            f"from {cache_path}."
        )

    def _save_valve_first_states(self, cache_path: Path) -> None:
        """Persist the generated reset pairs with metadata for compatibility checks."""
        if self._robot_joint_states is None:
            raise RuntimeError("Cannot save reset states before they are generated")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": self._reset_state_cache_metadata(),
            "robot_joint_states": self._robot_joint_states.detach().cpu(),
            "valve_root_pos_b": self._valve_root_pos_b.detach().cpu(),
            "valve_root_quat_b": self._valve_root_quat_b.detach().cpu(),
            "paired_valve_joint_states": self._paired_valve_joint_states.detach()
            .cpu()
            .reshape(-1, 1),
        }
        temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(cache_path)
        print(
            f"[INFO] Saved {self._num_valid_states} reachable valve-first "
            f"reset states to {cache_path}."
        )

    def _setup_ik_solver(self, env: ManagerBasedEnv) -> None:
        """Initialize the kinematic chain and numerical IK solver for Spot's arm."""
        robot_urdf = self.cfg.params.get(
            "robot_urdf", f"{ASSET_DIR}/spot/spot_with_arm.urdf"
        )
        robot_root_link = self.cfg.params.get("robot_root_link", "body")
        robot_ee_link = self.cfg.params.get("robot_ee_link", "arm_link_wr1")
        self._arm_chain = pk.build_serial_chain_from_urdf(
            open(robot_urdf, mode="rb").read(),
            end_link_name=robot_ee_link,
            root_link_name=robot_root_link,
        ).to(dtype=torch.float32, device="cpu")
        self._ik_joint_names = self._arm_chain.get_joint_parameter_names()
        self._ik_joint_ids, resolved_ik_names = self._asset.find_joints(
            self._ik_joint_names, preserve_order=True
        )
        if resolved_ik_names != self._ik_joint_names:
            raise ValueError(
                "Spot IK joint order does not match the URDF chain: "
                f"expected {self._ik_joint_names}, received {resolved_ik_names}."
            )

        num_retries = int(self.cfg.params.get("num_ik_retries", 32))
        if num_retries < 1:
            raise ValueError("num_ik_retries must be greater than zero")
        joint_limits = torch.stack((self._arm_chain.low, self._arm_chain.high), dim=-1)
        if not torch.all(torch.isfinite(joint_limits)):
            raise ValueError("Spot arm URDF must provide finite IK joint limits")
        retry_configs = joint_limits[:, 0] + torch.rand(
            num_retries,
            len(self._ik_joint_names),
            device="cpu",
        ) * (joint_limits[:, 1] - joint_limits[:, 0])
        retry_configs[0] = self._asset.data.default_joint_pos[0, self._ik_joint_ids].cpu()

        self._ik_solver = pk.PseudoInverseIK(
            self._arm_chain,
            pos_tolerance=float(self.cfg.params.get("ik_position_tolerance", 0.005)),
            rot_tolerance=float(self.cfg.params.get("ik_rotation_tolerance", 0.05)),
            retry_configs=retry_configs,
            joint_limits=joint_limits,
            max_iterations=int(self.cfg.params.get("ik_max_iterations", 100)),
            lr=0.5,
            regularlization=1.0e-4,
            lm_damping=0.1,
            early_stopping_any_converged=True,
        )

        self._valve_chain = pk.build_serial_chain_from_urdf(
            open(self._valve_urdf, mode="rb").read(),
            end_link_name=self._valve_ee_link,
            root_link_name=self._valve_root_link,
        )

    def _generate_valve_first_states(self, env: ManagerBasedEnv) -> None:
        """Sample valve poses first and retain the pairs reachable by Spot's arm."""
        valve_joint_states = self._get_valve_joint_states(env)
        tcp_pos_v, tcp_quat_v = self._get_valve_tcp_transforms(env, valve_joint_states)

        position_offsets, orientation_matrices = get_pose_grid(
            n_x=_assert_get_item(self.cfg.params, "n_x", 1),
            n_y=_assert_get_item(self.cfg.params, "n_y", 1),
            n_z=_assert_get_item(self.cfg.params, "n_z", 1),
            n_roll=_assert_get_item(self.cfg.params, "n_roll", 1),
            n_pitch=_assert_get_item(self.cfg.params, "n_pitch", 1),
            n_yaw=_assert_get_item(self.cfg.params, "n_yaw", 1),
            max_x=_assert_positive_float(self.cfg.params, "max_x", 0.0),
            max_y=_assert_positive_float(self.cfg.params, "max_y", 0.0),
            max_z=_assert_positive_float(self.cfg.params, "max_z", 0.0),
            max_roll=_assert_positive_float(self.cfg.params, "max_roll", 0.0),
            max_pitch=_assert_positive_float(self.cfg.params, "max_pitch", 0.0),
            max_yaw=_assert_positive_float(self.cfg.params, "max_yaw", 0.0),
        )
        root_pos_b = torch.tensor(
            self._valve_root_pose.pos, device=env.device, dtype=torch.float32
        ).repeat(len(position_offsets), 1)
        root_pos_b += torch.tensor(
            position_offsets, device=env.device, dtype=torch.float32
        )
        nominal_root_quat = torch.tensor(
            self._valve_root_pose.rot, device=env.device, dtype=torch.float32
        ).repeat(len(position_offsets), 1)
        orientation_quat = math_utils.quat_from_matrix(
            torch.tensor(orientation_matrices, device=env.device, dtype=torch.float32)
        )
        root_quat_b = math_utils.quat_mul(orientation_quat, nominal_root_quat)

        num_valve_states = valve_joint_states.shape[0]
        root_pos_b = root_pos_b.repeat_interleave(num_valve_states, dim=0)
        root_quat_b = root_quat_b.repeat_interleave(num_valve_states, dim=0)
        valve_state_ids = torch.arange(num_valve_states, device=env.device).repeat(
            len(position_offsets)
        )
        candidate_valve_q = valve_joint_states[valve_state_ids]

        tcp_pos_b, tcp_quat_b = math_utils.combine_frame_transforms(
            root_pos_b,
            root_quat_b,
            tcp_pos_v[valve_state_ids],
            tcp_quat_v[valve_state_ids],
        )
        tcp_quat_b = canonicalize_ee_orientation_upward(tcp_quat_b)

        # The robot IK chain ends at arm_link_wr1, while the target above is
        # the offset TCP. Convert TCP targets back to flange targets for IK.
        ee_offset_pos = self._ee_offset_pos.repeat(tcp_pos_b.shape[0], 1)
        ee_offset_quat = self._ee_offset_quat.repeat(tcp_pos_b.shape[0], 1)
        tcp_to_flange_quat = math_utils.quat_inv(ee_offset_quat)
        tcp_to_flange_pos = math_utils.quat_apply(tcp_to_flange_quat, -ee_offset_pos)
        flange_pos_b, flange_quat_b = math_utils.combine_frame_transforms(
            tcp_pos_b,
            tcp_quat_b,
            tcp_to_flange_pos,
            tcp_to_flange_quat,
        )

        if not hasattr(self, "_ik_solver"):
            self._setup_ik_solver(env)

        ik_batch_size = int(self.cfg.params.get("ik_batch_size", 2048))
        if ik_batch_size < 1:
            raise ValueError("ik_batch_size must be greater than zero")
        show_progress = bool(self.cfg.params.get("show_reset_state_progress", False))
        valid_problem_ids_chunks: list[torch.Tensor] = []
        ik_solution_chunks: list[torch.Tensor] = []
        candidate_count = flange_pos_b.shape[0]
        batch_starts = range(0, candidate_count, ik_batch_size)
        for start in tqdm(
            batch_starts,
            total=(candidate_count + ik_batch_size - 1) // ik_batch_size,
            desc="Filtering reachable Spot valve resets",
            unit="batch",
            disable=not show_progress,
        ):
            end = min(start + ik_batch_size, candidate_count)
            ik_targets = pk.Transform3d(
                pos=flange_pos_b[start:end].cpu(),
                rot=flange_quat_b[start:end].cpu(),
            )
            ik_result = self._ik_solver.solve(ik_targets)
            valid = ik_result.converged_any
            if not torch.any(valid):
                continue
            valid_problem_ids_chunks.append(torch.where(valid)[0] + start)
            retry_ids = ik_result.converged[valid].to(torch.int64).argmax(dim=1)
            ik_solution_chunks.append(ik_result.solutions[valid, retry_ids])

        if not valid_problem_ids_chunks:
            raise RuntimeError(
                "None of the sampled valve poses are reachable by Spot's arm."
            )
        valid_problem_ids = torch.cat(valid_problem_ids_chunks)
        ik_solutions = torch.cat(ik_solution_chunks)

        full_joint_states = self._asset.data.default_joint_pos[
            0, self._joint_ids
        ].repeat(len(valid_problem_ids), 1)
        joint_column_by_name = {
            name: index for index, name in enumerate(self._joint_names)
        }
        ik_columns = [joint_column_by_name[name] for name in self._ik_joint_names]
        full_joint_states[:, ik_columns] = ik_solutions.to(env.device)

        self._robot_joint_states = full_joint_states
        valid_problem_ids_device = valid_problem_ids.to(env.device)
        self._valve_root_pos_b = root_pos_b[valid_problem_ids_device]
        self._valve_root_quat_b = root_quat_b[valid_problem_ids_device]
        self._paired_valve_joint_states = candidate_valve_q[valid_problem_ids_device]
        self._num_valid_states = len(valid_problem_ids)
        self._num_valid_valve_states = self._num_valid_states
        print(
            f"[INFO] Generated {self._num_valid_states} reachable valve-first "
            f"reset states from {len(candidate_valve_q)} candidates."
        )

    def _generate_valid_ee_poses(
        self, env: ManagerBasedEnv, joint_names
    ) -> torch.Tensor:
        """Generate reachable flange configurations with cuRobo v2 batched IK."""
        position_grid_offset, orientation_grid = get_pose_grid(
            n_x=_assert_get_item(self.cfg.params, "n_x", 1),
            n_y=_assert_get_item(self.cfg.params, "n_y", 1),
            n_z=_assert_get_item(self.cfg.params, "n_z", 1),
            n_roll=_assert_get_item(self.cfg.params, "n_roll", 1),
            n_pitch=_assert_get_item(self.cfg.params, "n_pitch", 1),
            n_yaw=_assert_get_item(self.cfg.params, "n_yaw", 1),
            max_x=_assert_positive_float(self.cfg.params, "max_x", 0.0),
            max_y=_assert_positive_float(self.cfg.params, "max_y", 0.0),
            max_z=_assert_positive_float(self.cfg.params, "max_z", 0.0),
            max_roll=_assert_positive_float(self.cfg.params, "max_roll", 0.0),
            max_pitch=_assert_positive_float(self.cfg.params, "max_pitch", 0.0),
            max_yaw=_assert_positive_float(self.cfg.params, "max_yaw", 0.0),
        )
        n_poses = position_grid_offset.shape[0]
        if n_poses < 1:
            raise RuntimeError(
                "There should be more than one point in the grid for RandomizeValveHandlePoseEvent."
            )

        robot_cfg = load_spot_robot_cfg(
            yaml_path=f"{ASSET_DIR}/spot/cumotion/spot.yaml",
            urdf_path=f"{ASSET_DIR}/spot/spot_with_arm.urdf",
        )
        chunk_size = min(n_poses, 64)
        device_cfg = DeviceCfg()
        with curobo_compatible_warp():
            ik_config = InverseKinematicsCfg.create(
                robot=robot_cfg,
                num_seeds=20,
                position_tolerance=0.005,
                orientation_tolerance=0.05,
                self_collision_check=True,
                use_cuda_graph=True,
                max_batch_size=chunk_size,
                device_cfg=device_cfg,
            )
            ik_solver = InverseKinematics(ik_config)
            target_link = ik_solver.tool_frames[0]

            position_grid_offset = device_cfg.to_device(position_grid_offset)
            orientation_quat = math_utils.quat_from_matrix(
                device_cfg.to_device(orientation_grid)
            )

            retract_state = ik_solver.compute_kinematics(ik_solver.default_joint_state)
            retract_pose = retract_state.tool_poses.get_link_pose(target_link)
            # Keep the retract wrist orientation and apply the sampled
            # world-frame roll, pitch, and yaw perturbation.
            goal_position = retract_pose.position[0] + position_grid_offset
            retract_quat = retract_pose.quaternion[0].expand_as(orientation_quat)
            goal_quaternion = math_utils.quat_mul(orientation_quat, retract_quat)

            joint_chunks: list[torch.Tensor] = []

            for start in range(0, n_poses, chunk_size):
                end = min(start + chunk_size, n_poses)
                goal_poses = Pose(
                    position=goal_position[start:end],
                    quaternion=goal_quaternion[start:end],
                )
                result = ik_solver.solve_pose(
                    GoalToolPose.from_poses(
                        {target_link: goal_poses},
                        ordered_tool_frames=ik_solver.tool_frames,
                        num_goalset=1,
                    )
                )
                success = result.success.reshape(-1)[: end - start]
                if not torch.any(success):
                    continue

                js_solution = _squeeze_seed_dim(result.js_solution)
                js_solution = JointState.from_position(
                    js_solution.position[: end - start][success],
                    joint_names=js_solution.joint_names,
                )
                joint_chunks.append(js_solution.reorder(joint_names).position)

        if not joint_chunks:
            raise RuntimeError(
                "Failed to precompute valid valve handle poses for RandomizeValveHandlePoseEvent."
            )

        return torch.cat(joint_chunks, dim=0).to(env.device)

    def _generate_valid_valve_poses(
        self,
        env: ManagerBasedEnv,
        urdf_name: str,
        offset: OffsetCfg,
        th: torch.Tensor,
        root_link_name: str,
        end_link_name: str,
        align_rotation: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return body-from-handle transforms ``(pos, quat_wxyz)`` for each valve angle."""
        chain = pk.build_serial_chain_from_urdf(
            open(urdf_name, mode="rb").read(),
            end_link_name=end_link_name,
            root_link_name=root_link_name,
        )

        th_cpu = th.detach().to(device="cpu")
        if th_cpu.ndim == 1:
            th_cpu = th_cpu[:, None]
        n_dof = getattr(chain, "n_joints", th_cpu.shape[-1])
        if th_cpu.shape[-1] < n_dof:
            pad = torch.zeros(th_cpu.shape[0], n_dof - th_cpu.shape[-1])
            th_cpu = torch.cat([th_cpu, pad], dim=-1)
        ret: pk.Transform3d = chain.forward_kinematics(th_cpu, end_only=True)  # type: ignore
        offset_tf = pk.Transform3d(pos=offset.pos, rot=offset.rot)

        if align_rotation:
            offset_ = ret.compose(offset_tf).inverse()
            pos, rot = math_utils.unmake_pose(offset_.get_matrix().to(env.device))
            return pos, math_utils.quat_from_matrix(rot)
        else:
            full_tf = ret.compose(offset_tf)
            full_mat = full_tf.get_matrix().to(env.device)
            p_tcp_in_root = full_mat[:, :3, 3]  # [batch, 3]

            nom_inv_mat = (
                pk.Transform3d(rot=offset.rot).inverse().get_matrix().to(env.device)
            )  # [1, 4, 4]
            r_tcp_to_root = nom_inv_mat[:, :3, :3]  # [1, 3, 3]
            p_tcp_to_root = torch.bmm(
                r_tcp_to_root.expand(p_tcp_in_root.shape[0], -1, -1),
                (-p_tcp_in_root).unsqueeze(-1),
            ).squeeze(-1)  # [batch, 3]
            q_tcp_to_root = math_utils.quat_from_matrix(
                r_tcp_to_root.expand(p_tcp_in_root.shape[0], -1, -1)
            )  # [batch, 4]
            return p_tcp_to_root, q_tcp_to_root

    def _apply_valve_first_reset_on_the_fly(
        self, env: ManagerBasedEnv, env_ids: torch.Tensor
    ) -> None:
        """Sample reachable valve and arm reset states dynamically on the fly."""
        num_resets = env_ids.numel()
        max_x = _assert_positive_float(self.cfg.params, "max_x", 0.0)
        max_y = _assert_positive_float(self.cfg.params, "max_y", 0.0)
        max_z = _assert_positive_float(self.cfg.params, "max_z", 0.0)
        max_roll = _assert_positive_float(self.cfg.params, "max_roll", 0.0)
        max_pitch = _assert_positive_float(self.cfg.params, "max_pitch", 0.0)
        max_yaw = _assert_positive_float(self.cfg.params, "max_yaw", 0.0)

        valve_joint_range = self.cfg.params.get("valve_joint_range", (0.0, 0.0001))
        if isinstance(valve_joint_range, (tuple, list)) and len(valve_joint_range) >= 2:
            min_valve_q = float(valve_joint_range[0])
            max_valve_q = float(valve_joint_range[1])
        else:
            min_valve_q, max_valve_q = 0.0, 0.0001

        max_sample_attempts = int(self.cfg.params.get("max_reset_attempts", 5))

        active_indices = torch.arange(num_resets, device=env.device)
        final_solutions = torch.zeros(
            num_resets, len(self._ik_joint_names), device=env.device
        )
        final_root_pos_b = torch.zeros(num_resets, 3, device=env.device)
        final_root_quat_b = torch.zeros(num_resets, 4, device=env.device)
        final_valve_q = torch.zeros(num_resets, 1, device=env.device)

        nominal_root_pos = torch.tensor(
            self._valve_root_pose.pos, device=env.device, dtype=torch.float32
        )
        nominal_root_quat = torch.tensor(
            self._valve_root_pose.rot, device=env.device, dtype=torch.float32
        )

        for _ in range(max_sample_attempts):
            n_curr = active_indices.numel()
            if n_curr == 0:
                break

            pos_offset = (
                torch.rand((n_curr, 3), device=env.device) * 2.0 - 1.0
            ) * torch.tensor([max_x, max_y, max_z], device=env.device)
            rpy = (
                torch.rand((n_curr, 3), device=env.device) * 2.0 - 1.0
            ) * torch.tensor([max_roll, max_pitch, max_yaw], device=env.device)

            cos_r, sin_r = torch.cos(rpy[:, 0]), torch.sin(rpy[:, 0])
            cos_p, sin_p = torch.cos(rpy[:, 1]), torch.sin(rpy[:, 1])
            cos_y, sin_y = torch.cos(rpy[:, 2]), torch.sin(rpy[:, 2])

            rotations = torch.empty((n_curr, 3, 3), device=env.device)
            rotations[:, 0, 0] = cos_y * cos_p
            rotations[:, 0, 1] = cos_y * sin_p * sin_r - sin_y * cos_r
            rotations[:, 0, 2] = cos_y * sin_p * cos_r + sin_y * sin_r
            rotations[:, 1, 0] = sin_y * cos_p
            rotations[:, 1, 1] = sin_y * sin_p * sin_r + cos_y * cos_r
            rotations[:, 1, 2] = sin_y * sin_p * cos_r - cos_y * sin_r
            rotations[:, 2, 0] = -sin_p
            rotations[:, 2, 1] = cos_p * sin_r
            rotations[:, 2, 2] = cos_p * cos_r

            orientation_quat = math_utils.quat_from_matrix(rotations)
            root_pos_b = nominal_root_pos.repeat(n_curr, 1) + pos_offset
            root_quat_b = math_utils.quat_mul(
                orientation_quat, nominal_root_quat.repeat(n_curr, 1)
            )

            valve_q = min_valve_q + torch.rand(n_curr, device=env.device) * (
                max_valve_q - min_valve_q
            )
            th_cpu = valve_q.detach().to("cpu")[:, None]
            n_dof = getattr(self._valve_chain, "n_joints", th_cpu.shape[-1])
            if th_cpu.shape[-1] < n_dof:
                th_cpu = torch.cat(
                    [
                        th_cpu,
                        torch.zeros(th_cpu.shape[0], n_dof - th_cpu.shape[-1]),
                    ],
                    dim=-1,
                )

            root_to_handle = self._valve_chain.forward_kinematics(
                th_cpu, end_only=True
            )
            handle_to_tcp = pk.Transform3d(
                pos=self._valve_offset.pos, rot=self._valve_offset.rot
            )
            root_to_tcp = root_to_handle.compose(handle_to_tcp)
            tcp_pos_v, rot_v = math_utils.unmake_pose(
                root_to_tcp.get_matrix().to(env.device)
            )
            tcp_quat_v = math_utils.quat_from_matrix(rot_v)

            tcp_pos_b, tcp_quat_b = math_utils.combine_frame_transforms(
                root_pos_b, root_quat_b, tcp_pos_v, tcp_quat_v
            )
            tcp_quat_b = canonicalize_ee_orientation_upward(tcp_quat_b)

            ee_offset_pos = self._ee_offset_pos.repeat(n_curr, 1)
            ee_offset_quat = self._ee_offset_quat.repeat(n_curr, 1)
            tcp_to_flange_quat = math_utils.quat_inv(ee_offset_quat)
            tcp_to_flange_pos = math_utils.quat_apply(
                tcp_to_flange_quat, -ee_offset_pos
            )
            flange_pos_b, flange_quat_b = math_utils.combine_frame_transforms(
                tcp_pos_b, tcp_quat_b, tcp_to_flange_pos, tcp_to_flange_quat
            )

            ik_targets = pk.Transform3d(
                pos=flange_pos_b.cpu(), rot=flange_quat_b.cpu()
            )
            ik_result = self._ik_solver.solve(ik_targets)
            valid = ik_result.converged_any

            if torch.any(valid):
                valid_local_ids = torch.where(valid)[0]
                retry_ids = (
                    ik_result.converged[valid].to(torch.int64).argmax(dim=1)
                )
                solved_q = ik_result.solutions[valid, retry_ids].to(env.device)

                orig_ids = active_indices[valid_local_ids]
                final_solutions[orig_ids] = solved_q
                final_root_pos_b[orig_ids] = root_pos_b[valid_local_ids]
                final_root_quat_b[orig_ids] = root_quat_b[valid_local_ids]
                final_valve_q[orig_ids] = valve_q[valid_local_ids].unsqueeze(-1)

                active_indices = active_indices[~valid]

        if active_indices.numel() > 0:
            last_best = ik_result.solutions[~valid, 0].to(env.device)
            final_solutions[active_indices] = last_best
            final_root_pos_b[active_indices] = root_pos_b[~valid]
            final_root_quat_b[active_indices] = root_quat_b[~valid]
            final_valve_q[active_indices] = valve_q[~valid].unsqueeze(-1)

        full_joint_states = self._asset.data.default_joint_pos[env_ids][
            :, self._joint_ids
        ].clone()
        joint_column_by_name = {
            name: index for index, name in enumerate(self._joint_names)
        }
        ik_columns = [
            joint_column_by_name[name] for name in self._ik_joint_names
        ]
        full_joint_states[:, ik_columns] = final_solutions

        self._asset.write_joint_state_to_sim(
            full_joint_states,
            torch.zeros_like(full_joint_states),
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )

        default_valve_q = self._valve.data.default_joint_pos[env_ids].clone()
        if final_valve_q.shape[-1] == default_valve_q.shape[-1]:
            valve_states_to_write = final_valve_q
        else:
            valve_states_to_write = default_valve_q
        self._valve.write_joint_state_to_sim(
            valve_states_to_write,
            torch.zeros_like(valve_states_to_write),
            env_ids=env_ids,
        )

        valve_pos_w, valve_quat_w = math_utils.combine_frame_transforms(
            self._asset.data.root_pos_w[env_ids],
            self._asset.data.root_quat_w[env_ids],
            final_root_pos_b,
            final_root_quat_b,
        )
        self._valve.write_root_pose_to_sim(
            torch.cat((valve_pos_w, valve_quat_w), dim=-1), env_ids=env_ids
        )

        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.0)

    def _apply_valve_first_reset(
        self, env: ManagerBasedEnv, env_ids: torch.Tensor
    ) -> None:
        """Write one cached, mutually consistent valve/arm state per environment."""
        pair_ids = torch.randint(
            low=0,
            high=self._num_valid_states,
            size=(env_ids.numel(),),
            device=env.device,
        )
        robot_joint_pos = self._robot_joint_states[pair_ids]
        self._asset.write_joint_state_to_sim(
            robot_joint_pos,
            torch.zeros_like(robot_joint_pos),
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )

        default_valve_q = self._valve.data.default_joint_pos[env_ids].clone()
        selected_valve_q = self._paired_valve_joint_states[pair_ids]
        if selected_valve_q.ndim == 1:
            selected_valve_q = selected_valve_q.unsqueeze(-1)
        if selected_valve_q.shape[-1] != default_valve_q.shape[-1]:
            raise ValueError(
                "Valve-first reset requires one sampled value per valve joint; "
                f"received {selected_valve_q.shape[-1]} for "
                f"{default_valve_q.shape[-1]} joints."
            )
        self._valve.write_joint_state_to_sim(
            selected_valve_q,
            torch.zeros_like(selected_valve_q),
            env_ids=env_ids,
        )

        valve_pos_w, valve_quat_w = math_utils.combine_frame_transforms(
            self._asset.data.root_pos_w[env_ids],
            self._asset.data.root_quat_w[env_ids],
            self._valve_root_pos_b[pair_ids],
            self._valve_root_quat_b[pair_ids],
        )
        self._valve.write_root_pose_to_sim(
            torch.cat((valve_pos_w, valve_quat_w), dim=-1), env_ids=env_ids
        )

        # FrameTransformer data is consumed while the command manager resets.
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.0)

    def __call__(  # type: ignore
        self,
        env: ManagerBasedEnv,
        env_ids: Sequence[int] | None,
        asset_cfg: SceneEntityCfg,
        n_x=0,
        n_y=0,
        n_z=0,
        n_roll=0,
        n_pitch=0,
        n_yaw=0,
        max_x=0,
        max_y=0,
        max_z=0,
        max_roll=0,
        max_pitch=0,
        max_yaw=0,
        valve_cfg: SceneEntityCfg = None,
        valve_offset: OffsetCfg = None,
        ee_offset: OffsetCfg = None,
        frame_name: str = "target_frame",
        target_frame_name: str | None = None,
        command_name: str = "pose_command",
        valve_urdf: str = None,
        valve_root_link: str = None,
        valve_ee_link: str = None,
        valve_root_pose: OffsetCfg | None = None,
        valve_joint_range: tuple | list | float | None = None,
        n_valve_states: int | None = None,
        align_rotation: bool | None = None,
        reset_state_cache_path: str | None = None,
        reset_state_cache_force_regenerate: bool = False,
        show_reset_state_progress: bool = False,
        ik_batch_size: int = 2048,
        on_the_fly: bool = False,
    ) -> None:
        """Randomize the valve handle pose by setting the joint positions.

        Args:
            env: The environment instance.
            env_ids: The environment ids to apply the event to.
        """
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        num_envs = env_ids.numel()

        if self._valve_root_pose is not None:
            if getattr(self, "_on_the_fly", False):
                self._apply_valve_first_reset_on_the_fly(env, env_ids)
            else:
                self._apply_valve_first_reset(env, env_ids)
            return

        if (
            not HAS_CUROBO
            or self._robot_joint_states is None
            or self._num_valid_states == 0
        ):
            default_joint_poses = self._asset.data.default_joint_pos[
                env_ids, self._joint_ids
            ]
            default_joint_vels = torch.zeros_like(default_joint_poses)
            self._asset.write_joint_state_to_sim(
                default_joint_poses,
                default_joint_vels,
                joint_ids=self._joint_ids,
                env_ids=env_ids,
            )
            return

        rand_indices = torch.randint(
            low=0,
            high=self._num_valid_states,
            size=(num_envs,),
            device=env.device,
        )

        selected_joint_poses = self._robot_joint_states[rand_indices]
        selected_joint_velocities = torch.zeros_like(selected_joint_poses)

        self._asset.write_joint_state_to_sim(
            selected_joint_poses,
            selected_joint_velocities,
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )

        # Position the valve body according to the offset
        rand_indices_valve = torch.randint(
            low=0,
            high=self._num_valid_valve_states,
            size=(num_envs,),
            device=env.device,
        )
        default_valve_q = self._valve.data.default_joint_pos[env_ids].clone()
        sample_valve_q = self._valve_joint_states[rand_indices_valve]
        if sample_valve_q.ndim == 1:
            sample_valve_q = sample_valve_q.unsqueeze(-1)
        if sample_valve_q.shape[-1] == default_valve_q.shape[-1]:
            valve_joint_states = sample_valve_q
        else:
            # Sample is the serial-chain handle DoF only (e.g. nut yaw).
            valve_joint_states = default_valve_q
        valve_joint_velos = torch.zeros_like(valve_joint_states)
        self._valve.write_joint_state_to_sim(
            valve_joint_states,
            valve_joint_velos,
            env_ids=env_ids,
        )

        # Resolve the *simulated* TCP after the joint write. Using cuRobo FK
        # here drifted from Isaac's URDF import, so the lever missed the hand.
        env.scene.write_data_to_sim()
        env.sim.forward()

        wr1_pos_w = self._asset.data.body_pos_w[env_ids, self._ee_body_idx]
        wr1_quat_w = self._asset.data.body_quat_w[env_ids, self._ee_body_idx]
        tcp_pos_w, tcp_quat_w = math_utils.combine_frame_transforms(
            wr1_pos_w,
            wr1_quat_w,
            self._ee_offset_pos.expand(num_envs, -1),
            self._ee_offset_quat.expand(num_envs, -1),
        )

        valve_pos_w, valve_quat_w = math_utils.combine_frame_transforms(
            tcp_pos_w,
            tcp_quat_w,
            self._valve_inv_pos[rand_indices_valve],
            self._valve_inv_quat[rand_indices_valve],
        )
        valve_pose = torch.cat([valve_pos_w, valve_quat_w], dim=-1)
        self._valve.write_root_pose_to_sim(valve_pose, env_ids=env_ids)

        # Command handlers snapshot FrameTransformer data during env.reset(),
        # which happens before the env's own forward(). Refresh sensors now so
        # GoToFrame / RotateFrame see the lever under the TCP.
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.0)
