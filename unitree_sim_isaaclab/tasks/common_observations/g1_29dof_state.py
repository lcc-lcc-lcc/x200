# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""G1-compatible body-state utilities with X200 compatibility."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import torch

from robots.body_layouts import (
    G1_COMPAT_BODY_JOINT_NAMES,
    build_g1_compat_body_slot_map,
    get_robot_arm_joint_names as get_robot_arm_joint_names_for_layout,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from dds.dds_master import dds_manager


def get_robot_boy_joint_names() -> list[str]:
    return list(G1_COMPAT_BODY_JOINT_NAMES)


def get_robot_arm_joint_names(joint_names: list[str] | None = None) -> list[str]:
    return list(get_robot_arm_joint_names_for_layout(joint_names))


_g1_robot_dds = None
_dds_initialized = False

_obs_cache = {
    "device": None,
    "batch": None,
    "layout_signature": None,
    "slot_idx_t": None,
    "source_idx_t": None,
    "pos_buf": None,
    "vel_buf": None,
    "torque_buf": None,
    "combined_buf": None,
    "dds_last_ms": 0,
    "dds_min_interval_ms": 20,
}

_imu_acc_cache = {
    "prev_vel": None,
    "dt": 0.01,
    "initialized": False,
}


def _get_g1_robot_dds_instance():
    global _g1_robot_dds, _dds_initialized

    if not _dds_initialized or _g1_robot_dds is None:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager as _dds_manager
            _g1_robot_dds = _dds_manager.get_object("g129")
            print("[g1_state] G1 robot DDS communication instance obtained")

            import atexit

            def cleanup_dds():
                try:
                    if _g1_robot_dds:
                        _dds_manager.unregister_object("g129")
                        print("[g1_state] DDS communication closed correctly")
                except Exception as e:
                    print(f"[g1_state] Error closing DDS: {e}")

            atexit.register(cleanup_dds)
        except Exception as e:
            print(f"[g1_state] Failed to get G1 robot DDS instance: {e}")
            _g1_robot_dds = None

        _dds_initialized = True

    return _g1_robot_dds


def get_robot_boy_joint_states(
    env: ManagerBasedRLEnv,
    enable_dds: bool = True,
) -> torch.Tensor:
    """Return a G1-compatible 29-slot body state tensor, zero-filling joints absent in X200."""
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel
    joint_torque = env.scene["robot"].data.applied_torque
    all_joint_names = env.scene["robot"].data.joint_names
    device = joint_pos.device
    batch = joint_pos.shape[0]
    num_slots = len(G1_COMPAT_BODY_JOINT_NAMES)

    global _obs_cache
    layout_signature = tuple(all_joint_names)
    if (
        _obs_cache["device"] != device
        or _obs_cache["batch"] != batch
        or _obs_cache["layout_signature"] != layout_signature
    ):
        slot_map = build_g1_compat_body_slot_map(all_joint_names)
        present_slots = [slot for slot, src_idx in enumerate(slot_map) if src_idx is not None]
        source_indices = [src_idx for src_idx in slot_map if src_idx is not None]
        _obs_cache["slot_idx_t"] = torch.tensor(present_slots, dtype=torch.long, device=device)
        _obs_cache["source_idx_t"] = torch.tensor(source_indices, dtype=torch.long, device=device)
        _obs_cache["pos_buf"] = torch.zeros(batch, num_slots, device=device, dtype=joint_pos.dtype)
        _obs_cache["vel_buf"] = torch.zeros(batch, num_slots, device=device, dtype=joint_pos.dtype)
        _obs_cache["torque_buf"] = torch.zeros(batch, num_slots, device=device, dtype=joint_pos.dtype)
        _obs_cache["combined_buf"] = torch.empty(batch, num_slots * 3, device=device, dtype=joint_pos.dtype)
        _obs_cache["device"] = device
        _obs_cache["batch"] = batch
        _obs_cache["layout_signature"] = layout_signature

    slot_idx_t = _obs_cache["slot_idx_t"]
    source_idx_t = _obs_cache["source_idx_t"]
    pos_buf = _obs_cache["pos_buf"]
    vel_buf = _obs_cache["vel_buf"]
    torque_buf = _obs_cache["torque_buf"]
    combined_buf = _obs_cache["combined_buf"]

    pos_buf.zero_()
    vel_buf.zero_()
    torque_buf.zero_()

    if source_idx_t.numel() > 0:
        pos_selected = joint_pos.index_select(1, source_idx_t)
        vel_selected = joint_vel.index_select(1, source_idx_t)
        torque_selected = joint_torque.index_select(1, source_idx_t)
        pos_buf.index_copy_(1, slot_idx_t, pos_selected)
        vel_buf.index_copy_(1, slot_idx_t, vel_selected)
        torque_buf.index_copy_(1, slot_idx_t, torque_selected)

    combined_buf[:, 0:num_slots].copy_(pos_buf)
    combined_buf[:, num_slots:2 * num_slots].copy_(vel_buf)
    combined_buf[:, 2 * num_slots:3 * num_slots].copy_(torque_buf)

    if enable_dds and combined_buf.shape[0] > 0:
        try:
            import time

            now_ms = int(time.time() * 1000)
            if now_ms - _obs_cache["dds_last_ms"] >= _obs_cache["dds_min_interval_ms"]:
                g1_robot_dds = _get_g1_robot_dds_instance()
                if g1_robot_dds:
                    imu_data = get_robot_imu_data(env)
                    if imu_data.shape[0] > 0:
                        g1_robot_dds.write_robot_state(
                            pos_buf[0].contiguous().cpu().numpy(),
                            vel_buf[0].contiguous().cpu().numpy(),
                            torque_buf[0].contiguous().cpu().numpy(),
                            imu_data[0].contiguous().cpu().numpy(),
                        )
                        _obs_cache["dds_last_ms"] = now_ms
        except Exception as e:
            print(f"[g1_state] Error writing robot state to DDS: {e}")

    return combined_buf


def quat_to_rot_matrix(q):
    """Convert quaternions in (w, x, y, z) to rotation matrices."""
    w = q[:, 0:1]
    x = q[:, 1:2]
    y = q[:, 2:3]
    z = q[:, 3:4]

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    r00 = ww + xx - yy - zz
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)

    r10 = 2 * (xy + wz)
    r11 = ww - xx + yy - zz
    r12 = 2 * (yz - wx)

    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = ww - xx - yy + zz

    R = torch.cat([
        torch.cat([r00, r01, r02], dim=1).unsqueeze(1),
        torch.cat([r10, r11, r12], dim=1).unsqueeze(1),
        torch.cat([r20, r21, r22], dim=1).unsqueeze(1),
    ], dim=1)
    return R.view(-1, 3, 3)


def ensure_quat_w_first(quat, assume_w_first=None):
    if assume_w_first is True:
        return quat
    if assume_w_first is False:
        return torch.cat([quat[:, 3:4], quat[:, 0:3]], dim=1)

    mean0 = torch.mean(torch.abs(quat[:, 0]))
    mean3 = torch.mean(torch.abs(quat[:, 3]))
    if mean0 > 0.9:
        return quat
    if mean3 > 0.9:
        return torch.cat([quat[:, 3:4], quat[:, 0:3]], dim=1)
    return quat


def get_robot_imu_data(env, use_torso_imu: bool = True, quat_w_first: bool = None) -> torch.Tensor:
    data = env.scene["robot"].data
    global _imu_acc_cache

    dt = _imu_acc_cache["dt"]
    try:
        if hasattr(env, "physics_dt"):
            dt = float(env.physics_dt)
        elif hasattr(env, "step_dt"):
            dt = float(env.step_dt)
        elif hasattr(env, "dt"):
            dt = float(env.dt)
    except Exception:
        pass
    if dt <= 0:
        dt = _imu_acc_cache["dt"]

    if use_torso_imu:
        try:
            body_names = data.body_names
            imu_idx = body_names.index("imu_in_torso")
            body_pose = data.body_link_pose_w
            body_vel = data.body_link_vel_w
            pos = body_pose[:, imu_idx, :3]
            quat = body_pose[:, imu_idx, 3:7]
            lin_vel = body_vel[:, imu_idx, :3]
            ang_vel_world = body_vel[:, imu_idx, 3:6]
        except ValueError:
            use_torso_imu = False

    if not use_torso_imu:
        root_state = data.root_state_w
        pos = root_state[:, :3]
        quat = root_state[:, 3:7]
        lin_vel = root_state[:, 7:10]
        ang_vel_world = root_state[:, 10:13]

    device = lin_vel.device if isinstance(lin_vel, torch.Tensor) else torch.device("cpu")
    quat = quat.to(device)
    lin_vel = lin_vel.to(device)
    ang_vel_world = ang_vel_world.to(device)

    if _imu_acc_cache["prev_vel"] is None:
        _imu_acc_cache["prev_vel"] = lin_vel.clone().detach().to(device)
        _imu_acc_cache["initialized"] = False
    else:
        if _imu_acc_cache["prev_vel"].device != device:
            _imu_acc_cache["prev_vel"] = _imu_acc_cache["prev_vel"].to(device)

    a_world = (lin_vel - _imu_acc_cache["prev_vel"]) / dt
    g_world = torch.zeros_like(a_world)
    g_world[:, 2] = -9.81
    a_world_corrected = a_world - g_world

    quat_wxyz = ensure_quat_w_first(quat, assume_w_first=True)
    R_body_to_world = quat_to_rot_matrix(quat_wxyz)
    R_world_to_body = R_body_to_world.transpose(1, 2)

    a_body = torch.bmm(R_world_to_body, a_world_corrected.unsqueeze(-1)).squeeze(-1)
    omega_body = torch.bmm(R_world_to_body, ang_vel_world.unsqueeze(-1)).squeeze(-1)

    if not _imu_acc_cache["initialized"]:
        a_body = torch.bmm(R_world_to_body, (-g_world).unsqueeze(-1)).squeeze(-1)
        _imu_acc_cache["initialized"] = True

    _imu_acc_cache["prev_vel"] = lin_vel.clone().detach()
    _imu_acc_cache["dt"] = dt

    return torch.cat([pos, quat_wxyz, a_body, omega_body], dim=1)
