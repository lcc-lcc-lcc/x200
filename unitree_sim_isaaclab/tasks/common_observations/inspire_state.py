# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""Inspire/X200 hand state observation and DDS publishing."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import torch

from robots.hand_layouts import get_inspire_primary_joint_names

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_obs_cache = {
    "device": None,
    "batch": None,
    "joint_name_signature": None,
    "inspire_idx_t": None,
    "inspire_idx_batch": None,
    "pos_buf": None,
    "vel_buf": None,
    "torque_buf": None,
    "dds_last_ms": 0,
    "dds_min_interval_ms": 20,
}


def get_robot_girl_joint_names(joint_names: list[str] | None = None) -> list[str]:
    return get_inspire_primary_joint_names(joint_names or [])


_inspire_dds = None
_dds_initialized = False


def _get_inspire_dds_instance():
    """Get the DDS instance with lazy initialization."""
    global _inspire_dds, _dds_initialized

    if not _dds_initialized or _inspire_dds is None:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dds'))
            from dds.dds_master import dds_manager

            _inspire_dds = dds_manager.get_object("inspire")
            print("[Observations] DDS communication instance obtained")

            import atexit

            def cleanup_dds():
                try:
                    if _inspire_dds:
                        dds_manager.unregister_object("inspire")
                        print("[inspire_state] DDS communication closed correctly")
                except Exception as exc:
                    print(f"[inspire_state] Error closing DDS: {exc}")

            atexit.register(cleanup_dds)
        except Exception as exc:
            print(f"[Observations] Failed to get DDS instances: {exc}")
            _inspire_dds = None

        _dds_initialized = True

    return _inspire_dds


def get_robot_inspire_joint_states(
    env: ManagerBasedRLEnv,
    enable_dds: bool = True,
) -> torch.Tensor:
    """Get the active Inspire/X200 hand joint states and publish them to DDS."""
    joint_pos = env.scene["robot"].data.joint_pos
    joint_vel = env.scene["robot"].data.joint_vel
    joint_torque = env.scene["robot"].data.applied_torque
    all_joint_names = env.scene["robot"].data.joint_names
    primary_joint_names = get_inspire_primary_joint_names(all_joint_names)
    joint_name_signature = tuple(primary_joint_names)
    device = joint_pos.device
    batch = joint_pos.shape[0]

    global _obs_cache
    if (
        _obs_cache["device"] != device
        or _obs_cache["inspire_idx_t"] is None
        or _obs_cache["joint_name_signature"] != joint_name_signature
    ):
        joint_to_index = {name: idx for idx, name in enumerate(all_joint_names)}
        inspire_joint_indices = [joint_to_index[name] for name in primary_joint_names]
        _obs_cache["inspire_idx_t"] = torch.tensor(inspire_joint_indices, dtype=torch.long, device=device)
        _obs_cache["device"] = device
        _obs_cache["batch"] = None
        _obs_cache["joint_name_signature"] = joint_name_signature

    idx_t = _obs_cache["inspire_idx_t"]
    n = idx_t.numel()

    if _obs_cache["batch"] != batch or _obs_cache["inspire_idx_batch"] is None:
        _obs_cache["inspire_idx_batch"] = idx_t.unsqueeze(0).expand(batch, n)
        _obs_cache["pos_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["vel_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["torque_buf"] = torch.empty(batch, n, device=device, dtype=joint_pos.dtype)
        _obs_cache["batch"] = batch

    idx_batch = _obs_cache["inspire_idx_batch"]
    pos_buf = _obs_cache["pos_buf"]
    vel_buf = _obs_cache["vel_buf"]
    torque_buf = _obs_cache["torque_buf"]

    try:
        torch.gather(joint_pos, 1, idx_batch, out=pos_buf)
        torch.gather(joint_vel, 1, idx_batch, out=vel_buf)
        torch.gather(joint_torque, 1, idx_batch, out=torque_buf)
    except TypeError:
        pos_buf.copy_(torch.gather(joint_pos, 1, idx_batch))
        vel_buf.copy_(torch.gather(joint_vel, 1, idx_batch))
        torque_buf.copy_(torch.gather(joint_torque, 1, idx_batch))

    if enable_dds and len(pos_buf) > 0:
        try:
            import time

            now_ms = int(time.time() * 1000)
            if now_ms - _obs_cache["dds_last_ms"] >= _obs_cache["dds_min_interval_ms"]:
                inspire_dds = _get_inspire_dds_instance()
                if inspire_dds:
                    inspire_dds.write_inspire_state(
                        pos_buf[0].contiguous().cpu().numpy(),
                        vel_buf[0].contiguous().cpu().numpy(),
                        torque_buf[0].contiguous().cpu().numpy(),
                    )
                    _obs_cache["dds_last_ms"] = now_ms
        except Exception as exc:
            print(f"[inspire_state] Failed to write to shared memory: {exc}")

    return pos_buf
