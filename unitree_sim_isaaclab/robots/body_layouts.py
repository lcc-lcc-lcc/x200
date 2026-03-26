"""Shared body layouts for G1-compatible DDS flows and X200-backed body assets."""

from __future__ import annotations

from typing import Iterable

G1_COMPAT_BODY_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_COMPAT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

X200_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
]

X200_TORSO_INIT_POS = {"torso_joint": 0.0}
X200_ARM_INIT_POS = {name: 0.0 for name in X200_ARM_JOINT_NAMES}

G1_COMPAT_ARM_COMMAND_SOURCE_INDICES = {
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19,
    "left_wrist_pitch_joint": 20,
    "left_wrist_yaw_joint": 21,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26,
    "right_wrist_pitch_joint": 27,
    "right_wrist_yaw_joint": 28,
}

X200_ARM_COMMAND_SOURCE_INDICES = {
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26,
}

X200_COMPAT_BODY_ALIASES = {
    "waist_yaw_joint": "torso_joint",
    "waist_roll_joint": None,
    "waist_pitch_joint": None,
    "left_wrist_pitch_joint": None,
    "left_wrist_yaw_joint": None,
    "right_wrist_pitch_joint": None,
    "right_wrist_yaw_joint": None,
}


def uses_x200_body_layout(joint_names: Iterable[str]) -> bool:
    joint_name_set = set(joint_names)
    return (
        "torso_joint" in joint_name_set
        and "left_wrist_roll_joint" in joint_name_set
        and "left_wrist_pitch_joint" not in joint_name_set
    )


def get_robot_arm_joint_names(joint_names: Iterable[str] | None = None) -> list[str]:
    if joint_names is not None and uses_x200_body_layout(joint_names):
        return X200_ARM_JOINT_NAMES
    return G1_COMPAT_ARM_JOINT_NAMES


def build_g1_compat_body_slot_map(joint_names: Iterable[str]) -> list[int | None]:
    joint_to_index = {name: idx for idx, name in enumerate(joint_names)}
    x200_layout = uses_x200_body_layout(joint_to_index.keys())
    slot_map: list[int | None] = []
    for compat_name in G1_COMPAT_BODY_JOINT_NAMES:
        if compat_name in joint_to_index:
            slot_map.append(joint_to_index[compat_name])
            continue
        actual_name = X200_COMPAT_BODY_ALIASES.get(compat_name) if x200_layout else None
        slot_map.append(joint_to_index.get(actual_name) if actual_name is not None else None)
    return slot_map


def build_arm_command_mapping(joint_names: Iterable[str]) -> dict[str, int]:
    if uses_x200_body_layout(joint_names):
        return dict(X200_ARM_COMMAND_SOURCE_INDICES)
    return dict(G1_COMPAT_ARM_COMMAND_SOURCE_INDICES)
