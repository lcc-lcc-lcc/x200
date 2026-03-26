"""Shared hand layouts for legacy Inspire and X200-backed Inspire DDS flows."""

from __future__ import annotations

from typing import Iterable
import os


LEGACY_INSPIRE_DDS_JOINT_NAMES = [
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_middle_proximal_joint",
    "L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
]

LEGACY_INSPIRE_SPECIAL_JOINT_MAP = {
    "L_index_intermediate_joint": ("L_index_proximal_joint", 1.0),
    "L_middle_intermediate_joint": ("L_middle_proximal_joint", 1.0),
    "L_pinky_intermediate_joint": ("L_pinky_proximal_joint", 1.0),
    "L_ring_intermediate_joint": ("L_ring_proximal_joint", 1.0),
    "L_thumb_intermediate_joint": ("L_thumb_proximal_pitch_joint", 1.5),
    "L_thumb_distal_joint": ("L_thumb_proximal_pitch_joint", 2.4),
    "R_index_intermediate_joint": ("R_index_proximal_joint", 1.0),
    "R_middle_intermediate_joint": ("R_middle_proximal_joint", 1.0),
    "R_pinky_intermediate_joint": ("R_pinky_proximal_joint", 1.0),
    "R_ring_intermediate_joint": ("R_ring_proximal_joint", 1.0),
    "R_thumb_intermediate_joint": ("R_thumb_proximal_pitch_joint", 1.5),
    "R_thumb_distal_joint": ("R_thumb_proximal_pitch_joint", 2.4),
}

LEGACY_INSPIRE_DDS_JOINT_LIMITS = (
    (0.0, 1.7),
    (0.0, 1.7),
    (0.0, 1.7),
    (0.0, 1.7),
    (0.0, 0.5),
    (-0.1, 1.3),
    (0.0, 1.7),
    (0.0, 1.7),
    (0.0, 1.7),
    (0.0, 1.7),
    (0.0, 0.5),
    (-0.1, 1.3),
)

X200_DDS_JOINT_NAMES = [
    "r_pinky_mcp_pitch",
    "r_ring_mcp_pitch",
    "r_middle_mcp_pitch",
    "r_index_mcp_pitch",
    "r_thumb_cmc_pitch",
    "r_thumb_cmc_yaw",
    "l_pinky_mcp_pitch",
    "l_ring_mcp_pitch",
    "l_middle_mcp_pitch",
    "l_index_mcp_pitch",
    "l_thumb_cmc_pitch",
    "l_thumb_cmc_yaw",
]

X200_SPECIAL_JOINT_MAP = {
    "l_index_dip": ("l_index_mcp_pitch", 0.89),
    "l_middle_dip": ("l_middle_mcp_pitch", 0.89),
    "l_ring_dip": ("l_ring_mcp_pitch", 0.89),
    "l_pinky_dip": ("l_pinky_mcp_pitch", 0.89),
    "l_thumb_ip": ("l_thumb_cmc_pitch", 1.83),
    "r_index_dip": ("r_index_mcp_pitch", 0.89),
    "r_middle_dip": ("r_middle_mcp_pitch", 0.89),
    "r_ring_dip": ("r_ring_mcp_pitch", 0.89),
    "r_pinky_dip": ("r_pinky_mcp_pitch", 0.89),
    "r_thumb_ip": ("r_thumb_cmc_pitch", 1.83),
}

X200_DDS_JOINT_LIMITS = (
    (0.0, 1.57),
    (0.0, 1.57),
    (0.0, 1.57),
    (0.0, 1.57),
    (0.0, 0.52),
    (0.0, 1.54),
    (0.0, 1.57),
    (0.0, 1.57),
    (0.0, 1.57),
    (0.0, 1.57),
    (0.0, 0.52),
    (0.0, 1.54),
)

X200_ALL_HAND_JOINT_NAMES = X200_DDS_JOINT_NAMES + list(X200_SPECIAL_JOINT_MAP.keys())
X200_HAND_INIT_POS = {name: 0.0 for name in X200_ALL_HAND_JOINT_NAMES}


def uses_x200_hand_layout(joint_names: Iterable[str]) -> bool:
    joint_name_set = set(joint_names)
    return all(name in joint_name_set for name in X200_DDS_JOINT_NAMES)


def get_inspire_primary_joint_names(joint_names: Iterable[str]) -> list[str]:
    if uses_x200_hand_layout(joint_names):
        return X200_DDS_JOINT_NAMES
    return LEGACY_INSPIRE_DDS_JOINT_NAMES


def get_inspire_special_joint_map(joint_names: Iterable[str]) -> dict[str, tuple[str, float]]:
    if uses_x200_hand_layout(joint_names):
        return X200_SPECIAL_JOINT_MAP
    return LEGACY_INSPIRE_SPECIAL_JOINT_MAP


def build_inspire_primary_joint_mapping(joint_names: Iterable[str]) -> dict[str, int]:
    primary_joint_names = get_inspire_primary_joint_names(joint_names)
    joint_name_set = set(joint_names)
    missing = [name for name in primary_joint_names if name not in joint_name_set]
    if missing:
        raise ValueError(f"Missing inspire/X200 primary joints in loaded robot asset: {missing}")
    return {name: idx for idx, name in enumerate(primary_joint_names)}


def build_inspire_special_joint_mapping(joint_names: Iterable[str]) -> dict[str, list[float]]:
    primary_joint_names = get_inspire_primary_joint_names(joint_names)
    primary_source_indices = {name: idx for idx, name in enumerate(primary_joint_names)}
    special_joint_map = get_inspire_special_joint_map(joint_names)
    available_joint_names = set(joint_names)
    return {
        target_name: [primary_source_indices[source_name], scale]
        for target_name, (source_name, scale) in special_joint_map.items()
        if target_name in available_joint_names
    }


def get_inspire_dds_joint_limits(profile: str | None = None) -> tuple[tuple[float, float], ...]:
    selected_profile = (profile or os.environ.get("UNITREE_INSPIRE_HAND_PROFILE", "x200")).lower()
    if selected_profile in {"legacy", "legacy_inspire", "inspire"}:
        return LEGACY_INSPIRE_DDS_JOINT_LIMITS
    return X200_DDS_JOINT_LIMITS
