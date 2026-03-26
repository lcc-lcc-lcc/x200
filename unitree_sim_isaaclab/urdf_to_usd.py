# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from __future__ import annotations

import os  # 处理临时 URDF 文件删除等文件系统操作。
import re  # 解析并替换 URDF 里的 package:// 路径。
import sys  # 用于在失败时返回非零退出码，并把错误打印到 stderr。
import tempfile  # 为预处理后的 URDF 生成临时文件。
import traceback  # 失败时打印完整堆栈，方便定位 Isaac Sim/URDF 导入问题。
import xml.etree.ElementTree as ET  # 读取 ROS package.xml，解析 package 名称。
from pathlib import Path  # 用统一的 Path 接口处理仓库内外路径。

from isaaclab.app import AppLauncher  # 启动 Isaac Lab / Isaac Sim 应用。

# 仓库根目录。当前脚本位于 repo root。
REPO_ROOT = Path(__file__).resolve().parent
# 固定输入/输出路径。
# 现在这个脚本完全不读取 YAML，而是直接把要转换的 URDF 和输出 USD 路径写死在代码里。
FIXED_ASSET_PATH = REPO_ROOT / "assets/robots/x200-11-03/urdf/x200-11-03_with_hand.urdf"
FIXED_USD_DIR = REPO_ROOT / "assets/robots/x200-11-03-base-fix/usd"
FIXED_USD_FILE_NAME = "x200-11-03-base-fix.usd"
# 启动底层 Isaac Sim 应用。这个脚本不再解析业务参数，直接启动即可。
app_launcher = AppLauncher()
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # Stage 创建等仿真工具。
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # Isaac Lab 提供的 URDF -> USD 转换器。
import omni.kit.commands  # 直接调用 URDF importer 命令，构造 ImportConfig。

class CompatibleUrdfConverter(UrdfConverter):
    """兼容 Isaac Sim 4.5 的 URDF 转换器。

    当前仓库里的 Isaac Lab `UrdfConverter` 会调用
    `import_config.set_merge_fixed_ignore_inertia(...)`。
    但在用户当前环境里的 URDF importer 版本
    `isaacsim.asset.importer.urdf-2.3.10` 上，这个接口并不存在，
    会直接抛出 AttributeError。

    因此这里仅覆写 `_get_urdf_import_config()`：
    - 其余逻辑保持和 Isaac Lab 原实现一致；
    - 仅在 importer 真正支持该接口时才调用；
    - 对 Isaac Sim 4.5 这种旧版 importer，则跳过该设置。

    这样可以避免去修改外部 Isaac Lab 安装，同时保持当前脚本自包含可用。
    """

    def _get_urdf_import_config(self):
        """构造并填充 URDF ImportConfig，同时兼容旧版 importer 接口。"""
        _, import_config = omni.kit.commands.execute("URDFCreateImportConfig")

        # 基本导入选项。
        import_config.set_distance_scale(1.0)
        import_config.set_make_default_prim(True)
        import_config.set_create_physics_scene(False)

        # 资产相关选项。
        import_config.set_density(self.cfg.link_density)
        import_config.set_convex_decomp(self.cfg.collider_type == "convex_decomposition")
        import_config.set_collision_from_visuals(self.cfg.collision_from_visuals)
        import_config.set_merge_fixed_joints(self.cfg.merge_fixed_joints)

        # 这个接口在较新的 importer 中存在，但 Isaac Sim 4.5 自带的
        # isaacsim.asset.importer.urdf-2.3.10 没有它。旧版 importer 本身就走旧行为，
        # 因此这里安全地按能力探测处理即可。
        if hasattr(import_config, "set_merge_fixed_ignore_inertia"):
            import_config.set_merge_fixed_ignore_inertia(self.cfg.merge_fixed_joints)

        # 物理相关选项。
        import_config.set_fix_base(self.cfg.fix_base)
        import_config.set_self_collision(self.cfg.self_collision)
        import_config.set_parse_mimic(self.cfg.convert_mimic_joints_to_normal_joints)
        import_config.set_replace_cylinders_with_capsules(self.cfg.replace_cylinders_with_capsules)

        return import_config

    def _convert_asset(self, cfg: UrdfConverterCfg):
        """执行 URDF -> USD 转换，并额外兼容 package:// mesh 路径。

        这里覆写 Isaac Lab 默认实现，主要多做两件事：
        1. 在导入前把 `package://pkg_name/...` 解析成真实绝对路径；
        2. 对 `URDFImportRobot` 的返回状态做显式检查，避免日志里已经报错但外层还误判成功。
        """
        import_config = self._get_urdf_import_config()
        prepared_urdf_path, cleanup_path = self._prepare_urdf_for_import(cfg.asset_path)

        try:
            # 先解析 URDF；如果解析阶段就失败，直接抛错停止。
            parse_result, self._robot_model = omni.kit.commands.execute(
                "URDFParseFile", urdf_path=prepared_urdf_path, import_config=import_config
            )
            if not parse_result:
                raise ValueError(f"Failed to parse URDF file: {prepared_urdf_path}")

            if cfg.joint_drive:
                self._update_joint_parameters()

            if cfg.root_link_name:
                self._robot_model.root_link = cfg.root_link_name

            # 真正导入机器人并写出 USD。这里必须检查 execute 的布尔返回值，
            # 否则 importer 仅在日志中报错时，外层可能还会继续往下执行。
            import_result, imported_path = omni.kit.commands.execute(
                "URDFImportRobot",
                urdf_path=prepared_urdf_path,
                urdf_robot=self._robot_model,
                import_config=import_config,
                dest_path=self.usd_path,
            )
            if not import_result:
                raise RuntimeError(f"Failed to import URDF robot into USD: {prepared_urdf_path}")

            print(f"[INFO] Imported robot prim path: {imported_path}")
        finally:
            if cleanup_path is not None and os.path.exists(cleanup_path):
                os.remove(cleanup_path)

    def _prepare_urdf_for_import(self, urdf_path: str) -> tuple[str, str | None]:
        """为导入器准备一份 URDF。

        如果 URDF 中包含 `package://...`，则生成一份临时 URDF，把这些 URI 展开成真实绝对路径。
        这样即便 Isaac Sim 当前进程没有 ROS package 环境，也能找到 mesh 文件。

        Returns:
            一个二元组 `(prepared_path, cleanup_path)`：
            - `prepared_path` 是实际传给 importer 的 URDF 路径；
            - `cleanup_path` 是需要在导入后删除的临时文件路径；若无需清理则为 None。
        """
        original_path = Path(urdf_path).resolve(strict=True)
        urdf_text = original_path.read_text(encoding="utf-8")

        if "package://" not in urdf_text:
            return str(original_path), None

        package_roots = self._discover_ros_package_roots(original_path)
        resolved_text, replacement_count, unresolved_packages = self._replace_package_urls(urdf_text, package_roots)

        if unresolved_packages:
            raise FileNotFoundError(
                "Unable to resolve the following ROS package names in URDF mesh paths: "
                + ", ".join(sorted(unresolved_packages))
            )

        if replacement_count == 0:
            return str(original_path), None

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".urdf",
            prefix=f"{original_path.stem}_resolved_",
            delete=False,
            encoding="utf-8",
        ) as temp_file:
            temp_file.write(resolved_text)
            temp_path = temp_file.name

        print(f"[INFO] Resolved {replacement_count} package:// mesh paths via temporary URDF: {temp_path}")
        return temp_path, temp_path

    def _discover_ros_package_roots(self, urdf_path: Path) -> dict[str, Path]:
        """从 URDF 所在目录向上搜索 ROS package.xml，构建 package_name -> package_root 映射。"""
        package_roots: dict[str, Path] = {}

        for search_dir in [urdf_path.parent, *urdf_path.parent.parents]:
            package_xml_path = search_dir / "package.xml"
            if not package_xml_path.is_file():
                continue

            try:
                package_xml = ET.parse(package_xml_path)
                package_name = package_xml.getroot().findtext("name")
            except ET.ParseError as exc:
                raise ValueError(f"Failed to parse ROS package.xml: {package_xml_path}") from exc

            if package_name:
                package_roots[package_name.strip()] = search_dir

        return package_roots

    def _replace_package_urls(
        self, urdf_text: str, package_roots: dict[str, Path]
    ) -> tuple[str, int, set[str]]:
        """把 `package://pkg_name/...` 替换为真实绝对路径。"""
        replacement_count = 0
        unresolved_packages: set[str] = set()

        def _replacement(match: re.Match[str]) -> str:
            nonlocal replacement_count
            package_name = match.group("package")
            relative_path = match.group("relative_path")
            package_root = package_roots.get(package_name)
            if package_root is None:
                unresolved_packages.add(package_name)
                return match.group(0)

            resolved_path = (package_root / relative_path).resolve(strict=False)
            replacement_count += 1
            return resolved_path.as_posix()

        resolved_text = re.sub(
            r"package://(?P<package>[^/]+)/(?P<relative_path>[^\"'>\s]+)",
            _replacement,
            urdf_text,
        )
        return resolved_text, replacement_count, unresolved_packages

def _build_fixed_cfg() -> UrdfConverterCfg:
    """构造固定的 URDF 转换配置。

    这里不再读取任何 YAML。
    需要换机器人时，直接改本文件顶部的 `FIXED_ASSET_PATH`、`FIXED_USD_DIR`、`FIXED_USD_FILE_NAME`。
    """
    return UrdfConverterCfg(
        asset_path=str(FIXED_ASSET_PATH.resolve(strict=False)),  # 固定 URDF 输入路径。
        usd_dir=str(FIXED_USD_DIR.resolve(strict=False)),  # 固定 USD 输出目录。
        usd_file_name=FIXED_USD_FILE_NAME,  # 固定 USD 文件名。
        force_usd_conversion=True,  # 当前脚本默认强制重导，避免旧缓存影响排查。
        make_instanceable=True,  # 默认导出为 instanceable 资产。
        # fix_base=False,  # 固定 base，导出 base-fix 版本。
        fix_base=True,  # 固定 base，导出 base-fix 版本。
        root_link_name=None,  # 默认让 PhysX/导入器决定根 link。
        link_density=0.0,  # 缺失惯量信息时使用自动计算。
        merge_fixed_joints=True,  # 默认合并 fixed joint 连接的 link。
        convert_mimic_joints_to_normal_joints=False,  # 默认不展开 mimic joint。
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",  # 关节驱动方式。
            target_type="none",  # 默认不写 position/velocity 目标。
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),  # 缺省 PD 参数。
        ),
        collision_from_visuals=False,  # 默认不从 visual mesh 生成碰撞体。
        collider_type="convex_hull",  # 默认使用 convex hull。
        self_collision=False,  # 默认关闭自碰撞。
        replace_cylinders_with_capsules=False,  # 默认不替换 cylinder。
    )

def main() -> None:
    """脚本主入口。

    整体流程：
    1. 组装最终配置。
    2. 打印关键路径和参数，便于确认当前到底在导什么。
    3. 创建一个新的 USD stage。
    4. 调用 UrdfConverter 完成导入和导出。
    5. 打印输出文件路径。
    """
    cfg = _build_fixed_cfg()
    if not Path(cfg.asset_path).is_file():
        raise FileNotFoundError(f"URDF file not found: {cfg.asset_path}")

    # 打印最关键的 I/O 信息，确认脚本当前正在处理哪份 URDF、往哪导出。
    print("[INFO] Using built-in converter config from urdf_to_usd.py")
    print(f"[INFO] Converting URDF: {cfg.asset_path}")
    print(f"[INFO] USD directory: {cfg.usd_dir if cfg.usd_dir is not None else '<auto>'}")
    if cfg.usd_file_name is not None:
        print(f"[INFO] USD file name: {cfg.usd_file_name}")

    try:
        sim_utils.create_new_stage()  # 先准备一个干净的 stage，避免拿旧场景继续操作。
        converter = CompatibleUrdfConverter(cfg)  # 使用带版本兼容处理的转换器执行 URDF -> USD。
    except Exception as exc:
        # 失败时把最核心的上下文都打出来，方便快速定位是路径、配置还是导入器本身的问题。
        print("[ERROR] URDF to USD conversion failed.", file=sys.stderr)
        print(f"[ERROR] URDF path: {cfg.asset_path}", file=sys.stderr)
        print(f"[ERROR] USD directory: {cfg.usd_dir if cfg.usd_dir is not None else '<auto>'}", file=sys.stderr)
        if cfg.usd_file_name is not None:
            print(f"[ERROR] USD file name: {cfg.usd_file_name}", file=sys.stderr)
        print(f"[ERROR] Exception: {exc}", file=sys.stderr)
        print("[ERROR] Traceback:", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)

    # 走到这里说明转换器已经成功生成了 USD。
    print(f"[INFO] Generated USD: {converter.usd_path}")
    print(f"[INFO] Saved converter config: {Path(converter.usd_dir) / 'config.yaml'}")

if __name__ == "__main__":
    try:
        main()  # 执行主流程。
    except SystemExit:
        raise
    except Exception as exc:
        print("[ERROR] Script failed before completing URDF conversion.", file=sys.stderr)
        print(f"[ERROR] Exception: {exc}", file=sys.stderr)
        print("[ERROR] Traceback:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)  # 失败时显式返回非零退出码，方便 shell/脚本链路判断失败。
    finally:
        simulation_app.close()  # 无论成功失败都关闭 Isaac Sim 应用，避免进程残留。
