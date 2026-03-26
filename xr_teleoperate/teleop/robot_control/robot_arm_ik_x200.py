import casadi
import meshcat.geometry as mg
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from pinocchio.visualize import MeshcatVisualizer
import os
import sys
import pickle
import logging_mp
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TELEOP_DIR = os.path.dirname(THIS_DIR)
XR_ROOT_DIR = os.path.dirname(TELEOP_DIR)

logger_mp = logging_mp.getLogger(__name__)
parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)

from teleop.utils.weighted_moving_filter import WeightedMovingFilter


class X200_ArmIK:
    def __init__(self, Unit_Test=False, Visualization=False):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization
        self.cache_path = os.path.join(TELEOP_DIR, "x200_arm_model_cache.pkl")

        assets_root = os.path.join(XR_ROOT_DIR, "assets")
        if self.Unit_Test:
            assets_root = os.path.abspath(os.path.join(THIS_DIR, "../../assets"))
        self.urdf_path = os.path.join(assets_root, "x200-11-03", "urdf", "x200-11-03_with_hand.urdf")
        self.model_dir = assets_root
        self.arm_joint_names = [
            'left_shoulder_pitch_joint',
            'left_shoulder_roll_joint',
            'left_shoulder_yaw_joint',
            'left_elbow_joint',
            'left_wrist_roll_joint',
            'right_shoulder_pitch_joint',
            'right_shoulder_roll_joint',
            'right_shoulder_yaw_joint',
            'right_elbow_joint',
            'right_wrist_roll_joint',
        ]
        self.ee_offset = np.array([0.1365, 0.0, 0.0]).T
        self.g1_head_to_waist_offset = np.array([0.15, 0.0, 0.45]).T
        self.x200_head_to_torso_offset = np.array([0.0, 0.0, 0.388]).T
        self.arm_scale_factor = 0.95
        self.workspace_min = np.array([0.10, -0.75, -0.10]).T
        self.workspace_max = np.array([0.75, 0.75, 0.75]).T

        if os.path.exists(self.cache_path) and (not self.Visualization):
            logger_mp.info(f"[X200_ArmIK] >>> Loading cached robot model: {self.cache_path}")
            self.robot, self.reduced_robot = self.load_cache()
        else:
            logger_mp.info("[X200_ArmIK] >>> Loading URDF (slow)...")
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)
            self.mixed_jointsToLockIDs = [
                joint_name
                for joint_name in self.robot.model.names
                if joint_name not in self.arm_joint_names and joint_name not in {'universe', 'root_joint'}
            ]
            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=self.mixed_jointsToLockIDs,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )

            self.reduced_robot.model.addFrame(
                pin.Frame(
                    'L_ee',
                    self.reduced_robot.model.getJointId('left_wrist_roll_joint'),
                    pin.SE3(np.eye(3), self.ee_offset),
                    pin.FrameType.OP_FRAME,
                )
            )
            self.reduced_robot.model.addFrame(
                pin.Frame(
                    'R_ee',
                    self.reduced_robot.model.getJointId('right_wrist_roll_joint'),
                    pin.SE3(np.eye(3), self.ee_offset),
                    pin.FrameType.OP_FRAME,
                )
            )

            if not os.path.exists(self.cache_path):
                self.save_cache()
                logger_mp.info(f">>> Cache saved to {self.cache_path}")

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym('q', self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym('tf_l', 4, 4)
        self.cTf_r = casadi.SX.sym('tf_r', 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.L_hand_id = self.reduced_robot.model.getFrameId('L_ee')
        self.R_hand_id = self.reduced_robot.model.getFrameId('R_ee')

        self.translational_error = casadi.Function(
            'translational_error',
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            'rotational_error',
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T),
                    cpin.log3(self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T),
                )
            ],
        )
        self.pointing_error = casadi.Function(
            'pointing_error',
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].rotation[:, 0] - self.cTf_l[:3, 0],
                    self.cdata.oMf[self.R_hand_id].rotation[:, 0] - self.cTf_r[:3, 0],
                )
            ],
        )
        self.roll_assist_error = casadi.Function(
            'roll_assist_error',
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].rotation[:, 1] - self.cTf_l[:3, 1],
                    self.cdata.oMf[self.R_hand_id].rotation[:, 1] - self.cTf_r[:3, 1],
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)

        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.pointing_cost = casadi.sumsqr(self.pointing_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.roll_assist_cost = casadi.sumsqr(self.roll_assist_error(self.var_q, self.param_tf_l, self.param_tf_r))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(100 * self.translational_cost + 1.2 * self.pointing_cost + 1.5 * self.roll_assist_cost + 0.005 * self.regularization_cost + 0.005 * self.smooth_cost)

        opts = {
            'expand': True,
            'detect_simple_bounds': True,
            'calc_lam_p': False,
            'print_time': False,
            'ipopt.sb': 'yes',
            'ipopt.print_level': 0,
            'ipopt.max_iter': 30,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 5e-4,
            'ipopt.acceptable_iter': 5,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.derivative_test': 'none',
            'ipopt.jacobian_approximation': 'exact',
        }
        self.opti.solver('ipopt', opts)

        self.init_data = np.zeros(self.reduced_robot.model.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([1.0]), 10)
        self.vis = None
        self._last_debug_ts = 0.0

        if self.Visualization:
            self.vis = MeshcatVisualizer(self.reduced_robot.model, self.reduced_robot.collision_model, self.reduced_robot.visual_model)
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel('pinocchio')
            self.vis.displayFrames(True, frame_ids=[self.L_hand_id, self.R_hand_id], axis_length=0.15, axis_width=5)
            self.vis.display(pin.neutral(self.reduced_robot.model))

            frame_viz_names = ['L_ee_target', 'R_ee_target']
            frame_axis_positions = (
                np.array([[0, 0, 0], [1, 0, 0],
                          [0, 0, 0], [0, 1, 0],
                          [0, 0, 0], [0, 0, 1]]).astype(np.float32).T
            )
            frame_axis_colors = (
                np.array([[1, 0, 0], [1, 0.6, 0],
                          [0, 1, 0], [0.6, 1, 0],
                          [0, 0, 1], [0, 0.6, 1]]).astype(np.float32).T
            )
            axis_length = 0.1
            axis_width = 20
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * frame_axis_positions,
                            color=frame_axis_colors,
                        ),
                        mg.LineBasicMaterial(
                            linewidth=axis_width,
                            vertexColors=True,
                        ),
                    )
                )

    def save_cache(self):
        data = {
            'robot_model': self.robot.model,
            'reduced_model': self.reduced_robot.model,
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(data, f)

    def load_cache(self):
        with open(self.cache_path, 'rb') as f:
            data = pickle.load(f)

        robot = pin.RobotWrapper()
        robot.model = data['robot_model']
        robot.data = robot.model.createData()

        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data['reduced_model']
        reduced_robot.data = reduced_robot.model.createData()
        return robot, reduced_robot

    def scale_arms(self, human_left_pose, human_right_pose, human_arm_length=0.60, robot_arm_length=0.38):
        scale_factor = robot_arm_length / human_arm_length
        robot_left_pose = human_left_pose.copy()
        robot_right_pose = human_right_pose.copy()
        robot_left_pose[:3, 3] *= scale_factor
        robot_right_pose[:3, 3] *= scale_factor
        return robot_left_pose, robot_right_pose

    def preprocess_targets(self, left_wrist, right_wrist):
        left_target = left_wrist.copy()
        right_target = right_wrist.copy()
        for target in (left_target, right_target):
            head_relative_translation = target[:3, 3] - self.g1_head_to_waist_offset
            target[:3, 3] = head_relative_translation * self.arm_scale_factor + self.x200_head_to_torso_offset
            target[:3, 3] = np.clip(target[:3, 3], self.workspace_min, self.workspace_max)
        return left_target, right_target

    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        if current_lr_arm_motor_q is not None:
            self.init_data = current_lr_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)

        left_wrist, right_wrist = self.preprocess_targets(left_wrist, right_wrist)

        if self.Visualization:
            self.vis.viewer['L_ee_target'].set_transform(left_wrist)
            self.vis.viewer['R_ee_target'].set_transform(right_wrist)

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            self.opti.solve()
            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q
            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            if self.Visualization:
                self.vis.display(sol_q)

            now = time.time()
            if now - self._last_debug_ts >= 1.0:
                self._last_debug_ts = now
                current_wrists = None if current_lr_arm_motor_q is None else np.round(current_lr_arm_motor_q[[4, 9]], 3).tolist()
                logger_mp.info(
                    f"[X200_ArmIK] target_xyz_L={np.round(left_wrist[:3, 3], 3).tolist()} "
                    f"target_xyz_R={np.round(right_wrist[:3, 3], 3).tolist()} "
                    f"sol_wrist_roll={np.round(sol_q[[4, 9]], 3).tolist()} "
                    f"current_wrist_roll={current_wrists} "
                    f"delta_norm={round(float(np.linalg.norm(sol_q - current_lr_arm_motor_q)), 3) if current_lr_arm_motor_q is not None else -1.0}"
                )

            return sol_q, sol_tauff

        except Exception as e:
            logger_mp.error(f"ERROR in convergence, plotting debug info.{e}")
            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q
            sol_tauff = pin.rnea(self.reduced_robot.model, self.reduced_robot.data, sol_q, v, np.zeros(self.reduced_robot.model.nv))

            logger_mp.error(f"sol_q:{sol_q} \nmotorstate: \n{current_lr_arm_motor_q} \nleft_pose: \n{left_wrist} \nright_pose: \n{right_wrist}")
            if self.Visualization:
                self.vis.display(sol_q)

            return current_lr_arm_motor_q, np.zeros(self.reduced_robot.model.nv)
