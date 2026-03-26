import numpy as np
import threading
import time
from enum import IntEnum

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd, LowState_ as hg_LowState
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

import logging_mp
from teleop.robot_control.robot_arm import DataBuffer, kTopicLowCommand_Debug, kTopicLowCommand_Motion, kTopicLowState

logger_mp = logging_mp.getLogger(__name__)

X200_Num_Motors = 35


class MotorState:
    def __init__(self):
        self.q = None
        self.dq = None


class X200_LowState:
    def __init__(self):
        self.motor_state = [MotorState() for _ in range(X200_Num_Motors)]


class X200_JointArmIndex(IntEnum):
    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19

    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26


class X200_JointIndex(IntEnum):
    kLeftHipPitch = 0
    kLeftHipRoll = 1
    kLeftHipYaw = 2
    kLeftKnee = 3
    kLeftAnklePitch = 4
    kLeftAnkleRoll = 5

    kRightHipPitch = 6
    kRightHipRoll = 7
    kRightHipYaw = 8
    kRightKnee = 9
    kRightAnklePitch = 10
    kRightAnkleRoll = 11

    kTorsoYaw = 12
    kTorsoRollNotUsed = 13
    kTorsoPitchNotUsed = 14

    kLeftShoulderPitch = 15
    kLeftShoulderRoll = 16
    kLeftShoulderYaw = 17
    kLeftElbow = 18
    kLeftWristRoll = 19
    kLeftWristPitchNotUsed = 20
    kLeftWristYawNotUsed = 21

    kRightShoulderPitch = 22
    kRightShoulderRoll = 23
    kRightShoulderYaw = 24
    kRightElbow = 25
    kRightWristRoll = 26
    kRightWristPitchNotUsed = 27
    kRightWristYawNotUsed = 28

    kNotUsedJoint0 = 29
    kNotUsedJoint1 = 30
    kNotUsedJoint2 = 31
    kNotUsedJoint3 = 32
    kNotUsedJoint4 = 33
    kNotUsedJoint5 = 34


class X200_ArmController:
    def __init__(self, motion_mode=False, simulation_mode=False):
        logger_mp.info("Initialize X200_ArmController...")
        self.simulation_mode = simulation_mode
        self.motion_mode = motion_mode

        self.q_target = np.zeros(10)
        self.tauff_target = np.zeros(10)

        self.kp_high = 300.0
        self.kd_high = 3.0
        self.kp_low = 80.0
        self.kd_low = 3.0
        self.kp_wrist = 40.0
        self.kd_wrist = 1.5

        self.all_motor_q = None
        self.arm_velocity_limit = 20.0
        self.control_dt = 1.0 / 250.0

        self._speed_gradual_max = False
        self._gradual_start_time = None
        self._gradual_time = None

        if self.motion_mode:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Motion, hg_LowCmd)
        else:
            self.lowcmd_publisher = ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()
        self.lowstate_buffer = DataBuffer()

        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.daemon = True
        self.subscribe_thread.start()

        while not self.lowstate_buffer.GetData():
            time.sleep(0.1)
            logger_mp.warning("[X200_ArmController] Waiting to subscribe dds...")
        logger_mp.info("[X200_ArmController] Subscribe dds ok.")

        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0
        self.msg.mode_machine = self.get_mode_machine()

        self.all_motor_q = self.get_current_motor_q()
        logger_mp.info(f"Current all body motor state q:\n{self.all_motor_q}\n")
        logger_mp.info(f"Current two arms motor state q:\n{self.get_current_dual_arm_q()}\n")
        logger_mp.info("Lock all joints except X200 two arms...")

        arm_indices = set(member.value for member in X200_JointArmIndex)
        for motor_id in X200_JointIndex:
            self.msg.motor_cmd[motor_id].mode = 1
            if motor_id.value in arm_indices:
                if self._Is_wrist_motor(motor_id):
                    self.msg.motor_cmd[motor_id].kp = self.kp_wrist
                    self.msg.motor_cmd[motor_id].kd = self.kd_wrist
                else:
                    self.msg.motor_cmd[motor_id].kp = self.kp_low
                    self.msg.motor_cmd[motor_id].kd = self.kd_low
            else:
                if self._Is_weak_motor(motor_id):
                    self.msg.motor_cmd[motor_id].kp = self.kp_low
                    self.msg.motor_cmd[motor_id].kd = self.kd_low
                else:
                    self.msg.motor_cmd[motor_id].kp = self.kp_high
                    self.msg.motor_cmd[motor_id].kd = self.kd_high
            self.msg.motor_cmd[motor_id].q = self.all_motor_q[motor_id]
        logger_mp.info("Lock OK!")

        self.publish_thread = threading.Thread(target=self._ctrl_motor_state)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        logger_mp.info("Initialize X200_ArmController OK!")

    def _subscribe_motor_state(self):
        while True:
            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = X200_LowState()
                for motor_id in range(X200_Num_Motors):
                    lowstate.motor_state[motor_id].q = msg.motor_state[motor_id].q
                    lowstate.motor_state[motor_id].dq = msg.motor_state[motor_id].dq
                self.lowstate_buffer.SetData(lowstate)
            time.sleep(0.002)

    def clip_arm_q_target(self, target_q, velocity_limit):
        current_q = self.get_current_dual_arm_q()
        delta = target_q - current_q
        motion_scale = np.max(np.abs(delta)) / (velocity_limit * self.control_dt)
        return current_q + delta / max(motion_scale, 1.0)

    def _ctrl_motor_state(self):
        if self.motion_mode:
            self.msg.motor_cmd[X200_JointIndex.kNotUsedJoint0].q = 1.0

        while True:
            start_time = time.time()
            with self.ctrl_lock:
                arm_q_target = self.q_target
                arm_tauff_target = self.tauff_target

            if self.simulation_mode:
                cliped_arm_q_target = arm_q_target
            else:
                cliped_arm_q_target = self.clip_arm_q_target(arm_q_target, velocity_limit=self.arm_velocity_limit)

            for idx, motor_id in enumerate(X200_JointArmIndex):
                self.msg.motor_cmd[motor_id].q = cliped_arm_q_target[idx]
                self.msg.motor_cmd[motor_id].dq = 0
                self.msg.motor_cmd[motor_id].tau = arm_tauff_target[idx]

            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)

            if self._speed_gradual_max is True:
                t_elapsed = start_time - self._gradual_start_time
                self.arm_velocity_limit = 20.0 + (10.0 * min(1.0, t_elapsed / 5.0))

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, self.control_dt - all_t_elapsed)
            time.sleep(sleep_time)

    def ctrl_dual_arm(self, q_target, tauff_target):
        with self.ctrl_lock:
            self.q_target = q_target
            self.tauff_target = tauff_target

    def get_mode_machine(self):
        return self.lowstate_subscriber.Read().mode_machine

    def get_current_motor_q(self):
        return np.array([self.lowstate_buffer.GetData().motor_state[motor_id].q for motor_id in X200_JointIndex])

    def get_current_dual_arm_q(self):
        return np.array([self.lowstate_buffer.GetData().motor_state[motor_id].q for motor_id in X200_JointArmIndex])

    def get_current_dual_arm_dq(self):
        return np.array([self.lowstate_buffer.GetData().motor_state[motor_id].dq for motor_id in X200_JointArmIndex])

    def ctrl_dual_arm_go_home(self):
        logger_mp.info("[X200_ArmController] ctrl_dual_arm_go_home start...")
        max_attempts = 100
        current_attempts = 0
        with self.ctrl_lock:
            self.q_target = np.zeros(10)
        tolerance = 0.05
        while current_attempts < max_attempts:
            current_q = self.get_current_dual_arm_q()
            if np.all(np.abs(current_q) < tolerance):
                if self.motion_mode:
                    for weight in np.linspace(1, 0, num=101):
                        self.msg.motor_cmd[X200_JointIndex.kNotUsedJoint0].q = weight
                        time.sleep(0.02)
                logger_mp.info("[X200_ArmController] both arms have reached the home position.")
                break
            current_attempts += 1
            time.sleep(0.05)

    def speed_gradual_max(self, t=5.0):
        self._gradual_start_time = time.time()
        self._gradual_time = t
        self._speed_gradual_max = True

    def speed_instant_max(self):
        self.arm_velocity_limit = 30.0

    def _Is_weak_motor(self, motor_index):
        weak_motors = [
            X200_JointIndex.kLeftAnklePitch.value,
            X200_JointIndex.kRightAnklePitch.value,
            X200_JointIndex.kLeftShoulderPitch.value,
            X200_JointIndex.kLeftShoulderRoll.value,
            X200_JointIndex.kLeftShoulderYaw.value,
            X200_JointIndex.kLeftElbow.value,
            X200_JointIndex.kRightShoulderPitch.value,
            X200_JointIndex.kRightShoulderRoll.value,
            X200_JointIndex.kRightShoulderYaw.value,
            X200_JointIndex.kRightElbow.value,
        ]
        return motor_index.value in weak_motors

    def _Is_wrist_motor(self, motor_index):
        wrist_motors = [
            X200_JointIndex.kLeftWristRoll.value,
            X200_JointIndex.kRightWristRoll.value,
        ]
        return motor_index.value in wrist_motors
