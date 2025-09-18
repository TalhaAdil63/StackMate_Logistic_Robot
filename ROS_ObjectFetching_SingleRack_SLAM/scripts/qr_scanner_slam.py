#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import actionlib
from pyzbar.pyzbar import decode
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from yahboomcar_msgs.msg import ArmJoint
import time

# === SimplePID class from line-following code ===
class SimplePID:
    def __init__(self, target, Kp, Ki, Kd):
        self.target = np.array(target, dtype=float)
        self.Kp = np.array(Kp, dtype=float)
        self.Ki = np.array(Ki, dtype=float)
        self.Kd = np.array(Kd, dtype=float)
        self.error = np.zeros_like(self.target)
        self.error_last = np.zeros_like(self.target)
        self.error_sum = np.zeros_like(self.target)
        self.output = np.zeros_like(self.target)

    def update(self, feedback):
        feedback = np.array(feedback, dtype=float)
        self.error = self.target - feedback
        self.error_sum += self.error
        self.error_sum = np.clip(self.error_sum, -1.0, 1.0)
        self.output = (self.Kp * self.error +
                       self.Ki * self.error_sum +
                       self.Kd * (self.error - self.error_last))
        self.error_last = self.error.copy()
        return self.output, self.error

# === Define known rack goals ===
rack_goals = {
    "B2": {'x': -1.07438886166, 'y': -1.99986743927, 'qx': 0.0, 'qy': 0.0, 'qz': -0.703451754601, 'qw': 0.710743011889},
    "B1": {'x': -1.07438886166, 'y': -1.99986743927, 'qx': 0.0, 'qy': 0.0, 'qz': -0.703451754601, 'qw': 0.710743011889},
    "A1": {'x': -1.344, 'y': -3.89,  'qx': 0.0, 'qy': 0.0, 'qz': -0.5780, 'qw': 0.6789}
}

# === Define initial position ===
start_pose = {
    'x': 0.685157835484,
    'y': -0.374455153942,
    'qx': 0.0,
    'qy': 0.0,
    'qz': 0.00563939279311,
    'qw': 0.999984098498
}

class ROSCtrl:
    def __init__(self):
        self.Joy_active = False
        self.pub_cmdVel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sub_JoyState = rospy.Subscriber('/JoyState', Bool, self.JoyStateCallback)
        self.pub_Arm = rospy.Publisher('/TargetAngle', ArmJoint, queue_size=1000)
        rospy.loginfo("Arm publisher initialized on /TargetAngle")

    def JoyStateCallback(self, msg):
        if not isinstance(msg, Bool): return
        self.Joy_active = msg.data
        self.pub_cmdVel.publish(Twist())

    def cancel(self):
        self.sub_JoyState.unregister()
        self.pub_cmdVel.unregister()
        self.pub_Arm.unregister()

    def pubArm(self, joints, id=10, angle=90, run_time=500):
        armjoint = ArmJoint()
        armjoint.run_time = run_time
        if len(joints) != 0:
            armjoint.joints = joints
        else:
            armjoint.id = id
            armjoint.angle = angle
        self.pub_Arm.publish(armjoint)
        rospy.loginfo("Published arm command: joints={}, id={}, angle={}".format(joints, id, angle))

    def grip_lowerclosed_rack(self):
        self.pubArm([90, 0, 25, 180, 90, 120], run_time=1000)
        rospy.sleep(3)

    def grip_upperclosed_rack(self):
        self.pubArm([97, 97, 97, 0, 90, 120], run_time=1000)
        rospy.sleep(3)

    def pick_uprack(self):
        self.pubArm([97, 97, 97, 0, 90, 120], run_time=1000)
        rospy.sleep(2)

    def close_grip(self):
        self.pubArm([], id=6, angle=120)
        rospy.sleep(3)

    def release_grip(self):
        self.pubArm([], id=6, angle=0)
        rospy.sleep(2)

    def lower_arm_for_box(self):
        rospy.loginfo("Lowering arm for box alignment...")
        self.pubArm([90, 0, 15, 180, 90, 0], run_time=1000)
        rospy.sleep(3)
        rospy.loginfo("Arm lowered for box alignment")

    def move_forward(self, duration, speed):
        rospy.loginfo("Moving Forward for %.2f seconds at speed %.2f" % (duration, speed))
        stop_twist = Twist()
        self.pub_cmdVel.publish(stop_twist)
        rospy.sleep(0.2)
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = 0
        self.pub_cmdVel.publish(twist)
        rospy.sleep(duration)
        self.pub_cmdVel.publish(Twist())
        time.sleep(0.2)
        rospy.loginfo("Move completed!")

    def move_backward(self, duration):
        rospy.loginfo("Moving Backward for %.2f seconds" % duration)
        stop_twist = Twist()
        self.pub_cmdVel.publish(stop_twist)
        rospy.sleep(0.2)
        twist = Twist()
        twist.linear.x = -0.5
        self.pub_cmdVel.publish(twist)
        rospy.sleep(duration)
        self.pub_cmdVel.publish(Twist())
        time.sleep(0.2)
        rospy.loginfo("Move completed!")

    def turn_left(self, duration=1.0, angular_speed=0.5):
        rospy.loginfo("Turning left for %.2f seconds at angular speed %.2f" % (duration, angular_speed))
        stop_twist = Twist()
        self.pub_cmdVel.publish(stop_twist)
        rospy.sleep(0.2)
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = angular_speed
        self.pub_cmdVel.publish(twist)
        rospy.sleep(duration)
        self.pub_cmdVel.publish(Twist())
        time.sleep(0.2)
        rospy.loginfo("Turn completed!")

    def turn_right(self, duration=1.0, angular_speed=-0.5):
        rospy.loginfo("Turning right for %.2f seconds at angular speed %.2f" % (duration, angular_speed))
        stop_twist = Twist()
        self.pub_cmdVel.publish(stop_twist)
        rospy.sleep(0.2)
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = angular_speed
        self.pub_cmdVel.publish(twist)
        rospy.sleep(duration)
        self.pub_cmdVel.publish(Twist())
        time.sleep(0.2)
        rospy.loginfo("Turn completed!")

    def disable_obstacle_avoidance(self):
        rospy.set_param('/move_base/local_costmap/inflation_radius', 0.1)
        rospy.set_param('/move_base/global_costmap/inflation_radius', 0.1)

    def restore_obstacle_avoidance(self):
        rospy.set_param('/move_base/local_costmap/inflation_radius', 0.5)
        rospy.set_param('/move_base/global_costmap/inflation_radius', 0.5)

class QrScannerNode:
    def __init__(self):
        rospy.init_node('qr_scanner_node')
        rospy.loginfo("QR Scanner Node Started")
        self.ros_ctrl = ROSCtrl()
        self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            rospy.logerr("Failed to open arm camera (index 1)")
            raise Exception("Arm camera initialization failed")
        self.cap.set(cv2.CAP_PROP_FOCUS, 50)
        rospy.sleep(3)
        self.active_rack_code = None
        self.task_in_progress = False
        self.grip_closed = False
        self.Buzzer_state = False
        self.pub_initial_pose = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
        self.qr_pid = SimplePID([0, 0], [0.1, 0], [0.05, 0], [0.015, 0])
        self.display_enabled = True
        self.qr_window = "QR_Alignment"
        self.frame_counter = 0
        try:
            cv2.namedWindow(self.qr_window)
        except Exception as e:
            rospy.logerr("Failed to create display window: %s", e)
            self.display_enabled = False
        self.first_adjustment_done = False
        self.x_tolerance = 8  # Define x_tolerance as a class attribute

    def set_initial_pose(self):
        rospy.loginfo("Setting initial pose")
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.pose.position.x = start_pose['x']
        pose_msg.pose.pose.position.y = start_pose['y']
        pose_msg.pose.pose.orientation.x = start_pose['qx']
        pose_msg.pose.pose.orientation.y = start_pose['qy']
        pose_msg.pose.pose.orientation.z = start_pose['qz']
        pose_msg.pose.pose.orientation.w = start_pose['qw']
        pose_msg.pose.covariance = [0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                   0.0, 0.0, 0.0, 0.0, 0.0, 0.06853891945200942]
        self.pub_initial_pose.publish(pose_msg)
        rospy.sleep(1)

    def adjust_distance_and_angle(self):
        rospy.loginfo("Adjusting distance and angle based on QR code area and position...")
        if not self.cap.isOpened():
            rospy.logerr("Arm camera (index 1) not initialized or failed to open! Reinitializing...")
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                rospy.logerr("Arm camera (index 1) reinitialization failed")
                return False, None
            self.cap.set(cv2.CAP_PROP_FOCUS, 50)
        target_area = 33500
        area_tolerance = 3500
        deadband = 750
        x_tolerance = 5  # Local definition for this method
        max_linear_x = 0.05
        max_angular_z = 0.12
        max_lateral_x = 0.03  # Max speed for lateral correction
        timeout = 30.0
        twist = Twist()
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        no_qr_start = None
        start_time = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            if rospy.Time.now().to_sec() - start_time > timeout:
                rospy.logwarn("Adjustment timed out after %s seconds", timeout)
                self.ros_ctrl.pub_cmdVel.publish(Twist())
                return False, None
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                rospy.logwarn("Arm camera (index 1) read failed or invalid frame!")
                continue
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            enhanced_frame = clahe.apply(gray_frame)
            blurred_frame = cv2.GaussianBlur(enhanced_frame, (5, 5), 0)
            qr_codes = decode(blurred_frame)
            display_frame = frame.copy()
            center_x = 320
            if qr_codes:
                qr = qr_codes[0]
                qr_data = qr.data.decode('utf-8').strip()
                points = np.array(qr.polygon, dtype=np.int32).reshape(-1, 1, 2)
                center_x_qr = int(np.mean(points[:, 0, 0]))
                qr_area = qr.rect.width * qr.rect.height
                dx = center_x_qr - center_x
                area_error = qr_area - target_area
                rospy.loginfo("QR detected: area=%d, dx=%d", qr_area, dx)
                if self.display_enabled:
                    try:
                        cv2.polylines(display_frame, [points], True, (0, 255, 0), 2)
                        cv2.circle(display_frame, (center_x_qr, int(np.mean(points[:, 0, 1]))), 5, (0, 255, 255), -1)
                        cv2.putText(display_frame, "QR: %s, Area: %d" % (qr_data, qr_area), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        cv2.imshow(self.qr_window, display_frame)
                        cv2.waitKey(1)
                    except Exception as e:
                        rospy.logerr("Error displaying QR frame: %s", e)
                        self.display_enabled = False
                if not self.first_adjustment_done:
                    angular_z, _ = self.qr_pid.update([dx, 0])
                    angular_z = max(min(angular_z[0], max_angular_z), -max_angular_z)
                    if abs(area_error) > deadband:
                        linear_x = -0.001 * area_error
                        linear_x = max(min(linear_x, max_linear_x), -max_linear_x)
                    else:
                        linear_x = 0.0
                    twist.linear.x = linear_x
                    twist.angular.z = angular_z
                    self.ros_ctrl.pub_cmdVel.publish(twist)
                    rospy.loginfo("First adjustment: linear.x=%f, angular.z=%f, area=%d", linear_x, angular_z, qr_area)
                    self.first_adjustment_done = True
                    rospy.sleep(1.0)
                    continue
                if qr_area < 10000:
                    rospy.logwarn("Partial QR detection (area < 10000), moving forward to reposition...")
                    self.ros_ctrl.move_forward(0.5, 0.1)
                    continue
                if 30000 <= qr_area <= 37000 and abs(dx) <= x_tolerance:
                    rospy.loginfo("Target area (30000-37000) reached, applying final correction...")
                    self.ros_ctrl.pub_cmdVel.publish(Twist())
                    # Correct orientation
                    if dx != 0:
                        correction_angle = np.arctan2(dx, 240)
                        rospy.loginfo("Applying orientation correction: %.4f radians", correction_angle)
                        twist.angular.z = 0.1 * np.sign(correction_angle)
                        self.ros_ctrl.pub_cmdVel.publish(twist)
                        rospy.sleep(0.5)
                        self.ros_ctrl.pub_cmdVel.publish(Twist())
                    # Correct lateral position
                    if abs(dx) > x_tolerance / 2:  # Allow small deviation, correct if significant
                        lateral_x = -0.001 * dx  # Move left if dx positive (QR to right), right if negative
                        lateral_x = max(min(lateral_x, max_lateral_x), -max_lateral_x)
                        twist.linear.x = lateral_x
                        twist.angular.z = 0.0
                        self.ros_ctrl.pub_cmdVel.publish(twist)
                        rospy.sleep(abs(dx) * 0.01)  # Duration proportional to offset
                        self.ros_ctrl.pub_cmdVel.publish(Twist())
                        rospy.loginfo("Applied lateral correction: dx=%d, lateral_x=%f", dx, lateral_x)
                    return True, qr_data
                if abs(area_error) > deadband or (30000 <= qr_area <= 37000 and abs(area_error) > 200):
                    linear_x = -0.001 * area_error
                    linear_x = max(min(linear_x, max_linear_x), -max_linear_x)
                else:
                    linear_x = 0.0
                angular_z, _ = self.qr_pid.update([dx, 0])
                angular_z = max(min(angular_z[0], max_angular_z / 2), -max_angular_z / 2)
                twist.linear.x = linear_x
                twist.angular.z = angular_z
                self.ros_ctrl.pub_cmdVel.publish(twist)
                rospy.loginfo("Adjusting: linear.x=%f, angular.z=%f, area=%d", linear_x, angular_z, qr_area)
                no_qr_start = None
            else:
                if not self.first_adjustment_done:
                    self.first_adjustment_done = True
                if no_qr_start is None:
                    no_qr_start = rospy.Time.now().to_sec()
                elif rospy.Time.now().to_sec() - no_qr_start >= 5.0:
                    rospy.loginfo("No QR code detected for 5 seconds, moving forward and turning...")
                    self.ros_ctrl.move_forward(1.4, 0.2)
                    rospy.sleep(3)
                    self.ros_ctrl.turn_right(duration=1.1, angular_speed=-0.5)
                    rospy.sleep(1)
                    self.ros_ctrl.turn_left(duration=1.1, angular_speed=0.5)
                    no_qr_start = rospy.Time.now().to_sec()
                rospy.loginfo("No QR code detected in frame")
                if self.display_enabled:
                    try:
                        cv2.putText(display_frame, "No QR code detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow(self.qr_window, display_frame)
                        cv2.waitKey(1)
                    except Exception as e:
                        rospy.logerr("Error displaying QR frame: %s", e)
                        self.display_enabled = False
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.ros_ctrl.pub_cmdVel.publish(twist)
            self.frame_counter += 1
            if self.display_enabled and self.frame_counter % 5 == 0:
                try:
                    cv2.rectangle(display_frame, (center_x - x_tolerance, 210), (center_x + x_tolerance, 270), (255, 0, 0), 2)
                    cv2.imshow(self.qr_window, display_frame)
                    cv2.waitKey(1)
                except Exception as e:
                    rospy.logerr("Error displaying QR frame: %s", e)
                    self.display_enabled = False
            rospy.sleep(0.05)
        self.ros_ctrl.pub_cmdVel.publish(Twist())
        return False, None

    def align_with_qr(self):
        rospy.loginfo("[Scanning] Starting QR code alignment with arm camera...")
        if self.cap is None or not self.cap.isOpened():
            rospy.logerr("Arm camera (index 1) not initialized or failed to open! Reinitializing...")
            self.cap = cv2.VideoCapture(1)
            if not self.cap.isOpened():
                rospy.logerr("Arm camera (index 1) reinitialization failed")
                return False, None
            self.cap.set(cv2.CAP_PROP_FOCUS, 50)
        x_tolerance = 8  # Local definition for this method
        max_angular_z = 0.2
        stabilization_delay = 2.5
        timeout = 120.0
        max_retries = 1
        for attempt in range(max_retries + 1):
            start_time = rospy.Time.now().to_sec()
            stabilization_start = None
            no_qr_start = None
            coarse_search_direction = 1
            twist = Twist()
            while not rospy.is_shutdown():
                if rospy.Time.now().to_sec() - start_time > timeout:
                    rospy.logwarn("QR alignment timed out after %s seconds", timeout)
                    if attempt < max_retries:
                        rospy.loginfo("Retrying arm camera (index 1) initialization...")
                        self.cap.release()
                        self.cap = cv2.VideoCapture(1)
                        if not self.cap.isOpened():
                            rospy.logerr("Arm camera (index 1) reinitialization failed")
                        self.cap.set(cv2.CAP_PROP_FOCUS, 50)
                        break
                    return False, None
                if not self.cap.isOpened():
                    rospy.logerr("Arm camera (index 1) closed unexpectedly! Reinitializing...")
                    self.cap.release()
                    self.cap = cv2.VideoCapture(1)
                    if not self.cap.isOpened():
                        rospy.logerr("Arm camera (index 1) reinitialization failed")
                        return False, None
                    self.cap.set(cv2.CAP_PROP_FOCUS, 50)
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    rospy.logwarn("Arm camera (index 1) read failed or invalid frame! Shape: %s", str(frame.shape) if frame is not None else "None")
                    if self.display_enabled:
                        try:
                            display_frame = frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(display_frame, "Camera read failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv2.imshow(self.qr_window, display_frame)
                            cv2.waitKey(1)
                        except Exception as e:
                            rospy.logerr("Error displaying QR frame: %s", e)
                            self.display_enabled = False
                    continue
                rospy.loginfo("Frame captured: Shape=%s", str(frame.shape))
                qr_codes = decode(frame)
                center_x = 320
                qr_detected = False
                dx = 0
                qr_data = None
                display_frame = frame.copy()
                if qr_codes:
                    qr_detected = True
                    no_qr_start = None
                    qr = qr_codes[0]
                    qr_data = qr.data.decode('utf-8').strip() if qr.data else "No data"
                    self.active_rack_code = qr_data
                    rospy.loginfo("QR code detected: Value=%s, Codes found=%d", qr_data, len(qr_codes))
                    points = np.array(qr.polygon, dtype=np.int32).reshape(-1, 1, 2) if qr.polygon else np.array([])
                    if points.size > 0:
                        center_x_box = int(np.mean(points[:, 0, 0]))
                        center_y_box = int(np.mean(points[:, 0, 1]))
                        dx = center_x_box - center_x
                        qr_area = qr.rect.width * qr.rect.height
                        rospy.loginfo("QR boundaries detected, dx: %d, Center: (%d, %d), Area: %d", dx, center_x_box, center_y_box, qr_area)
                    else:
                        rospy.logwarn("No QR boundaries detected despite code detection")
                    if self.display_enabled:
                        try:
                            if points.size > 0:
                                cv2.polylines(display_frame, [points], True, (0, 255, 0), 2)
                                cv2.circle(display_frame, (center_x_box, center_y_box), 5, (0, 255, 255), -1)
                                cv2.line(display_frame, (center_x, 240), (center_x_box, center_y_box), (255, 255, 0), 2)
                            cv2.putText(display_frame, "QR: %s" % qr_data, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                        except Exception as e:
                            rospy.logerr("Error annotating QR frame: %s", e)
                            self.display_enabled = False
                    if qr_area < 10000 and abs(dx) <= x_tolerance:  # Handle distant but centered QR
                        rospy.loginfo("Distant QR detected (area < 10000) and centered, proceeding to grab...")
                        return True, qr_data
                    if abs(dx) <= x_tolerance:
                        if stabilization_start is None:
                            stabilization_start = rospy.Time.now().to_sec()
                            rospy.loginfo("[Centered] Starting %s-second stabilization...", stabilization_delay)
                            twist.angular.z = 0.0
                            self.ros_ctrl.pub_cmdVel.publish(twist)
                        elif rospy.Time.now().to_sec() - stabilization_start >= stabilization_delay:
                            rospy.loginfo("Stabilization complete. QR code centered with data: %s", qr_data)
                            return True, qr_data
                    else:
                        stabilization_start = None
                        error = dx
                        angular_z, _ = self.qr_pid.update([error, 0])
                        angular_z = max(min(angular_z[0], max_angular_z), -max_angular_z)
                        twist.angular.z = angular_z
                        rospy.loginfo("Adjusting angular - dx: %d, angular.z: %s", dx, twist.angular.z)
                        self.ros_ctrl.pub_cmdVel.publish(twist)
                else:
                    stabilization_start = None
                    if no_qr_start is None:
                        no_qr_start = rospy.Time.now().to_sec()
                    elif rospy.Time.now().to_sec() - no_qr_start >= 5.0:
                        rospy.logwarn("No QR code detected for 5 seconds. Starting coarse search...")
                        twist.angular.z = 0.1 * coarse_search_direction
                        self.ros_ctrl.pub_cmdVel.publish(twist)
                        rospy.sleep(3.0)
                        twist.angular.z = 0.0
                        self.ros_ctrl.pub_cmdVel.publish(twist)
                        coarse_search_direction *= -1
                        no_qr_start = rospy.Time.now().to_sec()
                    rospy.loginfo("No QR code detected in frame.")
                    if self.display_enabled:
                        try:
                            cv2.putText(display_frame, "No QR code detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        except Exception as e:
                            rospy.logerr("Error annotating QR frame: %s", e)
                            self.display_enabled = False
                    if twist.angular.z != 0.0:
                        twist.angular.z = 0.0
                        self.ros_ctrl.pub_cmdVel.publish(twist)
                self.frame_counter += 1
                if self.display_enabled and self.frame_counter % 5 == 0:
                    try:
                        frame_time = rospy.Time.now().to_sec() - start_time
                        cv2.putText(display_frame, "FPS: %.2f" % (1.0 / (frame_time + 0.001)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 200), 1)
                        cv2.rectangle(display_frame, (center_x - x_tolerance, 210), (center_x + x_tolerance, 270), (255, 0, 0), 2)
                        cv2.imshow(self.qr_window, display_frame)
                        cv2.waitKey(1)
                    except Exception as e:
                        rospy.logerr("Error displaying QR frame: %s", e)
                        self.display_enabled = False
            if attempt == max_retries:
                break
        self.ros_ctrl.pub_cmdVel.publish(Twist())
        rospy.loginfo("[Failed] QR alignment failed or timed out.")
        return False, None

    def send_goal(self, x, y, qx, qy, qz, qw):
        rospy.loginfo("Sending goal to: x={}, y={}, orientation=({}, {}, {}, {})".format(x, y, qx, qy, qz, qw))
        client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        client.wait_for_server()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        goal.target_pose.pose.orientation.x = qx
        goal.target_pose.pose.orientation.y = qy
        goal.target_pose.pose.orientation.z = qz
        goal.target_pose.pose.orientation.w = qw
        client.send_goal(goal)
        wait = client.wait_for_result(rospy.Duration(60.0))
        if not wait:
            rospy.logwarn("Navigation timed out after 60 seconds")
            client.cancel_goal()
            return False
        state = client.get_state()
        result = client.get_result()
        if state == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("Navigation complete. Result: %s", result)
            return True
        else:
            rospy.logerr("Navigation failed with state: %d", state)
            return False

    def main(self):
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries and not rospy.is_shutdown():
            self.set_initial_pose()
            if self.send_goal(start_pose['x'], start_pose['y'], start_pose['qx'], start_pose['qy'], start_pose['qz'], start_pose['qw']):
                break
            else:
                rospy.logwarn("Navigation to initial pose failed, retrying (%d/%d)", retry_count + 1, max_retries)
                retry_count += 1
                if retry_count == max_retries:
                    rospy.logerr("Failed to reach initial pose after %d retries, aborting", max_retries)
                    self.cleanup()
                    return
                rospy.sleep(2.0)  # Wait before retrying
        while not rospy.is_shutdown():
            if not self.task_in_progress:
                self.task_in_progress = True
                rospy.loginfo("Reached container, initiating box handling sequence...")
                self.ros_ctrl.pub_cmdVel.publish(Twist())
                rospy.sleep(1.0)
                self.Buzzer_state = True
                rospy.sleep(0.5)
                if not self.grip_closed:
                    self.ros_ctrl.lower_arm_for_box()
                    rospy.sleep(3)
                    self.first_adjustment_done = False
                    adjust_success, qr_data = self.adjust_distance_and_angle()
                    if adjust_success:
                        align_success, qr_data = self.align_with_qr()
                        if align_success:
                            self.active_rack_code = qr_data
                            rospy.sleep(3)
                            rospy.loginfo("Moving forward for precise box grabbing...")
                            self.ros_ctrl.move_forward(1.04, 0.2)
                            rospy.sleep(3)
                            self.ros_ctrl.close_grip()
                            rospy.sleep(3)
                            self.grip_closed = True
                        else:
                            rospy.logwarn("Failed to align with QR code, retrying in next cycle")
                            self.task_in_progress = False
                            continue
                    else:
                        rospy.logwarn("Failed to adjust distance and angle, retrying in next cycle")
                        self.task_in_progress = False
                        continue
                if self.grip_closed:
                    rospy.loginfo("Moving backward after grabbing box...")
                    self.ros_ctrl.move_backward(0.8)
                    rospy.sleep(3)
                    self.ros_ctrl.pick_uprack()
                    rospy.loginfo("Raising arm to default position for navigation...")
                    rospy.sleep(3)
                    if self.active_rack_code in rack_goals:
                        goal = rack_goals[self.active_rack_code]
                        rospy.loginfo("Navigating to rack for QR: %s", self.active_rack_code)
                        if self.send_goal(goal['x'], goal['y'], goal['qx'], goal['qy'], goal['qz'], goal['qw']):
                            if self.active_rack_code == "B1":
                                self.ros_ctrl.grip_lowerclosed_rack()
                                rospy.loginfo("Lowering arm for B1 box placement on LOWER shelf...")
                            elif self.active_rack_code == "B2":
                                self.ros_ctrl.grip_upperclosed_rack()
                                rospy.loginfo("Positioning arm for B2 box placement on UPPER shelf...")
                            else:
                                self.ros_ctrl.grip_lowerclosed_rack()
                                rospy.logwarn("Unknown QR code value: %s, defaulting to LOWER shelf", self.active_rack_code)
                            rospy.sleep(3)
                            if self.active_rack_code == "B1":
                                rospy.loginfo("Placing box B1 on LOWER shelf...")
                                self.ros_ctrl.move_forward(2.34, 0.2)
                            elif self.active_rack_code == "B2":
                                rospy.loginfo("Placing box B2 on UPPER shelf...")
                                self.ros_ctrl.move_forward(3.45, 0.14)
                            else:
                                rospy.loginfo("Placing box on default LOWER shelf...")
                                self.ros_ctrl.move_forward(2.9, 0.2)
                            rospy.sleep(3)
                            self.ros_ctrl.release_grip()
                            rospy.sleep(3)
                            self.grip_closed = False
                            self.active_rack_code = None
                            self.ros_ctrl.move_backward(0.8)
                            rospy.sleep(3)
                        else:
                            rospy.logwarn("Failed to navigate to rack, retrying in next cycle")
                            self.task_in_progress = False
                            continue
                    else:
                        rospy.logwarn("QR code %s not found in rack_goals, skipping placement", self.active_rack_code)
                        self.grip_closed = False
                        self.active_rack_code = None
                    self.ros_ctrl.pick_uprack()
                    rospy.sleep(3)
                    if not self.send_goal(start_pose['x'], start_pose['y'], start_pose['qx'], start_pose['qy'], start_pose['qz'], start_pose['qw']):
                        rospy.logwarn("Failed to return to container, attempting realignment...")
                    self.ros_ctrl.lower_arm_for_box()
                    rospy.sleep(3)
                    self.first_adjustment_done = False
                    adjust_success, qr_data = self.adjust_distance_and_angle()
                    if adjust_success:
                        align_success, qr_data = self.align_with_qr()
                        if align_success:
                            self.active_rack_code = qr_data
                            rospy.sleep(3)
                            rospy.loginfo("Moving forward for precise box grabbing...")
                            self.ros_ctrl.move_forward(1.04, 0.2)
                            rospy.sleep(3)
                            self.ros_ctrl.close_grip()
                            rospy.sleep(3)
                            self.grip_closed = True
                        else:
                            rospy.logwarn("Failed to align with QR code, retrying in next cycle")
                            self.task_in_progress = False
                            continue
                    else:
                        rospy.logwarn("Failed to adjust distance and angle, retrying in next cycle")
                        self.task_in_progress = False
                        continue
                    # Re-initialize pose and correct deviation precisely
                    self.set_initial_pose()
                    rospy.sleep(1)
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        enhanced_frame = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_frame)
                        qr_codes = decode(enhanced_frame)
                        if qr_codes:
                            qr = qr_codes[0]
                            points = np.array(qr.polygon, dtype=np.int32).reshape(-1, 1, 2)
                            center_x_qr = int(np.mean(points[:, 0, 0]))
                            dx = center_x_qr - 320
                            if abs(dx) > self.x_tolerance / 4:  # Very tight tolerance
                                rospy.loginfo("Correcting deviation: dx=%d", dx)
                                twist = Twist()
                                lateral_x = -0.0003 * dx  # Finer control
                                lateral_x = max(min(lateral_x, 0.01), -0.01)  # Reduced max speed
                                twist.linear.x = lateral_x
                                self.ros_ctrl.pub_cmdVel.publish(twist)
                                rospy.sleep(abs(dx) * 0.003)  # Shorter duration
                                self.ros_ctrl.pub_cmdVel.publish(Twist())
                                # Final verification
                                ret, frame = self.cap.read()
                                if ret and frame is not None and frame.size > 0:
                                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                    enhanced_frame = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_frame)
                                    qr_codes = decode(enhanced_frame)
                                    if qr_codes:
                                        qr = qr_codes[0]
                                        points = np.array(qr.polygon, dtype=np.int32).reshape(-1, 1, 2)
                                        center_x_qr = int(np.mean(points[:, 0, 0]))
                                        dx = center_x_qr - 320
                                        if abs(dx) > self.x_tolerance / 8:  # Ultra-tight tolerance
                                            lateral_x = -0.0003 * dx
                                            lateral_x = max(min(lateral_x, 0.01), -0.01)
                                            twist.linear.x = lateral_x
                                            self.ros_ctrl.pub_cmdVel.publish(twist)
                                            rospy.sleep(abs(dx) * 0.003)
                                            self.ros_ctrl.pub_cmdVel.publish(Twist())
                    self.task_in_progress = False
                    rospy.loginfo("Task completed, returned to container")
            rospy.sleep(0.1)
        self.cleanup()

    def cleanup(self):
        self.cap.release()
        if self.display_enabled:
            cv2.destroyAllWindows()
        self.ros_ctrl.cancel()

if __name__ == '__main__':
    try:
        node = QrScannerNode()
        node.main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("Error: %s", e)
