#!/usr/bin/env python2
# encoding: utf-8
import os
import math
import rospkg
import rospy
from follow_common import *
from std_msgs.msg import Bool
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage, LaserScan, Image
from time import sleep
import cv2 as cv
import numpy as np
from pyzbar import pyzbar
RAD2DEG = 180 / math.pi

class LineDetect:
    def __init__(self):
        rospy.on_shutdown(self.cancel)
        rospy.init_node("LineDetect", anonymous=False)
        self.robot_state = "following"
        self.ros_ctrl = ROSCtrl()
        self.ros_ctrl.Joy_active = False  # Disable joystick control
        self.grip_closed = False
        self.rack_shelf = False  # Start with lower shelf
        self.qr_value = None  # Store QR code value (B1 or B2)
        self.img = None
        self.circle = ""
        self.hsv_range = ((55, 75, 120), (125, 253, 255))  # Hardcoded for blue line
        self.Roi_init = ""
        self.warning = 1
        self.Start_state = True
        self.Buzzer_state = False
        self.select_flags = False
        self.Track_state = 'tracking'  # Start in tracking mode
        self.windows_name = 'frame'
        self.qr_window = 'qr_frame'
        self.color = color_follow()
        self.cols, self.rows = 0, 0
        self.Mouse_XY = (0, 0)
        self.img_flip = rospy.get_param("~img_flip", False)
        self.VideoSwitch = rospy.get_param("~VideoSwitch", False)
        self.hsv_text = rospkg.RosPack().get_path("yahboomcar_linefollw")+"/scripts/LineFollowHSV.text"
        self.scale = 1000  # Hardcoded from LineDetectPID.cfg
        self.FollowLinePID = (40, 0, 20)  # Hardcoded from LineDetectPID.cfg
        self.linear = 0.3  # Hardcoded from LineDetectPID.cfg
        self.LaserAngle = 10  # Hardcoded from LineDetectPID.cfg
        self.ResponseDist = 0.6  # Hardcoded from LineDetectPID.cfg
        self.PID_init()
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.registerScan, queue_size=1)
        self.pub_rgb = rospy.Publisher("/linefollw/rgb", Image, queue_size=1)
        self.pub_Buzzer = rospy.Publisher('/Buzzer', Bool, queue_size=1)
        self.capture = None
        self.initialize_camera()
        if self.VideoSwitch == False:
            from cv_bridge import CvBridge
            self.bridge = CvBridge()
            self.sub_img = rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.compressed_callback)
        self.qr_pid = simplePID(
            [0, 0],
            [0.2, 0],  # Kp
            [0.0, 0],  # Ki
            [0.015, 0]  # Kd
        )
        self.frame_counter = 0
        self.display_enabled = os.getenv('DISPLAY') is not None
        if self.display_enabled:
            try:
                cv.namedWindow(self.windows_name, cv.WINDOW_AUTOSIZE)
                cv.namedWindow(self.qr_window, cv.WINDOW_AUTOSIZE)
                rospy.sleep(0.1)
                rospy.loginfo("Initialized OpenCV windows: %s, %s", self.windows_name, self.qr_window)
            except Exception as e:
                rospy.logerr("Failed to initialize OpenCV windows: %s", e)
                self.display_enabled = False

    def initialize_camera(self):
        """Initialize arm camera (/dev/video1, index 1). Depth camera (/dev/video2, index 2) is unused."""
        max_retries = 5
        dev_path = "/dev/video1"
        cam_index = 1
        for attempt in range(max_retries):
            try:
                if not os.path.exists(dev_path):
                    rospy.logerr("%s not found on attempt %d", dev_path, attempt + 1)
                    rospy.sleep(7.0)
                    continue
                self.capture = cv.VideoCapture(cam_index)
                cv_edition = cv.__version__
                if cv_edition[0] == '3':
                    self.capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
                else:
                    self.capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
                self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                self.capture.set(cv.CAP_PROP_FPS, 30)
                if not self.capture.isOpened():
                    rospy.logwarn("Failed to open arm camera (index 1) with MJPG on attempt %d", attempt + 1)
                    self.capture.release()
                    self.capture = cv.VideoCapture(cam_index)
                    self.capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
                    self.capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
                    self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
                    self.capture.set(cv.CAP_PROP_FPS, 30)
                    if not self.capture.isOpened():
                        rospy.logerr("Failed to open arm camera (index 1) with YUYV on attempt %d", attempt + 1)
                        self.capture.release()
                        continue
                valid_frames = 0
                for _ in range(10):
                    ret, frame = self.capture.read()
                    if ret and frame is not None and frame.size != 0:
                        valid_frames += 1
                    rospy.sleep(0.1)
                if valid_frames < 6:
                    rospy.logwarn("Insufficient valid frames (%d/10) from arm camera (index 1) on attempt %d", valid_frames, attempt + 1)
                    self.capture.release()
                    continue
                fps = self.capture.get(cv.CAP_PROP_FPS)
                width = self.capture.get(cv.CAP_PROP_FRAME_WIDTH)
                height = self.capture.get(cv.CAP_PROP_FRAME_HEIGHT)
                fourcc_code = int(self.capture.get(cv.CAP_PROP_FOURCC))
                fourcc = ''.join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])
                rospy.loginfo("Arm camera (index 1) initialized: FPS=%.2f, %dx%d, FOURCC=%s", fps, int(width), int(height), fourcc)
                return
            except Exception as e:
                rospy.logerr("Error initializing arm camera (index 1) on attempt %d: %s", attempt + 1, e)
                if self.capture is not None:
                    self.capture.release()
                    self.capture = None
                rospy.sleep(7.0)
        rospy.logerr("Failed to initialize arm camera (index 1) after %d attempts", max_retries)
        raise Exception("Arm camera initialization failed")

    def cancel(self):
        self.Reset()
        self.ros_ctrl.cancel()
        self.sub_scan.unregister()
        if hasattr(self, "sub_img"):
            self.sub_img.unregister()
        self.pub_rgb.unregister()
        self.pub_Buzzer.unregister()
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            rospy.loginfo("Released arm camera (index 1)")
            rospy.sleep(3.0)
        if self.display_enabled:
            try:
                cv.destroyWindow(self.windows_name)
                cv.destroyWindow(self.qr_window)
                rospy.sleep(0.1)
                rospy.loginfo("Closed OpenCV windows")
            except Exception as e:
                rospy.logerr("Error closing OpenCV windows: %s", e)
        print "Shutting down this node."

    def align_with_qr(self):
        """Align robot with QR code using arm camera (index 1) and pyzbar."""
        rospy.loginfo("🔍 Starting QR code alignment with arm camera...")
        rospy.loginfo("📏 Ensuring arm is lowered for QR alignment...")
        self.lower_arm_for_box()
        sleep(3)
        if self.capture is None or not self.capture.isOpened():
            rospy.logerr("Arm camera (index 1) not initialized or failed to open! Reinitializing...")
            try:
                self.initialize_camera()
            except Exception as e:
                rospy.logerr("Arm camera (index 1) reinitialization failed: %s", e)
                return False
        x_tolerance = 10
        max_angular_z = 0.2
        stabilization_delay = 2.0
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
                        try:
                            self.initialize_camera()
                        except Exception as e:
                            rospy.logerr("Arm camera (index 1) reinitialization failed: %s", e)
                        break
                    return False
                if not self.capture.isOpened():
                    rospy.logerr("Arm camera (index 1) closed unexpectedly! Reinitializing...")
                    try:
                        self.initialize_camera()
                    except Exception as e:
                        rospy.logerr("Arm camera (index 1) reinitialization failed: %s", e)
                        return False
                ret, frame = self.capture.read()
                if not ret or frame is None or frame.size == 0:
                    rospy.logwarn("Arm camera (index 1) read failed or invalid frame! Shape: %s", str(frame.shape) if frame is not None else "None")
                    if self.display_enabled:
                        try:
                            cv.putText(frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8),
                                       "Camera read failed", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            cv.imshow(self.qr_window, frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8))
                            cv.waitKey(1)
                        except Exception as e:
                            rospy.logerr("Error displaying QR frame: %s", e)
                    continue
                rospy.loginfo("Frame captured: Shape=%s", str(frame.shape))
                qr_codes = pyzbar.decode(frame)
                center_x = 320
                qr_detected = False
                dx = 0
                display_frame = frame.copy()
                if qr_codes:
                    qr_detected = True
                    no_qr_start = None
                    qr = qr_codes[0]
                    self.qr_value = qr.data.decode('utf-8')  # Store QR code value (B1 or B2)
                    rospy.loginfo("QR code detected: Value=%s", self.qr_value)
                    points = np.array(qr.polygon, dtype=np.int32).reshape(-1, 1, 2)
                    center_x_box = int(np.mean(points[:, 0, 0]))
                    center_y_box = int(np.mean(points[:, 0, 1]))
                    dx = center_x_box - center_x
                    rospy.loginfo("QR code detected, Δx: %s, Center: (%d, %d)", dx, center_x_box, center_y_box)
                    cv.polylines(display_frame, [points], True, (0, 255, 0), 2)
                    cv.circle(display_frame, (center_x_box, center_y_box), 5, (0, 255, 255), -1)
                    cv.line(display_frame, (center_x, 240), (center_x_box, center_y_box), (255, 255, 0), 2)
                    cv.putText(display_frame, "QR: %s" % self.qr_value, (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                    if abs(dx) <= x_tolerance:
                        if stabilization_start is None:
                            stabilization_start = rospy.Time.now().to_sec()
                            rospy.loginfo("✅ QR code centered. Starting %s-second stabilization...", stabilization_delay)
                            twist.angular.z = 0.0
                            self.ros_ctrl.pub_cmdVel.publish(twist)
                        elif rospy.Time.now().to_sec() - stabilization_start >= stabilization_delay:
                            rospy.loginfo("✅ Stabilization complete. QR code centered.")
                            return True
                    else:
                        stabilization_start = None
                        error = dx
                        angular_z, _ = self.qr_pid.update([error, 0])
                        angular_z = max(min(angular_z, max_angular_z), -max_angular_z)
                        twist.angular.z = -angular_z if self.img_flip else angular_z
                        rospy.loginfo("Adjusting angular - dx: %s, angular.z: %s", dx, twist.angular.z)
                        self.ros_ctrl.pub_cmdVel.publish(twist)
                if not qr_detected:
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
                    cv.putText(display_frame, "No QR code detected", (10, 30),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if twist.angular.z != 0.0:
                        twist.angular.z = 0.0
                        self.ros_ctrl.pub_cmdVel.publish(twist)
                self.frame_counter += 1
                if self.display_enabled and self.frame_counter % 5 == 0:
                    try:
                        frame_time = rospy.Time.now().to_sec() - start_time
                        cv.putText(display_frame, "FPS: %.2f" % (1.0 / (frame_time + 0.001)),
                                   (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 200), 1)
                        cv.rectangle(display_frame, (center_x - x_tolerance, 210), (center_x + x_tolerance, 270),
                                     (255, 0, 0), 2)
                        cv.imshow(self.qr_window, display_frame)
                        cv.waitKey(1)
                    except Exception as e:
                        rospy.logerr("Error displaying QR frame: %s", e)
                        self.display_enabled = False
            if attempt == max_retries:
                break
        self.ros_ctrl.pub_cmdVel.publish(Twist())
        rospy.loginfo("🦋 QR alignment failed or timed out.")
        return False

    def compressed_callback(self, msg):
        if not isinstance(msg, CompressedImage):
            return
        start = time.time()
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        action = cv.waitKey(1) & 0xFF
        rgb_img, binary = self.process(frame, action)
        end = time.time()
        fps = 1 / (end - start + 0.001)
        text = "FPS : " + str(int(fps))
        cv.putText(rgb_img, text, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
        if self.display_enabled:
            try:
                self.frame_counter += 1
                if self.frame_counter % 3 == 0:
                    if len(binary) != 0:
                        cv.imshow(self.windows_name, ManyImgs(1, ([rgb_img, binary])))
                    else:
                        cv.imshow(self.windows_name, rgb_img)
                    cv.waitKey(1)
            except Exception as e:
                rospy.logerr("Error displaying frame: %s", e)
                self.display_enabled = False
        self.pub_rgb.publish(self.bridge.cv2_to_imgmsg(rgb_img, "bgr8"))

    def process(self, rgb_img, action):
        binary = ""
        rgb_img = cv.resize(rgb_img, (640, 480))
        if self.img_flip:
            rgb_img = cv.flip(rgb_img, 1)
        if action == 32:
            self.Track_state = 'tracking'
        elif action == ord('i') or action == 105:
            self.Track_state = "identify"
        elif action == ord('r') or action == 114:
            self.Reset()
        elif action == ord('q') or action == 113:
            self.cancel()
        if self.Track_state == 'init':
            if self.display_enabled:
                try:
                    cv.namedWindow(self.windows_name, cv.WINDOW_AUTOSIZE)
                    cv.setMouseCallback(self.windows_name, self.onMouse, 0)
                    rospy.sleep(0.1)
                except Exception as e:
                    rospy.logerr("Error setting up window for init: %s", e)
                    self.display_enabled = False
            if self.select_flags:
                cv.line(rgb_img, self.cols, self.rows, (255, 0, 0), 2)
                cv.rectangle(rgb_img, self.cols, self.rows, (0, 255, 0), 2)
                if self.Roi_init[0] != self.Roi_init[1] and self.Roi_init[2] != self.Roi_init[3]:
                    rgb_img, self.hsv_range = self.color.Roi_hsv(rgb_img, self.Roi_init)
                else:
                    self.Track_state = 'init'
        if self.Track_state == 'tracking':
            rgb_img, binary, self.circle = self.color.line_follow(rgb_img, self.hsv_range)
        if self.Track_state == 'tracking' and len(self.circle) != 0:
            self.execute(self.circle[0], self.circle[2])
        else:
            if self.Start_state:
                self.ros_ctrl.pub_cmdVel.publish(Twist())
                self.Start_state = False
        return rgb_img, binary

    def onMouse(self, event, x, y, flags, _):
        if event == 1:
            self.Track_state = 'init'
            self.select_flags = True
            self.Mouse_XY = (x, y)
        if event == 4:
            self.select_flags = False
            self.Track_state = 'mouse'
        if self.select_flags:
            self.cols = min(self.Mouse_XY[0], x), min(self.Mouse_XY[1], y)
            self.rows = max(self.Mouse_XY[0], x), max(self.Mouse_XY[1], y)
            self.Roi_init = (self.cols[0], self.cols[1], self.rows[0], self.rows[1])

    def execute(self, point_x, color_radius):
        if self.robot_state == "turning":
            return
        if self.ros_ctrl.Joy_active:
            if self.Start_state:
                self.PID_init()
                self.Start_state = False
            return
        self.Start_state = False
        if not hasattr(self, 'grip_closed'):
            self.grip_closed = False
        if color_radius == 0:
            self.ros_ctrl.pub_cmdVel.publish(Twist())
        else:
            twist = Twist()
            b = Bool()
            [z_Pid, _] = self.PID_controller.update([(point_x - 320)/16, 0])
            twist.angular.z = -z_Pid if self.img_flip else z_Pid
            twist.linear.x = self.linear
            if self.warning > 10:
                rospy.loginfo("Obstacles ahead !!!")
                self.robot_state = "turning"
                self.ros_ctrl.pub_cmdVel.publish(Twist())
                rospy.sleep(0.5)
                self.Buzzer_state = True
                rospy.sleep(0.5)
                if not self.grip_closed:
                    self.pick_downrack()
                    sleep(3)
                    rospy.loginfo("📏 Lowering arm to align with box...")
                    self.lower_arm_for_box()
                    sleep(3)
                    rospy.loginfo("🚶 Moving forward to center on box...")
                    self.move_forward(1.77, 0.2)
                    sleep(3)
                    aligned = self.align_with_qr()
                    sleep(3)
                    rospy.loginfo("🚶 Moving forward for precise box grabbing...")
                    self.move_forward(0.87, 0.2)
                    sleep(3)
                    self.close_grip()
                    sleep(3)
                    self.grip_closed = True
                else:
                    rospy.loginfo("Preparing to place box with QR value: %s", self.qr_value)
                   #self.grip_lowerclosed_rack()
                    sleep(3)
                    # Conditional rack placement based on QR code value
                    if self.qr_value == "B1":
                        self.grip_lowerclosed_rack()
                        rospy.loginfo("📦 Placing box B1 on LOWER shelf (shaft 1)...")
                        self.move_forward(2.04, 0.2)
                    elif self.qr_value == "B2":
                        self.grip_upperclosed_rack()
                        rospy.loginfo("📦 Placing box B2 on UPPER shelf (shaft 2)...")
                        self.move_forward(4.04, 0.12)
                    else:
                        rospy.logwarn("Unknown or unset QR code value: %s, defaulting to LOWER shelf", self.qr_value)
                       
                        self.move_forward(1.9, 0.2)
                    sleep(3)
                    self.release_grip()
                    sleep(3)
                    self.grip_closed = False
                    self.qr_value = None  # Reset QR value after placement
                self.move_backward(0.8)
                sleep(3)
                self.turn_left(3.14)
                sleep(3)
                self.pick_uprack()
                sleep(3)
                rospy.sleep(0.5)
                self.robot_state = "following"
                self.warning = 0
            else:
                if self.Buzzer_state:
                    b.data = False
                    for i in range(3):
                        self.pub_Buzzer.publish(b)
                    self.Buzzer_state = False
                self.ros_ctrl.pub_cmdVel.publish(twist)

    def lower_arm_for_box(self):
        rospy.loginfo("🦲 Lowering arm for box alignment...")
        self.ros_ctrl.pubArm([90, 0, 15, 180, 90, 0], run_time=1000)
        sleep(3)
        rospy.loginfo("✅ Arm lowered for box alignment")

    def Reset(self):
        self.PID_init()
        self.Track_state = 'init'
        self.hsv_range = ((55, 75, 120), (125, 253, 255))  # Hardcoded reset
        self.ros_ctrl.Joy_active = False
        self.Mouse_XY = (0, 0)
        self.pub_Buzzer.publish(False)
        self.ros_ctrl.pub_cmdVel.publish(Twist())
        self.qr_value = None  # Reset QR value
        rospy.loginfo("Reset successful!")

    def turn_left(self, duration):
        rospy.loginfo("🔄 Turning left for %s seconds..." % duration)
        stop_twist = Twist()
        self.ros_ctrl.pub_cmdVel.publish(stop_twist)
        rospy.sleep(0.5)
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 1.2
        self.ros_ctrl.pub_cmdVel.publish(twist)
        rospy.sleep(duration)
        self.ros_ctrl.pub_cmdVel.publish(stop_twist)
        rospy.sleep(0.5)
        rospy.loginfo("✅ Turn completed!")

    def turn_right(self, duration):
        rospy.loginfo("🔄 Turning right for %s seconds..." % duration)
        stop_twist = Twist()
        self.ros_ctrl.pub_cmdVel.publish(stop_twist)
        rospy.sleep(0.5)
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = -0.05
        self.ros_ctrl.pub_cmdVel.publish(twist)
        rospy.sleep(duration)
        self.ros_ctrl.pub_cmdVel.publish(stop_twist)
        rospy.sleep(0.5)
        rospy.loginfo("✅ Turn completed!")

    def move_forward(self, duration, speed):
        rospy.loginfo("Moving Forward for %.2f seconds at speed %.2f" % (duration, speed))
        stop_twist = Twist()
        self.ros_ctrl.pub_cmdVel.publish(stop_twist)
        rospy.sleep(0.5)
        twist = Twist()
        twist.linear.x = speed
        twist.angular.z = 0
        self.ros_ctrl.pub_cmdVel.publish(twist)
        rospy.sleep(duration)
        self.ros_ctrl.pub_cmdVel.publish(Twist())
        time.sleep(0.5)
        rospy.loginfo("✅ Move completed!")

    def move_backward(self, duration):
        rospy.loginfo("Moving Backward for %.2f seconds" % duration)
        stop_twist = Twist()
        self.ros_ctrl.pub_cmdVel.publish(stop_twist)
        rospy.sleep(1)
        twist = Twist()
        twist.linear.x = -0.5
        self.ros_ctrl.pub_cmdVel.publish(twist)
        rospy.sleep(duration)
        self.ros_ctrl.pub_cmdVel.publish(Twist())
        time.sleep(0.5)
        rospy.loginfo("✅ Move completed!")

    def close_grip(self):
        self.ros_ctrl.pubArm([], id=6, angle=120)
        sleep(5)

    def grip_lowerclosed_rack(self):
       
           
       
        self.ros_ctrl.pubArm([90, 0, 25, 180, 90, 120], run_time=1000)
       
    def grip_upperclosed_rack(self):
       
        self.ros_ctrl.pubArm([97, 97, 97, 0, 90, 120], run_time=1000)
       
           
       

    def release_grip(self):
        self.ros_ctrl.pubArm([], id=6, angle=0)
        sleep(2)

    def pick_downrack(self):
        self.ros_ctrl.pubArm([], id=6, angle=0)
        sleep(0.5)
        self.ros_ctrl.pubArm([90, 0, 20, 180, 90, 0], run_time=1000)
        sleep(2)

    def pick_uprack(self):
        self.ros_ctrl.pubArm([97, 97, 97, 0, 90, 120], run_time=1000)
        sleep(2)

    def PID_init(self):
        self.PID_controller = simplePID(
            [0, 0],
            [self.FollowLinePID[0] / 1.0 / self.scale, 0],
            [self.FollowLinePID[1] / 1.0 / self.scale, 0],
            [self.FollowLinePID[2] / self.scale, 0])

    def registerScan(self, scan_data):
        self.warning = 1
        if not isinstance(scan_data, LaserScan):
            return
        if self.ros_ctrl.Joy_active:
            return
        ranges = np.array(scan_data.ranges)
        for i in range(len(ranges)):
            angle = (scan_data.angle_min + i * scan_data.angle_increment) * RAD2DEG
            if abs(angle) < self.LaserAngle and ranges[i] < self.ResponseDist:
                self.warning += 1

if __name__ == '__main__':
    line_detect = LineDetect()
    if line_detect.VideoSwitch == False:
        rospy.spin()
    else:
        capture = cv.VideoCapture(2)  # Use arm camera (index 1, /dev/video1)
        cv_edition = cv.__version__
        if cv_edition[0] == '3':
            capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
        else:
            capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        while capture.isOpened() and not rospy.is_shutdown():
            start = time.time()
            ret, frame = capture.read()
            if not ret:
                print "Failed to read frame from camera"
                continue
            action = cv.waitKey(10) & 0xFF
            frame, binary = line_detect.process(frame, action)
            end = time.time()
            fps = 1 / (end - start + 0.0001)
            text = "FPS: " + str(int(fps))
            cv.putText(frame, text, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            if line_detect.display_enabled:
                try:
                    cv.imshow('frame', frame)
                    cv.waitKey(1)
                except Exception as e:
                    print "Error displaying frame: %s" % e
                    line_detect.display_enabled = False
            if action == ord('q') or action == 113:
                break
        capture.release()
        if line_detect.display_enabled:
            try:
                cv.destroyAllWindows()
                rospy.sleep(0.1)
                print "Closed OK!"
            except Exception as e:
                print "Error closing windows in main: %s" % e
