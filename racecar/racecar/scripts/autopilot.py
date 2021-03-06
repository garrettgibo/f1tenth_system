#!/usr/bin/env python
import importlib

import math
import rospy
import genpy.message
from rospy import ROSException
import sensor_msgs.msg
from sensor_msgs.msg import Image
import actionlib
import rostopic
import rosservice
from threading import Thread
from rosservice import ROSServiceException
from cv_bridge import CvBridge

import cv2
import numpy as np
from simple_pid import PID


class JoyTeleopException(Exception):
    pass

'''
Originally from https://github.com/ros-teleop/teleop_tools
Pulled on April 28, 2017.

Edited by Winter Guerra on April 28, 2017 to allow for default actions.
'''

class ImageProcessor:
    def __init__(self):
        self.image_topic = "/camera/color/image_raw"
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.bridge = CvBridge()
        self.latest_image = np.zeros((720, 480, 3), np.uint8)


    def image_callback(self, data):
        """Transform ROS image to OpenCV image array, then save latest image"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        self.latest_image = cv_image

    def get_image(self):
        """Return latest image that has been converted to OpenCV image array"""
        return self.latest_image


class LineFollower:
    def __init__(self):
        self.vert_scan_y = 240   # num pixels from the top to start horiz scan
        self.img_width = 640
        self.image_rgb = rospy.Publisher("camera_rgb", Image)
        self.image_gray = rospy.Publisher("camera_gray", Image)
        self.bridge = CvBridge()
        # self.avg_steerings = []
        self.steering = 0  # straight

    def display_heading_line(self, frame, steering_angle, line_color=(0, 0, 255), line_width=2):
        heading_image = np.zeros_like(frame)
        height, width, _ = frame.shape

        # figure out the heading line from steering angle
        # heading line (x1,y1) is always center bottom of the screen
        # (x2, y2) requires a bit of trigonometry

        # Note: the steering angle of:
        # 0-89 degree: turn left
        # 90 degree: going straight
        # 91-180 degree: turn right
        # steering_angle *= 180 / np.pi

        steering_angle = np.pi / 2 - steering_angle
            
        # steering_angle *= np.pi / 180

        steering_angle_radian  = steering_angle
        x1 = int(width / 2)
        y1 = height
        x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
        y2 = int(height / 2)

        cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
        heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)

        return heading_image

    def stabilize_steering_angle(
            self,
            curr_steering_angle,
            new_steering_angle,
            max_angle_deviation=5):
        new_steering_angle *= 180 / np.pi
        curr_steering_angle *= 180 / np.pi

        angle_deviation = new_steering_angle - curr_steering_angle
#        rospy.logerr(angle_deviation)
        if abs(angle_deviation) > max_angle_deviation:
            stabilized_steering_angle = int(curr_steering_angle 
                + max_angle_deviation * angle_deviation / abs(angle_deviation))
        else:
            stabilized_steering_angle = new_steering_angle

        stabilized_steering_angle *= np.pi / 180

        return stabilized_steering_angle

    def hough_steering(self, cam_img):
        scan_line = cam_img[self.vert_scan_y : -50, :, :]
        img_blur = cv2.GaussianBlur(scan_line, (9, 9), 0)
        img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

        # get white line
        img_hsv = cv2.cvtColor(scan_line, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 255 - 100])
        upper_white = np.array([255, 40, 255])
        mask_white = cv2.inRange(img_hsv, lower_white, upper_white)

        # get yellow line
        lower_yellow = np.array([40, 0, 0])
        upper_yellow = np.array([80, 255, 255])
        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

        mask_yellow = cv2.bitwise_not(mask_yellow)

        mask = cv2.bitwise_or(mask_white, mask_yellow)
        mask = mask_white



        edges = cv2.Canny(mask, 50, 150, apertureSize = 3)
        self.image_gray.publish(self.bridge.cv2_to_imgmsg(edges, "mono8"))
        # edges = cv2.Canny(img_blur, 50, 150, apertureSize = 3)
        # kernel = np.ones((5, 5),np.uint8)
        # edges = cv2.dilate(edges, kernel, iterations = 1)
        # self.image_gray.publish(self.bridge.cv2_to_imgmsg(edges, "mono8"))
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=40, maxLineGap=50)
        slopes = []

        height, width = mask.shape

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(scan_line, (x1, y1), (x2, y2), (0, 255, 0), 2)
                slope = (y2 - y1) / float(x2 - x1)
                
                # essential to make sharp turns`
                if abs(slope) < 0.5:
                    # rospy.logerr("1")
                    slope *= 9.5
                # should make it go straight better
                elif abs(slope) > 2 :
                    # rospy.logerr("2")
                    slope  *= 6
                # straight better on turns to weight one side more
                elif x1 < width / 2:
                    if x1 < width / 4 and height - y1 < height / 2:
                        # r# ospy.logerr("3")
                        slope *= 2.5
                else:
                    if abs(width / 4  - x1) < 75 and height - y2 < height / 2:
                        # rospy.logerr("4")
                        slope *= 2.5


                slopes.append(slope)
        else:
            slopes.append(self.avg_slope)

        self.avg_slope =  np.mean(slopes)
        # rospy.logerr(self.avg_slope)
        new_angle = np.arctan(self.avg_slope)

        img  = self.display_heading_line(scan_line, new_angle)

        return img, new_angle

    def stabilize_throttle(self, throttle, steering, max_steering_deviation):
        # get diff of old and new steering
        steering_deviation = (self.steering - steering) * 180 / np.pi
        
        if abs(steering_deviation) > max_steering_deviation:
            new_throttle = throttle - abs(steering_deviation * throttle / 90 )
        else:
            new_throttle = self.cruise_throttle

        return new_throttle

    def run(self, cam_img):
        #img_lane_mask, avg_lane_pos = self.lane_detection(cam_img)
        img_lane_mask, steering = self.hough_steering(cam_img)

        # rospy.logerr("angle delta: {}".format((self.steering - steering) * 180 / np.pi))
        max_angle_deviation = 5.5
        stabilized_angle = self.stabilize_steering_angle(self.steering, steering, max_angle_deviation)

        scale_throttle = 3.4
        scale_steering = 0.7
        self.cruise_throttle = 0.75 * scale_throttle

        # self.throttle  = self.stabilize_throttle(self.cruise_throttle, steering, 20)
        self.throttle = self.cruise_throttle
        self.steering = stabilized_angle * scale_steering

        # pub lanes "masks"
        self.image_rgb.publish(self.bridge.cv2_to_imgmsg(img_lane_mask, "bgr8"))
        rospy.logerr("steering: {} -- throttle: {}".format(self.steering * 180 / np.pi, self.throttle))
        return self.steering, self.throttle

class JoyTeleop:
    """
    Generic joystick teleoperation node.
    Will not start without configuration, has to be stored in 'teleop' parameter.
    See config/joy_teleop.yaml for an example.
    """
    def __init__(self):
        if not rospy.has_param("teleop"):
            rospy.logfatal("no configuration was found, taking node down")
            raise JoyTeleopException("no config")

        self.publishers = {}
        self.al_clients = {}
        self.srv_clients = {}
        self.service_types = {}
        self.message_types = {}
        self.command_list = {}
        self.offline_actions = []
        self.offline_services = []

        self.old_buttons = []

        # custom line follower and image processing pipeline
        self.image_processor = ImageProcessor()
        self.line_follower = LineFollower()

        teleop_cfg = rospy.get_param("teleop")

        for i in teleop_cfg:
            if i in self.command_list:
                rospy.logerr("command {} was duplicated".format(i))
                continue
            action_type = teleop_cfg[i]['type']
            self.add_command(i, teleop_cfg[i])
            if action_type == 'topic':
                self.register_topic(i, teleop_cfg[i])
            elif action_type == 'action':
                self.register_action(i, teleop_cfg[i])
            elif action_type == 'service':
                self.register_service(i, teleop_cfg[i])
            else:
                rospy.logerr("unknown type '%s' for command '%s'", action_type, i)

        # Don't subscribe until everything has been initialized.
        rospy.Subscriber('joy', sensor_msgs.msg.Joy, self.joy_callback)

        # Run a low-freq action updater
        rospy.Timer(rospy.Duration(2.0), self.update_actions)

    def joy_callback(self, data):
        try:
            for c in self.command_list:
                if self.match_command(c, data.buttons):
                    self.run_command(c, data)
                    # Only run 1 command at a time
                    break
        except JoyTeleopException as e:
            rospy.logerr("error while parsing joystick input: %s", str(e))
        self.old_buttons = data.buttons

    def register_topic(self, name, command):
        """Add a topic publisher for a joystick command"""
        topic_name = command['topic_name']
        try:
            topic_type = self.get_message_type(command['message_type'])
            self.publishers[topic_name] = rospy.Publisher(topic_name, topic_type, queue_size=1)
        except JoyTeleopException as e:
            rospy.logerr("could not register topic for command {}: {}".format(name, str(e)))

    def register_action(self, name, command):
        """Add an action client for a joystick command"""
        action_name = command['action_name']
        try:
            action_type = self.get_message_type(self.get_action_type(action_name))
            self.al_clients[action_name] = actionlib.SimpleActionClient(action_name, action_type)
            if action_name in self.offline_actions:
                self.offline_actions.remove(action_name)
        except JoyTeleopException:
            if action_name not in self.offline_actions:
                self.offline_actions.append(action_name)

    class AsyncServiceProxy(object):
        def __init__(self, name, service_class, persistent=True):
            try:
                rospy.wait_for_service(name, timeout=2.0)
            except ROSException:
                raise JoyTeleopException("Service {} is not available".format(name))
            self._service_proxy = rospy.ServiceProxy(name, service_class, persistent)
            self._thread = Thread(target=self._service_proxy, args=[])

        def __del__(self):
            # try to join our thread - no way I know of to interrupt a service
            # request
            if self._thread.is_alive():
                self._thread.join(1.0)

        def __call__(self, request):
            if self._thread.is_alive():
                self._thread.join(0.01)
                if self._thread.is_alive():
                    return False

            self._thread = Thread(target=self._service_proxy, args=[request])
            self._thread.start()
            return True

    def register_service(self, name, command):
        """ Add an AsyncServiceProxy for a joystick command """
        service_name = command['service_name']
        try:
            service_type = self.get_service_type(service_name)
            self.srv_clients[service_name] = self.AsyncServiceProxy(
                service_name,
                service_type)

            if service_name in self.offline_services:
                self.offline_services.remove(service_name)
        except JoyTeleopException:
            if service_name not in self.offline_services:
                self.offline_services.append(service_name)

    def match_command(self, c, buttons):
        """Find a command matching a joystick configuration"""
        # Buttons is a vector of the shape [0,1,0,1....
        # Turn it into a vector of form [1, 3...
        button_indexes = np.argwhere(buttons).flatten()

        # Check if the pressed buttons match the commands exactly.
        buttons_match = np.array_equal(self.command_list[c]['buttons'], button_indexes)

        #print button_indexes
        if buttons_match:
            return True

        # This might also be a default command.
        # We need to check if ANY commands match this set of pressed buttons.
        any_commands_matched = np.any([ np.array_equal(command['buttons'], button_indexes) for name, command in self.command_list.iteritems()])

        # Return the final result.
        return (buttons_match) or (not any_commands_matched and self.command_list[c]['is_default'])

    def add_command(self, name, command):
        """Add a command to the command list"""
        # Check if this is a default command
        if 'is_default' not in command:
            command['is_default'] = False

        if command['type'] == 'topic':
            if 'deadman_buttons' not in command:
                command['deadman_buttons'] = []
            command['buttons'] = command['deadman_buttons']
        elif command['type'] == 'action':
            if 'action_goal' not in command:
                command['action_goal'] = {}
        elif command['type'] == 'service':
            if 'service_request' not in command:
                command['service_request'] = {}
        self.command_list[name] = command

    def run_command(self, command, joy_state):
        """Run a joystick command"""
        cmd = self.command_list[command]

        if command == "autonomous_control":
            # new mode to detect autonomous control mode bound to RB
            self.run_auto_topic(command)
        elif cmd['type'] == 'topic':
            self.run_topic(command, joy_state)
        elif cmd['type'] == 'action':
            if cmd['action_name'] in self.offline_actions:
                rospy.logerr("command {} was not played because the action "
                             "server was unavailable. Trying to reconnect..."
                             .format(cmd['action_name']))
                self.register_action(command, self.command_list[command])
            else:
                if joy_state.buttons != self.old_buttons:
                    self.run_action(command, joy_state)
        elif cmd['type'] == 'service':
            if cmd['service_name'] in self.offline_services:
                rospy.logerr("command {} was not played because the service "
                             "server was unavailable. Trying to reconnect..."
                             .format(cmd['service_name']))
                self.register_service(command, self.command_list[command])
            else:
                if joy_state.buttons != self.old_buttons:
                    self.run_service(command, joy_state)
        else:
            raise JoyTeleopException('command {} is neither a topic publisher nor an action or service client'
                                     .format(command))

    def run_auto_topic(self, c):
        """Run command for autonmous mode."""
        cmd = self.command_list[c]
        msg = self.get_message_type(cmd["message_type"])()
        # rospy.logerr("Name: {}".format(c))

        # Do some image processing
        cam_img = self.image_processor.get_image()
        steering, throttle = self.line_follower.run(cam_img)

        # control car here
        self.set_member(msg, "drive.speed", throttle)
        self.set_member(msg, "drive.steering_angle", steering)

        self.publishers[cmd['topic_name']].publish(msg)

    def run_topic(self, c, joy_state):
        cmd = self.command_list[c]
        msg = self.get_message_type(cmd['message_type'])()
        if 'message_value' in cmd:
            for param in cmd['message_value']:
                self.set_member(msg, param['target'], param['value'])

        else:
            for mapping in cmd['axis_mappings']:
                if len(joy_state.axes)<=mapping['axis']:
                  rospy.logerr('Joystick has only {} axes (indexed from 0), but #{} was referenced in config.'.format(len(joy_state.axes), mapping['axis']))
                  val = 0.0
                else:
                  val = joy_state.axes[mapping['axis']] * mapping.get('scale', 1.0) + mapping.get('offset', 0.0)
                  # rospy.logerr("Testing: " + str(val))

                self.set_member(msg, mapping['target'], val)

        self.publishers[cmd['topic_name']].publish(msg)
        # rospy.logerr(msg)

    def run_action(self, c, joy_state):
        cmd = self.command_list[c]
        goal = self.get_message_type(self.get_action_type(cmd['action_name'])[:-6] + 'Goal')()
        genpy.message.fill_message_args(goal, [cmd['action_goal']])
        self.al_clients[cmd['action_name']].send_goal(goal)

    def run_service(self, c, joy_state):
        cmd = self.command_list[c]
        request = self.get_service_type(cmd['service_name'])._request_class()
        # should work for requests, too
        genpy.message.fill_message_args(request, [cmd['service_request']])
        if not self.srv_clients[cmd['service_name']](request):
            rospy.loginfo('Not sending new service request for command {} because previous request has not finished'
                          .format(c))

    def set_member(self, msg, member, value):
        ml = member.split('.')
        if len(ml) < 1:
            return
        target = msg
        for i in ml[:-1]:
            target = getattr(target, i)
        setattr(target, ml[-1], value)

    def get_message_type(self, type_name):
        if type_name not in self.message_types:
            try:
                package, message = type_name.split('/')
                mod = importlib.import_module(package + '.msg')
                self.message_types[type_name] = getattr(mod, message)
            except ValueError:
                raise JoyTeleopException("message type format error")
            except ImportError:
                raise JoyTeleopException("module {} could not be loaded".format(package))
            except AttributeError:
                raise JoyTeleopException("message {} could not be loaded from module {}".format(package, message))
        return self.message_types[type_name]

    def get_action_type(self, action_name):
        try:
            return rostopic._get_topic_type(rospy.resolve_name(action_name) + '/goal')[0][:-4]
        except TypeError:
            raise JoyTeleopException("could not find action {}".format(action_name))

    def get_service_type(self, service_name):
        if service_name not in self.service_types:
            try:
                self.service_types[service_name] = rosservice.get_service_class_by_name(service_name)
            except ROSServiceException, e:
                raise JoyTeleopException("service {} could not be loaded: {}".format(service_name, str(e)))
        return self.service_types[service_name]

    def update_actions(self, evt=None):
        for name, cmd in self.command_list.iteritems():
            if cmd['type'] != 'action':
                continue
            if cmd['action_name'] in self.offline_actions:
                self.register_action(name, cmd)


if __name__ == "__main__":
    try:
        rospy.init_node('autopilot')
        jt = JoyTeleop()
        rospy.spin()
    except JoyTeleopException:
        pass
    except rospy.ROSInterruptException:
        pass
