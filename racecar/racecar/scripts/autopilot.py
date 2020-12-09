#!/usr/bin/env python
import importlib

import rospy
import genpy.message
from rospy import ROSException
import sensor_msgs.msg
import actionlib
import rostopic
import rosservice
from threading import Thread
from rosservice import ROSServiceException

# import cv2
import numpy as np
from simple_pid import PID


class JoyTeleopException(Exception):
    pass

'''
Originally from https://github.com/ros-teleop/teleop_tools
Pulled on April 28, 2017.

Edited by Winter Guerra on April 28, 2017 to allow for default actions.
'''


class LineFollower:
    def __init__(self):
        self.test = 0
        self.vert_scan_y = 60   # num pixels from the top to start horiz scan
        self.vert_scan_height = 10 # num pixels high to grab from horiz scan
        self.color_thr_low = np.asarray((0, 50, 50)) # hsv dark yellow
        self.color_thr_hi = np.asarray((50, 255, 255)) # hsv light yellow
        self.target_pixel = None # of the N slots above, which is the ideal relationship target
        self.steering = 0.0 # from -1 to 1
        self.throttle = 0.15 # from -1 to 1
        self.recording = False # Set to true if desired to save camera frames
        self.delta_th = 0.1 # how much to change throttle when off
        self.throttle_max = 0.3
        self.throttle_min = 0.15
        self.pid_st = PID(Kp=-0.01, Ki=0.00, Kd=-0.001)
    """
    def __init__(self):
        self.vert_scan_y = 60   # num pixels from the top to start horiz scan
        self.vert_scan_height = 10 # num pixels high to grab from horiz scan
        self.color_thr_low = np.asarray((0, 50, 50)) # hsv dark yellow
        self.color_thr_hi = np.asarray((50, 255, 255)) # hsv light yellow
        self.target_pixel = None # of the N slots above, which is the ideal relationship target
        self.steering = 0.0 # from -1 to 1
        self.throttle = 0.15 # from -1 to 1
        self.recording = False # Set to true if desired to save camera frames
        self.delta_th = 0.1 # how much to change throttle when off
        self.throttle_max = 0.3
        self.throttle_min = 0.15
        self.pid_st = PID(Kp=-0.01, Ki=0.00, Kd=-0.001)
    
    def get_i_color(self, cam_img):
        # take a horizontal slice of the image
        iSlice = self.vert_scan_y
        scan_line = cam_img[iSlice : iSlice + self.vert_scan_height, :, :]

        # convert to HSV color space
        img_hsv = cv2.cvtColor(scan_line, cv2.COLOR_RGB2HSV)

        # make a mask of the colors in our range we are looking for
        mask = cv2.inRange(img_hsv, self.color_thr_low, self.color_thr_hi)

        # which index of the range has the highest amount of yellow?
        hist = np.sum(mask, axis=0)
        max_yellow = np.argmax(hist)

        return max_yellow, hist[max_yellow], mask
    
    def run(self, cam_img):
        max_yellow, confidense, mask = self.get_i_color(cam_img)
        conf_thresh = 0.001
        
        if self.target_pixel is None:
            # Use the first run of get_i_color to set our relationship with the yellow line.
            # You could optionally init the target_pixel with the desired value.
            self.target_pixel = max_yellow

            # this is the target of our steering PID controller
            self.pid_st.setpoint = self.target_pixel

        elif confidense > conf_thresh:
            # invoke the controller with the current yellow line position
            # get the new steering value as it chases the ideal
            self.steering = self.pid_st(max_yellow)
    """

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
        rospy.logerr("Name: {}".format(c))

        # Do some image processing
        # cam_img = ...
        #steering, throttle, recording = self.line_follower.run(cam_img)
        # steering, throttle = 0.5 , 0.5

        # rospy.logerr("steering: {} -- throttle: {}".format(steering, throttle))
        # control car here
        # self.set_member(msg, "drive.speed", throttle)
        # self.set_member(msg, "drive.steering_angle", steering)
        self.set_member(msg, "drive.speed", 0)
        self.set_member(msg, "drive.steering_angle", 0)

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
