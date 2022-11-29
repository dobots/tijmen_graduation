#!/usr/bin/env python3
# Copyright (c) 2020 The Plankton Authors.
# All rights reserved.
#
# This source code is derived from UUV Simulator
# (https://github.com/uuvsimulator/uuv_simulator)
# Copyright (c) 2016-2019 The UUV Simulator Authors
# licensed under the Apache license, Version 2.0
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np
import rclpy

from rcl_interfaces.msg import ParameterDescriptor

from uuv_PID import PIDRegulator

import geometry_msgs.msg as geometry_msgs
from nav_msgs.msg import Odometry

import tf_quaternion.transformations as transf

from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter
from rclpy.node import Node

from plankton_utils.time import time_in_float_sec_from_msg
from plankton_utils.time import is_sim_time

from transforms3d.euler import quat2euler
from transforms3d.euler import euler2quat

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

import time

class PositionControllerNode(Node):
    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)
        self.get_logger().info('PositionControllerNode: initializing node')
        # Visual marker publishers
        self.traj_pub = self.create_publisher(Path, 'trajectory_marker', 10)     

        self.config = {}

        self.pos_des = np.zeros(3)
        self.quat_des = np.array([0, 0, 0, 1])

        self.initialized = False
        
        # Create 2D points on ellipse which the robot is supposed to follow
        self.num_points = 100
        self.path_points = self.calc_points_on_ellipse(self.num_points)

        # Initialize pids with default parameters
        self.pid_rot = PIDRegulator(1, 0, 0, 1)
        self.pid_pos = PIDRegulator(1, 0, 0, 1)

        self._declare_and_fill_map("pos_p", 1., "p component of pid for position", self.config)
        self._declare_and_fill_map("pos_i", 0.0, "i component of pid for position.", self.config)
        self._declare_and_fill_map("pos_d", 0.0, "d component of pid for position.", self.config)
        self._declare_and_fill_map("pos_sat", 10.0, "saturation of pid for position.", self.config)

        self._declare_and_fill_map("rot_p", 1., "p component of pid for orientation.", self.config)
        self._declare_and_fill_map("rot_i", 0.0, "i component of pid for orientation.", self.config)
        self._declare_and_fill_map("rot_d", 0.0, "d component of pid for orientation.", self.config)
        self._declare_and_fill_map("rot_sat", 3.0, "saturation of pid for orientation.", self.config)

        self.add_on_set_parameters_callback(self.callback_params)

        self.create_pids(self.config)

        # ROS infrastructure
        self.sub_cmd_pose = self.create_subscription(geometry_msgs.PoseStamped, 'cmd_pose', self.cmd_pose_callback, 10)
        self.sub_odometry = self.create_subscription(Odometry, 'odom', self.odometry_callback, 10)
        
        self.pub_cmd_vel = self.create_publisher(geometry_msgs.Twist, 'cmd_vel', 10)   

    #==============================================================================
    def cmd_pose_callback(self, msg):
        """Handle updated set pose callback."""
        # Just store the desired pose. The actual control runs on odometry callbacks
        p = msg.pose.position
        q = msg.pose.orientation
        # self.pos_des = np.array([p.x, p.y, p.z])
        # self.quat_des = np.array([q.x, q.y, q.z, q.w])

    #==============================================================================
    def odometry_callback(self, msg):
        """Handle updated measured velocity callback."""
        if not bool(self.config):
            return

        tic = time.perf_counter()
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        p = np.array([p.x, p.y, p.z])
        q = np.array([q.x, q.y, q.z, q.w])
        
        angles_pose = quat2euler([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z])
        pose_pid = np.array([msg.pose.pose.position.x,    msg.pose.pose.position.y,    msg.pose.pose.position.z, 
                                    angles_pose[0], angles_pose[1], angles_pose[2]])
        next_path_points = self.extract_next_path_points(self.path_points, pose_pid, 2)
        # print(f'next_path_points: {next_path_points}')

        if not self.initialized:
            # If this is the first callback: Store and hold latest pose.
            self.pos_des  = p
            self.quat_des = q
            self.initialized = True

        # print(f'next_path_points: {next_path_points}')
        # print(f'next_path_points 0: {next_path_points[0:3,0]}')
        # print(f'next_path_points 1: {next_path_points[2,0]}')
        angles_pose_quat = euler2quat(next_path_points[3,1], next_path_points[4,1],next_path_points[5,1], 'sxyz')
        self.pos_des = np.array([next_path_points[0,1], next_path_points[1,1], next_path_points[2,1]])
        self.quat_des =  np.array([angles_pose_quat[1],angles_pose_quat[2],angles_pose_quat[3],angles_pose_quat[0]])

        # print(f'blapos: {blapos_des}')
        # print(f'blaquat: {blaquat_des}')
        # print(f'self.pos_des: {self.pos_des}')
        # print(f'self.quat_des: {self.quat_des}')
        # print(f'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

        # Compute control output:
        t = time_in_float_sec_from_msg(msg.header.stamp)

        # Position error
        e_pos_world = self.pos_des - p
        e_pos_body = transf.quaternion_matrix(q).transpose()[0:3,0:3].dot(e_pos_world)

        # Error quaternion wrt body frame
        e_rot_quat = transf.quaternion_multiply(transf.quaternion_conjugate(q), self.quat_des)

        if np.linalg.norm(e_pos_world[0:2]) > 5.0:
            # special case if we are far away from goal:
            # ignore desired heading, look towards goal position
            heading = math.atan2(e_pos_world[1],e_pos_world[0])
            quat_des = np.array([0, 0, math.sin(0.5*heading), math.cos(0.5*heading)])
            e_rot_quat = transf.quaternion_multiply(transf.quaternion_conjugate(q), quat_des)
            
        # Error angles
        e_rot = np.array(transf.euler_from_quaternion(e_rot_quat))

        v_linear = self.pid_pos.regulate(e_pos_body, t)
        v_angular = self.pid_rot.regulate(e_rot, t)

        # Convert and publish vel. command:
        cmd_vel = geometry_msgs.Twist()
        cmd_vel.linear = geometry_msgs.Vector3(x=v_linear[0], y=v_linear[1], z=v_linear[2])
        cmd_vel.angular = geometry_msgs.Vector3(x=v_angular[0], y=v_angular[1], z=v_angular[2])
        self.pub_cmd_vel.publish(cmd_vel)
        toc = time.perf_counter()
        print(f'time for one callback:{toc-tic}')
        print('--------------------------------------------------------------------------')


    #==============================================================================
    def callback_params(self, data):
        """Handle updated configuration values."""
        for parameter in data:
            #if parameter.name == "name":
            #if parameter.type_ == Parameter.Type.DOUBLE:
            self.config[parameter.name] = parameter.value

        # Config has changed, reset PID controllers
        self.create_pids(self.config)

        self.get_logger().warn("Parameters dynamically changed...")
        return SetParametersResult(successful=True)

    #==============================================================================
    def create_pids(self, config):
        self.pid_pos = PIDRegulator(config['pos_p'], config['pos_i'], config['pos_d'], config['pos_sat'])
        self.pid_rot = PIDRegulator(config['rot_p'], config['rot_i'], config['rot_d'], config['rot_sat'])

    #==============================================================================
    def _declare_and_fill_map(self, key, default_value, description, map):
        param = self.declare_parameter(key, default_value, ParameterDescriptor(description=description))
        map[key] = param.value

    def calc_points_on_ellipse(self, num_points):
        """Desired trajectory on ellipoid represented by 2D points"""
        # dT = 2 * np.pi / num_points
        dT = 1 / num_points
        # print(f"dT :{dT}")
        t = np.arange(dT,(num_points+1)*dT,dT)
        t_vert = np.arange(dT,(2*num_points+1)*dT,dT)
        dT_circle = np.pi / num_points
        # t_circle = np.arange(dT,(num_points)*dT,dT/1.96)
        t_circle = np.arange(dT_circle,(num_points)*dT_circle,dT_circle/1.96)
        # t2 = np.arange(dT,(num_points+1)*dT,dT*2)
        # print(f"t :{t}")
        # print(f"t_vert :{t_vert}")
        # print(f't_circle:{t_circle}')
        # print(f't_circle y:{np.sin(np.arange(dT_circle,(num_points)*dT_circle,dT_circle))}')
        # print(f't_circle z:{-np.cos(np.arange(dT_circle,(num_points)*dT_circle,dT_circle))}')
        # print(f"t2 :{t2}")
        # path_points = np.array([0.5*np.cos(t),
                        # 2.0*np.sin(t), 0.5*np.sin(t)])
        path_points = np.hstack([np.array([0*t_vert,0*t_vert,-1*t_vert-2, 0*t_vert, 0*t_vert, 0*t_vert]), 
                                    np.array([0*np.cos(t_circle),-1.250/2*np.cos(t_circle)+1.25/2,-1.25/2*np.sin(t_circle)-2-2, 0*np.cos(t_circle), 0*np.cos(t_circle), 0*np.cos(t_circle)]), 
                                    np.array([0*t_vert,0*t_vert+1.25, 1*t_vert-2-2, 0*t_vert, 0*t_vert, 0*t_vert]),
                                    np.array([0*np.cos(t_circle),-1.250/2*np.cos(t_circle)+1.25/2+1.25,1.25/2*np.sin(t_circle)-2, 0*np.cos(t_circle), 0*np.cos(t_circle), 0*np.cos(t_circle)]),
                                    np.array([0*t_vert,0*t_vert+2.5,-1*t_vert-2, 0*t_vert, 0*t_vert, 0*t_vert]),
                                    np.array([0*np.cos(t_circle),-1.250/2*np.cos(t_circle)+1.25/2+2.5,-1.25/2*np.sin(t_circle)-2-2, 0*np.cos(t_circle), 0*np.cos(t_circle), 0*np.cos(t_circle)]),
                                    np.array([0*t_vert,0*t_vert+3.75,1*t_vert-2-2, 0*t_vert, 0*t_vert, 0*t_vert]),
                                    np.array([0*np.cos(t_circle),-1.250/2*np.cos(t_circle)+1.25/2+3.75,1.25/2*np.sin(t_circle)-2, 0*np.cos(t_circle), 0*np.cos(t_circle), 0*np.cos(t_circle)]),
                                    np.array([0*t_vert,0*t_vert+5,-1*t_vert-2, 0*t_vert, 0*t_vert, 0*t_vert]),
                                    np.array([0*np.cos(t_circle),-1.250/2*np.cos(t_circle)+1.25/2+5,-1.25/2*np.sin(t_circle)-2-2, 0*np.cos(t_circle), 0*np.cos(t_circle), 0*np.cos(t_circle)]),
                                    np.array([0*t_vert,0*t_vert+6.25,1*t_vert-2-2, 0*t_vert, 0*t_vert, 0*t_vert]),
                                    np.array([0*t,0*t+6.25,0*t-2, 0*t, 0*t, 0*t])
                                    # np.array([0*t,-6.25*t+6.25,0*t-1, 0*t, 0*t, 0*t]),
                                    # np.array([0*t_vert,0*t_vert,-0.5*t_vert-1, 0*t_vert, 0*t_vert, 0*t_vert])
                                    ])
        # path_points = np.hstack([np.array([0*t,0*t,0*t-2, 0*t, 0*t, 0*t])
        #                             ])
        return path_points
        
    
    def find_closest_point(self, points, ref_point):
        """Find the index of the closest point in points from the current car position
        points = array of points on path
        ref_point = current car position
        """
        num_points = points.shape[1]
        diff = np.transpose(points) - ref_point
        diff = np.transpose(diff)
        squared_diff = np.power(diff,2)
        squared_dist = squared_diff[0,:] + squared_diff[1,:] + squared_diff[2,:]
        return np.argmin(squared_dist)

    def extract_next_path_points(self, path_points, pos, N):
        """Extract the next N points on the path for the next N stages starting from 
        the current car position pos
        """

        idx = self.find_closest_point(path_points,pos)
        num_points = path_points.shape[1]
        num_ellipses = np.ceil((idx+N+1)/num_points)
        path_points = np.tile(path_points,(1,int(num_ellipses)))
        # print(f"path_points: {path_points}")
        # print(f"pos: {pos}")
        # print(f"N: {N}")
        # print(f"idx: {idx}")
        # print(f"num_points: {num_points}")
        # print(f"num_ellipses: {num_ellipses}")
        # print(f"path_points: {path_points}")
        # print(f"return: {path_points[:,idx+1:idx+N+1]}")


        traj_marker = Path()
        traj_marker.header.stamp = self.get_clock().now().to_msg()
        traj_marker.header.frame_id = 'world'

        # traj_marker.header.frame_id = self._trajectory.header.frame_id
        # for pnt in path_points:
        #     p_msg = PoseStamped()
        #     p_msg.header.stamp = pnt.header.stamp
        #     p_msg.header.frame_id = self._trajectory.header.frame_id
        #     p_msg.pose = pnt.pose
        #     traj_marker.poses.append(p_msg)
        for i in range(len(path_points[0,:])):
            # print(f'i:{i}')
            # print(f'len(pred_x):{len(pred_x)}')
            pose = PoseStamped()
            pose.pose.position.x = path_points[0,i]
            pose.pose.position.y = path_points[1,i]
            pose.pose.position.z = path_points[2,i]
            pose_angles = euler2quat(path_points[3,i], path_points[4,i],path_points[5,i], 'sxyz')
            pose.pose.orientation.x = pose_angles[0]
            pose.pose.orientation.y = pose_angles[1]
            pose.pose.orientation.z = pose_angles[2]
            pose.pose.orientation.w = pose_angles[3]
            traj_marker.poses.append(pose)
        # print(f'traj_marker:{traj_marker}')
        self.traj_pub.publish(traj_marker)
        return path_points[:,idx+1:idx+N+1]

#==============================================================================
def main():
    print('Starting position_control.py')
    rclpy.init()

    #try:
    if 1:
        sim_time_param = is_sim_time()

        node = PositionControllerNode('position_control', parameter_overrides=[sim_time_param])
        rclpy.spin(node)
    #except Exception as e:
    #    print('Caught exception: ' + str(e))
    #finally:
        #if rclpy.ok():
            #rclpy.shutdown()
        #print('Exiting')

#==============================================================================
if __name__ == '__main__':
    main()
