#!/usr/bin/env python3
# Example script for getting started with FORCESPRO NLP solver.

# This example solves an optimization problem for a car with the simple
# continuous-time, nonlinear dynamics (bicycle model):

#    dxPos/dt = v*cos(theta + beta)
#    dyPos/dt = v*sin(theta + beta)
#    dv/dt = F/m
#    dtheta/dt = v/l_r*sin(beta)
#    ddelta/dt = phi

#    with:
#    beta = arctan(l_r/(l_f + l_r)*tan(delta))

# where xPos,yPos are the position, v the velocity in heading angle theta 
# of the car, and delta is the steering angle relative to the heading 
# angle. The inputs are acceleration force F and steering rate phi. The 
# physical constants m, l_r and l_f denote the car's mass and the distance 
# from the car's center of gravity to the rear wheels and the front wheels.

# The car starts from standstill with a certain heading angle, and the
# optimization problem is to minimize the distance of the car's position 
# to a given set of points on a path with respect to time.

# Quadratic costs for the acceleration force and steering rate are added to
# the objective to avoid excessive maneouvers.

# There are bounds on all variables except theta.

# Variables are collected stage-wise into 

#     z = [F phi xPos yPos v theta delta].

# This example models the task as a MPC problem using the SQP method.

# See also FORCES_NLP

# (c) Embotech AG, Zurich, Switzerland, 2013-2022.


import sys
import numpy as np
from plankton_utils.time import is_sim_time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

import casadi
sys.path.insert(0, '/home/tijmen/forcespro/forces_pro_client') # On Unix: CHANGE THIS TO OWN LOCAL FOLDER OF FORCESPRO
import forcespro
import forcespro.nlp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import casadi
from mpl_toolkits import mplot3d
from matplotlib.patches import Rectangle
from function_file_raybot_v7 import *
from generate_raybot_solver_v7 import generate_pathplanner

from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from uuv_gazebo_ros_plugins_msgs.msg import FloatStamped
from uuv_thrusters.models import Thruster
import time
from transforms3d.euler import quat2euler
from transforms3d.euler import euler2quat
import csv

class MPCPublisherSubscriber(Node):

    def __init__(self):
        super().__init__('mpc_controller')
        self.publisher_ = self.create_publisher(Path, '/raybot/planned_path', 10)
        self.traj_pub = self.create_publisher(Path, '/raybot/trajectory_marker', 10) 
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.subscription = self.create_subscription(
            Odometry,
            '/raybot/pose_gt',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.pose_twist = np.zeros(12)
        self.pose_quat = np.zeros(13)
        self.pose_quat[6] = 1
        self.pose_quat[2] = -2
        self.pose_twist[2] = -1.9
        self.pose_twist[1] = 0.1
        # self.pose_twist[5] = 1.57
        self.n_thrusters = 8
        self.thrusters = list()
        for i in range(self.n_thrusters):
            topic = '/raybot/thrusters/id_' + str(i) + '/input'
            thruster = Thruster.create_thruster(self, 'proportional', i, topic, None, None, **{'gain': 3.3e-06})
            self.thrusters.append(thruster)
        # print(f' self.thrusters ={self.thrusters}-------------------------------------------')
        self.plot_results = 0
        self.publish_path = 0
        # generate code for estimator
        self.create_new_solver = 1
        self.model, self.solver = generate_pathplanner(self.create_new_solver)        
        
        # Create 2D points on ellipse which the robot is supposed to follow
        self.num_points = 20
        self.path_points = calc_points_on_ellipse(self.num_points)
        self.comp_time = np.empty((0,1))
        self.comp_time_list = []
        self.u_list = []
        self.x_list = []
        self.time_list = []
        self.results = np.array([])
        self.driekeer = 0


    def listener_callback(self, msg):
        # angles_twist = quat2euler([msg.twist.twist.angular.w, msg.twist.twist.angular.x,msg.twist.twist.angular.y,msg.twist.twist.angular.z])
        # self.pose_quat = np.array([msg.pose.pose.position.x,    msg.pose.pose.position.y,    msg.pose.pose.position.z, 
        #                             msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,
        #                             msg.twist.twist.linear.x,    msg.twist.twist.linear.y,    msg.twist.twist.linear.z, 
        #                             msg.twist.twist.angular.x,msg.twist.twist.angular.y,msg.twist.twist.angular.z])
        angles_pose = quat2euler([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z])
        angles_twist = np.array([msg.twist.twist.angular.x,msg.twist.twist.angular.y,msg.twist.twist.angular.z])
        # angles_pose = np.array([msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z])
        self.pose_twist = np.array([msg.pose.pose.position.x,    msg.pose.pose.position.y,    msg.pose.pose.position.z, 
                                    angles_pose[0], angles_pose[1], angles_pose[2],
                                    msg.twist.twist.linear.x,    msg.twist.twist.linear.y,    msg.twist.twist.linear.z, 
                                    angles_twist[0],   angles_twist[1],   angles_twist[2]])
        # print(f' angles_pose: {angles_pose}')
        # self.get_logger().info('I heard twist_twist "%s"' % angles_twist)
        # self.get_logger().info('I heard pose_twist "%s"' % self.pose_twist)

    def timer_callback(self):
        # msg = String()
        # msg.data = 'Hello World: %d' % self.i
        # self.publisher_.publish(msg)
        # self.get_logger().info('Publishing: "%s"' % msg.data)
        # self.get_logger().info('Publishing x_linear: "%s"' % self.pose_twist)
        self.i += 1
        tic = time.perf_counter()
        # print(f' time tic: {tic}')

        # plot_results = 0
        # # generate code for estimator
        # create_new_solver = 0
        # model, solver = generate_pathplanner(create_new_solver)

        # Simulation
        # ----------
        sim_length = 1 # simulate 8sec

        # Variables for storing simulation data
        x = np.zeros((self.model.neq,sim_length+1)) # states
        u = np.zeros((self.model.nvar-self.model.neq,sim_length)) # inputs

        # Set initial guess to start solver from
        x0i = np.zeros((self.model.nvar,1))
        x0 = np.transpose(np.tile(x0i, (1, self.model.N)))
        # print(f'x0:{x0}')
        # Set initial condition
        # xinit = np.transpose(np.array([0, 0, 0, 0., 0., 0.,0, 0, 0, 0., 0., 0.]))
        xinit = self.pose_twist
        # xinit = self.pose_quat
        # print(f'xinitt: {xinit}')
        x[:,0] = xinit

        problem = {"x0": x0,
                "xinit": xinit}

        # # Create 2D points on ellipse which the robot is supposed to follow
        # num_points = 180
        # path_points = calc_points_on_ellipse(num_points)

        start_pred = np.reshape(problem["x0"],(self.model.nvar,self.model.N)) # first prediction corresponds to initial guess

        if self.plot_results:
            # generate plot with initial values
            createPlot(x,u,start_pred,sim_length,self.model,self.path_points,xinit)
        if not self.publish_path:
            traj_marker = Path()
            traj_marker.header.stamp = self.get_clock().now().to_msg()
            traj_marker.header.frame_id = 'world'
            for i in range(len(self.path_points[0,:])):
                # print(f'i:{i}')
                # print(f'len(pred_x):{len(pred_x)}')
                pose = PoseStamped()
                pose.pose.position.x = self.path_points[0,i]
                pose.pose.position.y = self.path_points[1,i]
                pose.pose.position.z = self.path_points[2,i]
                pose_angles = euler2quat(self.path_points[3,i], self.path_points[4,i],self.path_points[5,i], 'sxyz')
                pose.pose.orientation.x = pose_angles[0]
                pose.pose.orientation.y = pose_angles[1]
                pose.pose.orientation.z = pose_angles[2]
                pose.pose.orientation.w = pose_angles[3]
                traj_marker.poses.append(pose)
            self.publish_path = 1
            # print(f'traj_marker:{traj_marker}')
            self.traj_pub.publish(traj_marker)
            
    
        # Simulation
        for k in range(sim_length):
            
            # Set initial condition
            # problem["xinit"] = x[:,k]
            problem["xinit"] = self.pose_twist
            # Set runtime parameters (here, the next N points on the path)
            next_path_points = extract_next_path_points(self.path_points, x[0:self.model.npar,k], self.model.N)
            # print(f'next_path_points: = {next_path_points}')
            problem["all_parameters"] = np.reshape(np.transpose(next_path_points), \
                (self.model.npar*self.model.N,1))
            print(f'next_path_points - xinit: {next_path_points[:,0]- xinit[0:6]}')
            # print(f"problem {problem}")
            print('Solve', k, problem["xinit"])

            # Time to solve the NLP!
            output, exitflag, info = self.solver.solve(problem)
            print(f"exitflag = {exitflag}")
            # print(f'output: {output}')
            # Make sure the solver has exited properly.
            assert exitflag == 1, "bad exitflag"
            sys.stderr.write("FORCESPRO took {} iterations and {} seconds to solve the problem.\n"\
                .format(info.it, info.solvetime))

            # Extract output
            temp = np.zeros((np.max(self.model.nvar), self.model.N))
            for i in range(0, self.model.N):
                temp[:, i] = output['x{0:02d}'.format(i+1)]
            pred_u = temp[0:self.model.nvar-self.model.neq, :]
            pred_x = temp[self.model.nvar-self.model.neq:self.model.nvar, :]

            # Apply optimized input u of first stage to system and save simulation data
            u[:,k] = pred_u[:,0] #+ np.random.normal(0,0.1,1)                                               #CAN INPUT DISTURBANCES HERE!!!
            x[:,k+1] = np.transpose(self.model.eq(np.concatenate((u[:,k],x[:,k]))))
            # casadi.horzcat
            # x[:,k+1] = np.transpose(self.model.eq(casadi.horzcat((u[:,k],x[:,k]))))
            # x[:,k+1] = self.model.eq(casadi.horzcat((u[:,k],x[:,k])))
            print(f'u at k = {u[:,k]}')
            print(f'x at k = {x[:,k]}')

            for i in range(self.n_thrusters):
                self.thrusters[i].publish_command(u[i,k])

            # print(f'pred_x x= {pred_x[0,:]}')
            # print(f'pred_x y= {pred_x[1,:]}')
            # print(f'pred_x z= {pred_x[2,:]}')
            # print('--------------------------------------------------------------------------')
            if 0:
                msg = Path()
                msg.header.frame_id = "/map"
                msg.header.stamp = self.get_clock().now().to_msg()
                for i in range(len(pred_x[0,:])):
                    # print(f'i:{i}')
                    # print(f'len(pred_x):{len(pred_x)}')
                    pose = PoseStamped()
                    pose.pose.position.x = pred_x[0,i]
                    pose.pose.position.y = pred_x[1,i]
                    pose.pose.position.z = pred_x[2,i]
                    pose_angles = euler2quat(pred_x[3,i], pred_x[4,i],pred_x[5,i], 'sxyz')
                    pose.pose.orientation.x = pose_angles[0]
                    pose.pose.orientation.y = pose_angles[1]
                    pose.pose.orientation.z = pose_angles[2]
                    pose.pose.orientation.w = pose_angles[3]
                    msg.poses.append(pose)
                self.publisher_.publish(msg)
                # self.get_logger().info('Publishing: "%s"' % msg.poses[0].pose.position)

            if self.plot_results:
                # plot results of current simulation step
                updatePlots(x,u,pred_x,pred_u,self.model,k)
            
                if k == sim_length-1:
                    fig = plt.figure('1')
                    ax_list = fig.axes
                    ax_list[0].get_lines().pop(-1).remove()   # remove old prediction of trajectory
                    ax_list[0].legend(['desired trajectory','init pos','robot trajectory'], \
                        loc='lower right')
                    ax_list[1].get_lines().pop(-1).remove()   # remove old prediction of trajectory
                    ax_list[1].legend(['desired trajectory','init pos','robot trajectory'], \
                        loc='lower right')
                    plt.show()
                else:
                    plt.draw()
            toc = time.perf_counter()
            # np.append(self.comp_time, np.array([[toc-tic]]), axis=0)

            self.comp_time_list.append([toc-tic])
            self.time_list.append(tic)
            # self.time_list[-1] = self.time_list[-1] - self.time_list[0]
            self.u_list.append(u[:,k])
            self.x_list.append(x[:,k])
            self.driekeer = self.driekeer + 1
            # if (x[2,k] < -12.5):   
            if (x[1,k] > 6 and x[2,k] >= -2 and 0):
                print(f'finito!!------------------------------------------------------------------------------------------------')
                # print(f'comp_time:{self.comp_time}')
                # print(f'comp_time list: {self.comp_time_list}')
                comp_time_list_np = np.asarray(self.comp_time_list)
                time_list_np = np.asarray(self.time_list)
                time_list_np = time_list_np - time_list_np[0]
                print(f'comp_time list np: {comp_time_list_np}')
                x_list_np = np.zeros((len(self.comp_time_list),12))
                u_list_np = np.zeros((len(self.comp_time_list),8))
                for i in range(len(self.comp_time_list)):
                    x_list_np[i,:] = np.asarray(self.x_list[i]) 
                    u_list_np[i,:] = np.asarray(self.u_list[i])
                # print(f'x_list: {self.x_list}')
                # print(f'shape x list:{len(self.x_list)}')
                # print(f'shape x list:{len(self.x_list[0])}')
                # print(f'np asarray x_list:{np.asarray(self.x_list[i])} ')
                print(f'x_list np:{x_list_np}')
                print(f'u_list np:{u_list_np}')
                # csvdata = comp_time_list_np + x_list_np + u_list_np
                # print(f'csvdata: {csvdata}')
                print(f'shape comptime list: {np.shape(comp_time_list_np)}')
                print(f'shape x list: {np.shape(x_list_np)}')
                print(f'shape u list: {np.shape(u_list_np)}')
                csvdata = np.column_stack((time_list_np,comp_time_list_np,x_list_np, u_list_np))
                # print(f'csvdata: {csvdata}')
                with open('experiment_2.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(csvdata)
                checkpoint_reached = 0
                assert checkpoint_reached == 1, "checkpoint reached"
                
            print(f'time for one callback:{toc-tic}')
            print('--------------------------------------------------------------------------')

def main(args=None):
    print('starting run_raybot_solver_v7.py')
    rclpy.init(args=args)

    mpc_publisher_subscriber = MPCPublisherSubscriber()

    rclpy.spin(mpc_publisher_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    mpc_publisher_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

