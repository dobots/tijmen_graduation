import matplotlib.pyplot as plt
import csv
import numpy as np
from matplotlib.gridspec import GridSpec
import math
from sklearn.metrics import mean_squared_error
from scipy import interpolate
from scipy.ndimage import interpolation


def calc_points_on_ellipse(num_points):
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
                                np.array([0*t_vert,0*t_vert+6.25,1*t_vert-2-2, 0*t_vert, 0*t_vert, 0*t_vert])
                                # np.array([0*t,0*t+6.25,0*t-2, 0*t, 0*t, 0*t])
                                # np.array([0*t,-6.25*t+6.25,0*t-1, 0*t, 0*t, 0*t]),
                                # np.array([0*t_vert,0*t_vert,-0.5*t_vert-1, 0*t_vert, 0*t_vert, 0*t_vert])
                                ])
    # path_points = np.hstack([np.array([0*t,0*t,0*t-2, 0*t, 0*t, 0*t])
    #                             ])
    return path_points


def find_closest_point(points, ref_point):
    """Find the index of the closest point in points from the current car position
    points = array of points on path
    ref_point = current car position
    """
    num_points = points.shape[1]
    diff = np.transpose(points) - ref_point
    diff = np.transpose(diff)
    squared_diff = np.power(diff,2)
    squared_dist = squared_diff[0,:] + squared_diff[1,:] + squared_diff[2,:]
    return np.argmin(squared_dist), squared_dist

fig = plt.figure('1') 
t_pid = []
x_pid = []
y_pid = []
z_pid = []
x_rot_pid = []
y_rot_pid = []
z_rot_pid = []
x_vel_pid = []
y_vel_pid = []
z_vel_pid = []
  
with open('pid_pose_gt.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec)
        else:
            tt = float(sec + '.' + nanosec)
        t_pid.append(tt)
        x_pid.append(float(row[4]))
        y_pid.append(float(row[5]))
        z_pid.append(float(row[6]))
        x_rot_pid.append(float(row[7]))
        y_rot_pid.append(float(row[8]))
        z_rot_pid.append(float(row[9]))
        x_vel_pid.append(float(row[47]))
        y_vel_pid.append(float(row[48]))
        z_vel_pid.append(float(row[49]))

t_pid = np.transpose(np.array([t_pid[:3675]])-t_pid[0])
x_pid = x_pid[:3675]
y_pid = y_pid[:3675]
z_pid = z_pid[:3675]
x_rot_pid = x_rot_pid[:3675]
y_rot_pid = y_rot_pid[:3675]
z_rot_pid = z_rot_pid[:3675]
x_vel_pid = x_vel_pid[:3675]
y_vel_pid = y_vel_pid[:3675]
z_vel_pid = z_vel_pid[:3675]


f_id_0_pid = []
t_id_0_pid = []
  
with open('pid_id_0.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec)
        else:
            tt = float(sec + '.' + nanosec)
        t_id_0_pid.append(tt)
        f_id_0_pid.append(float(row[3]))

f_id_1_pid = []
t_id_1_pid = []
  
with open('pid_id_1.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec)
        else:
            tt = float(sec + '.' + nanosec)
        t_id_1_pid.append(tt)
        f_id_1_pid.append(float(row[3]))

f_id_2_pid = []
t_id_2_pid = []
  
with open('pid_id_2.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec)
        else:
            tt = float(sec + '.' + nanosec)
        t_id_2_pid.append(tt)
        f_id_2_pid.append(float(row[3]))


f_id_3_pid = []
t_id_3_pid = []
  
with open('pid_id_3.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec)
        else:
            tt = float(sec + '.' + nanosec)
        t_id_3_pid.append(tt)
        f_id_3_pid.append(float(row[3]))


f_id_4_pid = []
t_id_4_pid = []
  
with open('pid_id_4.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec)
        else:
            tt = float(sec + '.' + nanosec)
        t_id_4_pid.append(tt)
        f_id_4_pid.append(float(row[3]))


f_id_5_pid = []
t_id_5_pid = []
  
with open('pid_id_5.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec)
        else:
            tt = float(sec + '.' + nanosec)
        t_id_5_pid.append(tt)
        f_id_5_pid.append(float(row[3]))


f_id_6_pid = []
t_id_6_pid = []
  
with open('pid_id_6.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec)
        else:
            tt = float(sec + '.' + nanosec)
        t_id_6_pid.append(tt)
        f_id_6_pid.append(float(row[3]))


f_id_7_pid = []
t_id_7_pid = []
  
with open('pid_id_7.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec)
        else:
            tt = float(sec + '.' + nanosec)
        t_id_7_pid.append(tt)
        f_id_7_pid.append(float(row[3]))

f_id_0_mpc = []
t_id_0_mpc = []
  
with open('mpc_id_0.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec)
        else:
            tt = float(sec + '.' + nanosec)
        if float(row[3])!=0:
            t_id_0_mpc.append(tt)
            f_id_0_mpc.append(float(row[3]))

f_id_1_mpc = []
t_id_1_mpc = []
  
with open('mpc_id_1.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec)
        else:
            tt = float(sec + '.' + nanosec)
        if float(row[3])!=0:
            t_id_1_mpc.append(tt)
            f_id_1_mpc.append(float(row[3]))
f_id_2_mpc = []
t_id_2_mpc = []
  
with open('mpc_id_2.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec) 
        else:
            tt = float(sec + '.' + nanosec)
        if float(row[3])!=0:
            t_id_2_mpc.append(tt)
            f_id_2_mpc.append(float(row[3]))

f_id_3_mpc = []
t_id_3_mpc = []
  
with open('mpc_id_3.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec) 
        else:
            tt = float(sec + '.' + nanosec)
        if float(row[3])!=0:
            t_id_3_mpc.append(tt)
            f_id_3_mpc.append(float(row[3]))

f_id_4_mpc = []
t_id_4_mpc = []
  
with open('mpc_id_4.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec)
        else:
            tt = float(sec + '.' + nanosec)
        if float(row[3])!=0:
            t_id_4_mpc.append(tt)
            f_id_4_mpc.append(float(row[3]))


f_id_5_mpc = []
t_id_5_mpc = []
  
with open('mpc_id_5.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec) 
        else:
            tt = float(sec + '.' + nanosec)
        if float(row[3])!=0:
            t_id_5_mpc.append(tt)
            f_id_5_mpc.append(float(row[3]))


f_id_6_mpc = []
t_id_6_mpc = []
  
with open('mpc_id_6.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec) 
        else:
            tt = float(sec + '.' + nanosec)
        if float(row[3])!=0:
            t_id_6_mpc.append(tt)
            f_id_6_mpc.append(float(row[3]))

f_id_7_mpc = []
t_id_7_mpc = []
  
with open('mpc_id_7.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        # print(row)
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec)
        else:
            tt = float(sec + '.' + nanosec)
        if float(row[3])!=0:
            t_id_7_mpc.append(tt)
            f_id_7_mpc.append(float(row[3]))

t_id_0_pid = np.transpose(np.array([t_id_0_pid])-t_id_0_pid[0])
t_id_1_pid = np.transpose(np.array([t_id_1_pid])-t_id_1_pid[0])
t_id_2_pid = np.transpose(np.array([t_id_2_pid])-t_id_2_pid[0])
t_id_3_pid = np.transpose(np.array([t_id_3_pid])-t_id_3_pid[0])
t_id_4_pid = np.transpose(np.array([t_id_4_pid])-t_id_4_pid[0])
t_id_5_pid = np.transpose(np.array([t_id_5_pid])-t_id_5_pid[0])
t_id_6_pid = np.transpose(np.array([t_id_6_pid])-t_id_6_pid[0])
t_id_7_pid = np.transpose(np.array([t_id_7_pid])-t_id_7_pid[0])

t_id_0_mpc = np.transpose(np.array([t_id_0_mpc])-t_id_0_mpc[0])
t_id_1_mpc = np.transpose(np.array([t_id_1_mpc])-t_id_1_mpc[0])
t_id_2_mpc = np.transpose(np.array([t_id_2_mpc])-t_id_2_mpc[0])
t_id_3_mpc = np.transpose(np.array([t_id_3_mpc])-t_id_3_mpc[0])
t_id_4_mpc = np.transpose(np.array([t_id_4_mpc])-t_id_4_mpc[0])
t_id_5_mpc = np.transpose(np.array([t_id_5_mpc])-t_id_5_mpc[0])
t_id_6_mpc = np.transpose(np.array([t_id_6_mpc])-t_id_6_mpc[0])
t_id_7_mpc = np.transpose(np.array([t_id_7_mpc])-t_id_7_mpc[0])



t_mpc = []
x_mpc = []
y_mpc = []
z_mpc = []
x_rot_mpc = []
y_rot_mpc = []
z_rot_mpc = []
x_vel_mpc = []
y_vel_mpc = []
z_vel_mpc = []
  
with open('mpc_pose_gt.csv','r') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        sec = str(row[0])
        nanosec = str(row[1])
        if len(nanosec) == 8:
            tt = float(sec + '.0' + nanosec)
        else:
            tt = float(sec + '.' + nanosec)
        t_mpc.append(tt)
        x_mpc.append(float(row[4]))
        y_mpc.append(float(row[5]))
        z_mpc.append(float(row[6]))
        x_rot_mpc.append(float(row[7]))
        y_rot_mpc.append(float(row[8]))
        z_rot_mpc.append(float(row[9]))
        x_vel_mpc.append(float(row[47]))
        y_vel_mpc.append(float(row[48]))
        z_vel_mpc.append(float(row[49]))
 
t_mpc = np.transpose(np.array([t_mpc[324:2072]])-t_mpc[324])
x_mpc = x_mpc[324:2072]
y_mpc = y_mpc[324:2072]
z_mpc = z_mpc[324:2072]
x_rot_mpc = x_rot_mpc[324:2072]
y_rot_mpc = y_rot_mpc[324:2072]
z_rot_mpc = z_rot_mpc[324:2072]
x_vel_mpc = x_vel_mpc[324:2072]
y_vel_mpc = y_vel_mpc[324:2072]
z_vel_mpc = z_vel_mpc[324:2072]

plt.plot(t_pid, z_pid)
  
# plt.xticks(rotation = 25)
plt.xlabel('time')
plt.ylabel('z-axis')
plt.title('raybot z-position')
# plt.grid()
plt.legend()

fig = plt.figure('2') 

# plt.plot(y_pid, z_pid, label='PID')
plt.plot(y_mpc, z_mpc, label='MPC')
path_points = calc_points_on_ellipse(100)
# print(f'path_points:{path_points}')
plt.plot(path_points[1,:], path_points[2,:], label='path', color='k', linestyle='dashed')  
# plt.xticks(rotation = 25)
plt.xlabel('y [m]')
plt.ylabel('z [m]')
plt.title('raybot y and z-position')
# plt.grid()
plt.legend()

fig = plt.figure('2.2') 

plt.plot(y_pid, z_pid, label='PID')
# plt.plot(y_mpc, z_mpc, label='MPC')
path_points = calc_points_on_ellipse(100)
# print(f'path_points:{path_points}')
plt.plot(path_points[1,:], path_points[2,:], label='path', color='k', linestyle='dashed')  
# plt.xticks(rotation = 25)
plt.xlabel('y [m]')
plt.ylabel('z [m]')
plt.title('raybot y and z-position')
# plt.grid()
plt.legend()

### KPIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII-------------------------------------------------------------------------------------------------
#RMSE PID y

path_inter_pid_y = interpolation.zoom(path_points[1,:], np.size(y_pid)/np.size(path_points[1,:]))
# print(np.size(y_pid))
# print(np.size(path_inter_pid_y))
MSE = np.square(np.subtract(y_pid, path_inter_pid_y)).mean()
RMSE = math.sqrt(MSE)
print(f'RMSE path pid y: {RMSE}')

path_mpc_y = path_points[1,69:]
path_inter_mpc_y = interpolation.zoom(path_mpc_y, np.size(y_mpc)/np.size(path_mpc_y))
# print(np.size(y_mpc))
# print(np.size(path_inter_mpc_y))
# print(f'path mpc: {path_inter_mpc_y}')
# print(f'y_mpc: {y_mpc}')
MSE = np.square(np.subtract(y_mpc, path_inter_mpc_y)).mean()
RMSE = math.sqrt(MSE)
print(f'RMSE path mpc y: {RMSE}')

rmse_table = np.zeros([100,20])
for i in range(100):
    path_mpc_z = path_points[2,i:]
    for ii in range(20):
        zz_mpc = z_mpc[0:-ii-1]
        # print(f'path_mpc_z:{path_mpc_z}')
        path_inter_mpc_z = interpolation.zoom(path_mpc_z, np.size(zz_mpc)/np.size(path_mpc_z))
        # print(np.size(z_mpc))
        # print(np.size(path_inter_mpc_z))
        # print(f'path mpc z: {np.transpose(path_inter_mpc_z)}')
        # print(f'z_mpc: {z_mpc}')
        MSE = np.square(np.subtract(zz_mpc, path_inter_mpc_z)).mean()
        RMSE = math.sqrt(MSE)
        # print(f'RMSE path mpc z: {RMSE}')
        rmse_table[i,ii] = RMSE
print(f'rmse_table:{rmse_table}')
indexes = np.unravel_index(np.argmin(rmse_table, axis=None), rmse_table.shape)
print(f'indexes: {indexes}')
print(f'lowest rmse: {rmse_table[indexes]}')

mean_dist_array_mpc = np.zeros([np.size(z_mpc),1])
print(f'mean dist array:{mean_dist_array_mpc}')
path_points_rmse_mpc = calc_points_on_ellipse(1000)
print(f'np.size(z_mpc): {np.size(z_mpc)}')
for i in range(np.size(z_mpc)):
    # print(i)
    closest_point, squared_dist = find_closest_point(path_points_rmse_mpc, [x_mpc[i], y_mpc[i], z_mpc[i], x_rot_mpc[i],y_rot_mpc[i],z_rot_mpc[i]])
    # print(f'squared_dist: {np.shape(squared_dist)}')
    # print(f'closest_point: {closest_point}')
    mean_dist_array_mpc[i] = squared_dist[closest_point]
mean_squared_error_mpc = np.mean(mean_dist_array_mpc)


mean_dist_array_pid = np.zeros([np.size(z_pid),1])
print(f'mean dist array:{mean_dist_array_mpc}')
path_points_rmse_pid = calc_points_on_ellipse(1000)
print(f'np.size(z_mpc): {np.size(z_pid)}')
for i in range(np.size(z_pid)):
    # print(i)
    closest_point, squared_dist = find_closest_point(path_points_rmse_pid, [x_pid[i], y_pid[i], z_pid[i], x_rot_pid[i],y_rot_pid[i],z_rot_pid[i]])
    # print(f'squared_dist: {np.shape(squared_dist)}')
    # print(f'closest_point: {closest_point}')
    mean_dist_array_pid[i] = squared_dist[closest_point]
mean_squared_error_pid = np.mean(mean_dist_array_pid)


print(f'mean_squared_error: {mean_squared_error_mpc}')
print(f'completion_time mpc : {t_mpc[-1]-t_mpc[0]}')

print(f'mean_squared_error: {mean_squared_error_pid}')
print(f'completion_time mpc : {t_pid[-1]-t_pid[0]}')



### KPIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII-------------------------------------------------------------------------------------------------

plt.figure('4') 

plt.plot(np.arange(0, np.size(y_mpc),1), path_inter_mpc_y, label='MPC path')
plt.plot(np.arange(0, np.size(y_mpc),1),y_mpc, label='MPC')
# path_points = calc_points_on_ellipse(100)
# print(f'path_points:{path_points}')
# plt.plot(path_points[1,:], path_points[2,:], label='path', color='k', linestyle='dashed')  
# plt.xticks(rotation = 25)
plt.xlabel('y [m]')
plt.ylabel('z [m]')
plt.title('raybot y and z-position')
# plt.grid()
plt.legend()


#### ------------------ big figure --------------------------
fig = plt.figure('3')
plt.clf()
gs = GridSpec(6,4,figure=fig)

# Plot trajectory xy
# axy_pos = fig.add_subplot(gs[:,0:2])
ayz_pos = fig.add_subplot(gs[0:3,0:2])
l1, = ayz_pos.plot(y_pid, z_pid, label='PID')
l2, = ayz_pos.plot(y_mpc, z_mpc, label='MPC')
l0, = ayz_pos.plot(path_points[1,:], path_points[2,:], label='path', color='k', linestyle='dashed')
plt.title('Position yz')
# plt.axis('equal')
# plt.xlim([-0.5,1])
# plt.ylim([-1, 5])
plt.xlabel('y [m]')
plt.ylabel('z [m]')
plt.legend()
# l2, = axy_pos.plot(x[0,0],x[1,0],'b-')
# l3, = axy_pos.plot(start_pred[8,:], start_pred[9,:],'g-')
# rect1 = Rectangle(((0-(0.33/2)),(0-(1.2/2))),0.33,1.2,
#             angle=np.rad2deg(xinit[2+3]),
#             edgecolor='black',
#             facecolor='none',
#             lw=0.5)
# axy_pos.add_patch(rect1)
# rect2 = Rectangle(((0-(0.33/2)),(0-(1.2/2))),0.33,1.2,
#             angle=np.rad2deg(xinit[2+3]),
#             edgecolor='grey',
#             facecolor='none',
#             lw=0.3)
# axy_pos.add_patch(rect2)
# axy_pos.legend([l0,l1,l2,l3],['desired trajectory','init pos','robot trajectory',\
#     'predicted robot traj.'],loc='lower right')
axy_pos = fig.add_subplot(gs[3:6,0:2])
l1, = axy_pos.plot(y_pid, x_pid, label='PID')
l2, = axy_pos.plot(y_mpc, x_mpc, label='MPC')
l0, = axy_pos.plot(path_points[1,:], path_points[0,:], label='path', color='k', linestyle='dashed')
plt.title('Position xy')
# plt.axis('equal')
# plt.xlim([-0.5,1])
# plt.ylim([-1, 5])
plt.xlabel('y [m]')
plt.ylabel('x [m]')
plt.legend()


# Plot velocity
ax_velx = fig.add_subplot(6,4,3)
plt.grid("both")
plt.title('Velocity X')
ax_velx.plot(t_pid,x_vel_pid, label='PID')
ax_velx.plot(t_mpc,x_vel_mpc, label='MPC')
plt.xlabel('t [s]')
plt.ylabel('vel x [m/s]')
plt.legend()

# Plot velocity
ax_vely = fig.add_subplot(6,4,7)
plt.grid("both")
plt.title('Velocity Y')
ax_vely.plot(t_pid,y_vel_pid, label='PID')
ax_vely.plot(t_mpc,y_vel_mpc, label='MPC')
plt.xlabel('t [s]')
plt.ylabel('vel y [m/s]')
plt.legend()

# Plot velocity
ax_velz = fig.add_subplot(6,4,11)
plt.grid("both")
plt.title('Velocity Z')
ax_velz.plot(t_pid,z_vel_pid, label='PID')
ax_velz.plot(t_mpc,z_vel_mpc, label='MPC')
plt.xlabel('t [s]')
plt.ylabel('vel z [m/s]')
plt.legend()


# Plot force 0
ax_velz = fig.add_subplot(6,4,15)
plt.grid("both")
plt.title('Force thruster 0 [N]')
ax_velz.plot(t_id_0_pid,f_id_0_pid, label='PID')
ax_velz.plot(t_id_0_mpc,f_id_0_mpc, label='MPC')
plt.plot([t_id_0_pid[0], t_id_0_pid[-1]], np.transpose([36, 36]), 'r:')
plt.plot([t_id_0_pid[0], t_id_0_pid[-1]], np.transpose([-28, -28]), 'r:')
plt.xlabel('t [s]')
plt.ylabel('force thruster 0 [N]')
plt.legend()

# Plot force 1
ax_velz = fig.add_subplot(6,4,19)
plt.grid("both")
plt.title('Force thruster 1 [N]')
ax_velz.plot(t_id_1_pid,f_id_1_pid, label='PID')
ax_velz.plot(t_id_1_mpc,f_id_1_mpc, label='MPC')
plt.plot([t_id_1_pid[0], t_id_1_pid[-1]], np.transpose([36, 36]), 'r:')
plt.plot([t_id_1_pid[0], t_id_1_pid[-1]], np.transpose([-28, -28]), 'r:')
plt.xlabel('t [s]')
plt.ylabel('force thruster 1 [N]')
plt.legend()

# Plot force 2
ax_velz = fig.add_subplot(6,4,23)
plt.grid("both")
plt.title('Force thruster 2 [N]')
ax_velz.plot(t_id_2_pid,f_id_2_pid, label='PID')
ax_velz.plot(t_id_2_mpc,f_id_2_mpc, label='MPC')
plt.plot([t_id_2_pid[0], t_id_2_pid[-1]], np.transpose([36, 36]), 'r:')
plt.plot([t_id_2_pid[0], t_id_2_pid[-1]], np.transpose([-28, -28]), 'r:')
plt.xlabel('t [s]')
plt.ylabel('force thruster 2 [N]')
plt.legend()

# Plot force 3
ax_velz = fig.add_subplot(6,4,4)
plt.grid("both")
plt.title('Force thruster 3 [N]')
ax_velz.plot(t_id_3_pid,f_id_3_pid, label='PID')
ax_velz.plot(t_id_3_mpc,f_id_3_mpc, label='MPC')
plt.plot([t_id_3_pid[0], t_id_3_pid[-1]], np.transpose([36, 36]), 'r:')
plt.plot([t_id_3_pid[0], t_id_3_pid[-1]], np.transpose([-28, -28]), 'r:')
plt.xlabel('t [s]')
plt.ylabel('force thruster 3 [N]')
plt.legend()


# Plot force 4
ax_velz = fig.add_subplot(6,4,8)
plt.grid("both")
plt.title('Force thruster 4 [N]')
ax_velz.plot(t_id_4_pid,f_id_4_pid, label='PID')
ax_velz.plot(t_id_4_mpc,f_id_4_mpc, label='MPC')
plt.plot([t_id_4_pid[0], t_id_4_pid[-1]], np.transpose([36, 36]), 'r:')
plt.plot([t_id_4_pid[0], t_id_4_pid[-1]], np.transpose([-28, -28]), 'r:')
plt.xlabel('t [s]')
plt.ylabel('force thruster 4 [N]')
plt.legend()

# Plot force 5
ax_velz = fig.add_subplot(6,4,12)
plt.grid("both")
plt.title('Force thruster 5 [N]')
ax_velz.plot(t_id_5_pid,f_id_5_pid, label='PID')
ax_velz.plot(t_id_5_mpc,f_id_5_mpc, label='MPC')
plt.plot([t_id_5_pid[0], t_id_5_pid[-1]], np.transpose([36, 36]), 'r:')
plt.plot([t_id_5_pid[0], t_id_5_pid[-1]], np.transpose([-28, -28]), 'r:')
plt.xlabel('t [s]')
plt.ylabel('force thruster 5 [N]')
plt.legend()

# Plot force 6
ax_velz = fig.add_subplot(6,4,16)
plt.grid("both")
plt.title('Force thruster 6 [N]')
ax_velz.plot(t_id_6_pid,f_id_6_pid, label='PID')
ax_velz.plot(t_id_6_mpc,f_id_6_mpc, label='MPC')
plt.plot([t_id_6_pid[0], t_id_6_pid[-1]], np.transpose([36, 36]), 'r:')
plt.plot([t_id_6_pid[0], t_id_6_pid[-1]], np.transpose([-28, -28]), 'r:')
plt.xlabel('t [s]')
plt.ylabel('force thruster 6 [N]')
plt.legend()

# Plot force 7
ax_velz = fig.add_subplot(6,4,20)
plt.grid("both")
plt.title('Force thruster 7 [N]')
ax_velz.plot(t_id_7_pid,f_id_7_pid, label='PID')
ax_velz.plot(t_id_7_mpc,f_id_7_mpc, label='MPC')
plt.plot([t_id_7_pid[0], t_id_7_pid[-1]], np.transpose([36, 36]), 'r:')
plt.plot([t_id_7_pid[0], t_id_7_pid[-1]], np.transpose([-28, -28]), 'r:')
plt.xlabel('t [s]')
plt.ylabel('force thruster 7 [N]')
plt.legend()








plt.show()

