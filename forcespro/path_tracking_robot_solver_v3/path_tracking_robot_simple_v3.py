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
import casadi
sys.path.insert(0, '/home/tijmen/forcespro/forces_pro_client') # On Unix: CHANGE THIS TO OWN LOCAL FOLDER OF FORCESPRO
import forcespro
import forcespro.nlp
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.gridspec import GridSpec
import casadi
from mpl_toolkits import mplot3d


def continuous_dynamics(x, u):
    """Defines dynamics of the underwater robot
    parameters:
    state x = [xPos, yPos, zPos, Vx, Vy, Vz]
    input u = [Fx, Fy, Fz]
    """
    m = 1.0

    return casadi.vertcat(  x[3],       # dxPos/dt = Vx
                            x[4],       # dyPos/dt = Vy
                            x[5],       # dyPos/dt = Vz
                            u[0] / m,   # dVx/dt = Fx/m
                            u[1] / m,   # dVy/dt = Fy/m    
                            u[2] / m)   # dVz/dt = Fz/m                

def obj(z,current_target):
    """Least square costs on deviating from the path and on the inputs F and phi
    z = [Fx,Fy,Yz,xPos, yPos, zPos, Vx, Vy, Vz]
    current_target = point on path that is to be headed for
    """
    return (100.0*(z[3]-current_target[0])**2 # costs on deviating on the
#                                              path in x-direction
            + 100.0*(z[4]-current_target[1])**2 # costs on deviating on the
#                                               path in y-direction
            + 100*(z[5]-current_target[2])**2 # costs on deviating on the
 #                                               path in z-direction
            + 0.1*z[0]**2 # penalty on input Fx
            + 0.1*z[1]**2 # penalty on input Fy
            + 0.1*z[2]**2) # penalty on input Fz

def objN(z,current_target):
    """Increased least square costs for last stage on deviating from the path and 
    on the inputs F and phi
    z = [Fx,Fy,xPos, yPos, Vx, Vy]
    current_target = point on path that is to be headed for
    """
    return (200.0*(z[3]-current_target[0])**2 # costs on deviating on the
#                                              path in x-direction
        + 200.0*(z[4]-current_target[1])**2 # costs on deviating on the
#                                               path in y-direction
        + 200*(z[5]-current_target[2])**2 # costs on deviating on the
#                                               path in z-direction
        + 0.2*z[0]**2 # penalty on input Fx
        + 0.2*z[1]**2 # penalty on input Fy
        + 0.2*z[2]**2) # penalty on input Fz

def calc_points_on_ellipse(num_points):
    """Desired trajectory on ellipoid represented by 2D points"""
    dT = 2 * np.pi / num_points
    t = np.arange(dT,(num_points+1)*dT,dT)
    path_points = np.array([0.5*np.cos(t),
                    2.0*np.sin(t), 0.5*np.sin(t)])
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
    return np.argmin(squared_dist)

def extract_next_path_points(path_points, pos, N):
    """Extract the next N points on the path for the next N stages starting from 
    the current car position pos
    """

    idx = find_closest_point(path_points,pos)
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
    return path_points[:,idx+1:idx+N+1]


def generate_pathplanner():
    """Generates and returns a FORCESPRO solver that calculates a path based on
    constraints and dynamics while minimizing an objective function
    """
    # Model Definition
    # ----------------

    # Problem dimensions
    model = forcespro.nlp.SymbolicModel()
    model.N = 10  # horizon length
    model.nvar = 9  # number of variables = inputs + states
    model.neq = 6  # number of equality constraints = number of states
    model.npar = 3 # number of runtime parameters (number of params that are passed onto parametersStructure()...)

    # Objective function
    model.objective = obj
    model.objectiveN = objN # increased costs for the last stage
    # The function must be able to handle symbolic evaluation,
    # by passing in CasADi symbols. This means certain numpy funcions are not
    # available.

    # We use an explicit RK4 integrator here to discretize continuous dynamics
    integrator_stepsize = 0.1
    model.eq = lambda z: forcespro.nlp.integrate(continuous_dynamics, z[model.npar:model.nvar], z[0:model.npar],
                                                integrator=forcespro.nlp.integrators.RK4,
                                                stepsize=integrator_stepsize)
    # Indices on LHS of dynamical constraint - for efficiency reasons, make
    # sure the matrix E has structure [0 I] where I is the identity matrix.
    model.E = np.concatenate([np.zeros((model.neq,model.npar)), np.eye(model.neq)], axis=1)

    # Inequality constraints
    #  upper/lower variable bounds lb <= z <= ub
    #                     inputs                 |  states
    #                     Fx    Fy    Fz     x      y    z    Vx    Vy    Vz  
    model.lb = np.array([-10.,  -10.,  -10.,   -2.,  -2.,  -2.,  -4., -4., -4.])
    model.ub = np.array([+10.,  +10.,   10.,    2.,   2.,   2.,   4.,  4.,  4.])

    # Initial condition on vehicle states x
    model.xinitidx = range(model.npar,model.nvar) # use this to specify on which variables initial conditions
    # are imposed

    # Solver generation
    # -----------------

    # Set solver options
    codeoptions = forcespro.CodeOptions('FORCESNLPsolver')
    codeoptions.maxit = 200     # Maximum number of iterations
    codeoptions.printlevel = 0  # Use printlevel = 2 to print progress (but 
    #                             not for timings)
    codeoptions.optlevel = 0    # 0 no optimization, 1 optimize for size, 
    #                             2 optimize for speed, 3 optimize for size & speed
    codeoptions.cleanup = False
    codeoptions.timing = 1
    codeoptions.nlp.hessian_approximation = 'bfgs'
    codeoptions.solvemethod = 'SQP_NLP' # choose the solver method Sequential 
    #                              Quadratic Programming 'SQP_NLP' 
    codeoptions.nlp.bfgs_init = 2.5*np.identity(model.neq)
    codeoptions.sqp_nlp.maxqps = 1      # maximum number of quadratic problems to be solved
    codeoptions.sqp_nlp.reg_hessian = 5e-9 # increase this if exitflag=-8
    # change this to your server or leave uncommented for using the 
    # standard embotech server at https://forces.embotech.com 
    # codeoptions.server = 'https://forces.embotech.com'
    
    # Creates code for symbolic model formulation given above, then contacts 
    # server to generate new solver
    solver = model.generate_solver(options=codeoptions)

    return model,solver


def updatePlots(x,u,pred_x,pred_u,model,k):
    """Deletes old data sets in the current plot and adds the new data sets 
    given by the arguments x, u and predicted_z to the plot.
    x: matrix consisting of a set of state column vectors
    u: matrix consisting of a set of input column vectors
    pred_x: predictions for the next N state vectors
    pred_u: predictions for the next N input vectors
    model: model struct required for the code generation of FORCESPRO
    k: simulation step
    """
    # print(f" x: {x}")
    # print(f" u: {u}")
    # print(f" pred_x: {pred_x}")
    # print(f" pred_u: {pred_u}")
    fig = plt.figure('1')
    ax_list = fig.axes
    
    # Delete old data in plot
    ax_list[0].get_lines().pop(-1).remove() # remove old prediction of trajectory xy
    ax_list[0].get_lines().pop(-1).remove() # remove old trajectory xy 

    ax_list[1].get_lines().pop(-1).remove() # remove old prediction of trajectory xz 
    ax_list[1].get_lines().pop(-1).remove() # remove old trajectory xz

    ax_list[2].get_lines().pop(-1).remove() # remove old prediction of velocity x
    ax_list[2].get_lines().pop(-1).remove() # remove old velocity x

    ax_list[3].get_lines().pop(-1).remove() # remove old prediction of velocity y
    ax_list[3].get_lines().pop(-1).remove() # remove old velocity y

    ax_list[4].get_lines().pop(-1).remove() # remove old prediction of velocity y
    ax_list[4].get_lines().pop(-1).remove() # remove old velocity y

    ax_list[5].get_lines().pop(-1).remove() # remove old prediction of force x
    ax_list[5].get_lines().pop(-1).remove() # remove old force x

    ax_list[6].get_lines().pop(-1).remove() # remove old prediction of force y
    ax_list[6].get_lines().pop(-1).remove() # remove old force y

    ax_list[7].get_lines().pop(-1).remove() # remove old prediction of force z 
    ax_list[7].get_lines().pop(-1).remove() # remove old force z

    # Update plot with current simulation data
    ax_list[0].plot(x[0,0:k+2],x[1,0:k+2], '-b')             # plot new trajectory
    ax_list[0].plot(pred_x[0,1:], pred_x[1,1:], 'g-')        # plot new prediction of trajectory

    ax_list[1].plot(x[0,0:k+2],x[2,0:k+2], '-b')             # plot new trajectory xz
    ax_list[1].plot(pred_x[0,1:], pred_x[2,1:], 'g-')        # plot new prediction of trajectory xz

    ax_list[2].plot(x[3,0:k+2],'b-')                         # plot new velocity x
    ax_list[2].plot(range(k+1,k+model.N), pred_x[3,1:],'g-') # plot new prediction of velocity x

    ax_list[3].plot(x[4,0:k+2],'b-')                         # plot new velocity y
    ax_list[3].plot(range(k+1,k+model.N), pred_x[4,1:],'g-') # plot new prediciton of velocity y

    ax_list[4].plot(x[5,0:k+2],'b-')                         # plot new velocity z
    ax_list[4].plot(range(k+1,k+model.N), pred_x[5,1:],'g-') # plot new prediciton of velocity z

    # ax_list[3].plot(np.rad2deg(x[4, 0:k+2]),'b-')            # plot new steering angle
    # ax_list[3].plot(range(k+1,k+model.N), \
    #     np.rad2deg(pred_x[4,1:]),'g-')                       # plot new prediction of steering angle
    ax_list[5].step(range(0, k+1), u[0, 0:k+1],'b-')         # plot new acceleration force x
    ax_list[5].step(range(k, k+model.N), pred_u[0,:],'g-')   # plot new prediction of acceleration force x
    ax_list[6].step(range(0, k+1), u[1, 0:k+1],'b-')         # plot new acceleration force y 
    ax_list[6].step(range(k, k+model.N), pred_u[1,:],'g-')   # plot new prediction of acceleration force y 
    ax_list[7].step(range(0, k+1), u[2, 0:k+1],'b-')         # plot new acceleration force y 
    ax_list[7].step(range(k, k+model.N), pred_u[2,:],'g-')   # plot new prediction of acceleration force y 
    # ax_list[5].step(range(0, k+1), \
    #     np.rad2deg(u[1, 0:k+1]),'b-')                        # plot new steering rate
    # ax_list[5].step(range(k, k+model.N), \
    #     np.rad2deg(pred_u[1,:]),'g-')                        # plot new prediction of steering rate
    fig = plt.figure('2')
    ax_list = fig.axes
    
    # Delete old data in plot
    ax_list[0].get_lines().pop(-1).remove() # remove old prediction of trajectory xy
    ax_list[0].get_lines().pop(-1).remove() # remove old trajectory xy 

    ax_list[0].plot3D(x[0,0:k+2],x[1,0:k+2],x[2,0:k+2], '-b')             # plot new trajectory
    ax_list[0].plot3D(pred_x[0,1:], pred_x[1,1:], pred_x[2,1:], 'g-')        # plot new prediction of trajectory

    plt.pause(0.05)


def createPlot(x,u,start_pred,sim_length,model,path_points,xinit):
    """Creates a plot and adds the initial data provided by the arguments"""
    # print(f" x: {x}")
    # print(f" u: {u}")
    # print(f" start_pred: {start_pred}")
    # print(f" path_points: {path_points}")
     # Create empty plot
    fig = plt.figure('1')
    plt.clf()
    gs = GridSpec(6,3,figure=fig)
    
    # Plot trajectory xy
    axy_pos = fig.add_subplot(gs[:,0])
    l0, = axy_pos.plot(np.transpose(path_points[0,:]), np.transpose(path_points[1,:]), 'rx')
    l1, = axy_pos.plot(xinit[0], xinit[1], 'bx')
    plt.title('Position xy')
    #plt.axis('equal')
    plt.xlim([-1.,1.])
    plt.ylim([-3.5, 2.5])
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    l2, = axy_pos.plot(x[0,0],x[1,0],'b-')
    l3, = axy_pos.plot(start_pred[3,:], start_pred[4,:],'g-')
    axy_pos.legend([l0,l1,l2,l3],['desired trajectory','init pos','robot trajectory',\
        'predicted robot traj.'],loc='lower right')


    # Plot trajectory xz
    axz_pos = fig.add_subplot(gs[:,1])
    l0, = axz_pos.plot(np.transpose(path_points[0,:]), np.transpose(path_points[2,:]), 'rx')
    l1, = axz_pos.plot(xinit[0], xinit[2], 'bx')
    plt.title('Position xz')
    #plt.axis('equal')
    plt.xlim([-1.,1.])
    plt.ylim([-1, 1])
    plt.xlabel('x-coordinate')
    plt.ylabel('z-coordinate')
    l2, = axz_pos.plot(x[0,0],x[2,0],'b-')
    l3, = axz_pos.plot(start_pred[3,:], start_pred[5,:],'g-')
    axz_pos.legend([l0,l1,l2,l3],['desired trajectory','init pos','robot trajectory',\
        'predicted robot traj.'],loc='lower right')
    
    # Plot velocity
    ax_velx = fig.add_subplot(6,3,3)
    plt.grid("both")
    plt.title('Velocity X')
    plt.xlim([0., sim_length-1])
    plt.plot([0, sim_length-1], np.transpose([model.ub[6], model.ub[6]]), 'r:')
    plt.plot([0, sim_length-1], np.transpose([model.lb[6], model.lb[6]]), 'r:')
    ax_velx.plot(0.,x[3,0], '-b')
    ax_velx.plot(start_pred[6,:], 'g-')
    
    # Plot velocity Y
    ax_vely = fig.add_subplot(6,3,6)
    plt.grid("both")
    plt.title('Velocity Y')
    plt.xlim([0., sim_length-1])
    plt.plot([0, sim_length-1], np.transpose([model.ub[7], model.ub[7]]), 'r:')
    plt.plot([0, sim_length-1], np.transpose([model.lb[7], model.lb[7]]), 'r:')
    ax_vely.plot(0.,x[4,0], 'b-')
    ax_vely.plot(start_pred[7,:], 'g-')

    # Plot velocity Y
    ax_velz = fig.add_subplot(6,3,9)
    plt.grid("both")
    plt.title('Velocity Z')
    plt.xlim([0., sim_length-1])
    plt.plot([0, sim_length-1], np.transpose([model.ub[8], model.ub[8]]), 'r:')
    plt.plot([0, sim_length-1], np.transpose([model.lb[8], model.lb[8]]), 'r:')
    ax_velz.plot(0.,x[5,0], 'b-')
    ax_velz.plot(start_pred[8,:], 'g-')

    # # # Plot steering angle
    # ax_delta = fig.add_subplot(5,2,6)
    # plt.grid("both")
    # plt.title('Fx')
    # plt.xlim([0., sim_length-1])
    # plt.plot([0, sim_length-1], np.rad2deg(np.transpose([model.ub[6], model.ub[6]])), 'r:')
    # plt.plot([0, sim_length-1], np.rad2deg(np.transpose([model.lb[6], model.lb[6]])), 'r:')
    # ax_delta.plot(np.rad2deg(x[4,0]),'b-')
    # ax_delta.plot(np.rad2deg(start_pred[6,:]),'g-')

    # # Plot force x
    ax_Fx = fig.add_subplot(6,3,12)
    plt.grid("both")
    plt.title('Force x')
    plt.xlim([0., sim_length-1])
    plt.plot([0, sim_length-1], np.transpose([model.ub[0], model.ub[0]]), 'r:')
    plt.plot([0, sim_length-1], np.transpose([model.lb[0], model.lb[0]]), 'r:')
    ax_Fx.step(0, u[0,0], 'b-')
    ax_Fx.step(range(model.N), start_pred[0,:],'g-')

    # # Plot force y
    ax_Fy = fig.add_subplot(6,3,15)
    plt.grid("both")
    plt.title('Force y')
    plt.xlim([0., sim_length-1])
    plt.plot([0, sim_length-1], np.transpose([model.ub[1], model.ub[1]]), 'r:')
    plt.plot([0, sim_length-1], np.transpose([model.lb[1], model.lb[1]]), 'r:')
    ax_Fy.step(0, u[1,0], 'b-')
    ax_Fy.step(range(model.N), start_pred[1,:],'g-')

    # # plot force z
    ax_Fz = fig.add_subplot(6,3,18)
    plt.grid("both")
    plt.title('Force z')
    plt.xlim([0., sim_length-1])
    plt.plot([0, sim_length-1], np.transpose([model.ub[2], model.ub[2]]), 'r:')
    plt.plot([0, sim_length-1], np.transpose([model.lb[2], model.lb[2]]), 'r:')
    ax_Fz.step(0, u[2,0], 'b-')
    ax_Fz.step(range(model.N), start_pred[2,:],'g-')

    # plt.tight_layout()

    # # Make plot fullscreen. Comment out if platform dependent errors occur.
    mng = plt.get_current_fig_manager()
    # plt.pause(20)

    # TRYING NEW 3D PLOT........................................................................................................
    plt.figure('2')

    # syntax for 3-D projection
    ax3d = plt.axes(projection ='3d')
    
    # defining all 3 axes
    # z = np.linspace(0, 1, 100)
    # x = z * np.sin(25 * z)
    # y = z * np.cos(25 * z)
    
    # plotting
    ax3d.plot3D(np.transpose(path_points[0,:]), np.transpose(path_points[1,:]), np.transpose(path_points[2,:]), 'rx')
    ax3d.plot([xinit[0]], [xinit[1]], [xinit[2]], 'bx')
    ax3d.plot([x[0,0]],[x[1,0]],[x[2,0]],'b-')
    ax3d.plot(start_pred[3,:], start_pred[4,:], start_pred[5,:],'g-')
    ax3d.set_title('3D trajectory plot')
    # plt.show()
    # plt.pause(20)
    # ...........................................................................................................................


def main():
    # generate code for estimator
    model, solver = generate_pathplanner()

    # Simulation
    # ----------
    sim_length = 80 # simulate 8sec

    # Variables for storing simulation data
    x = np.zeros((model.neq,sim_length+1)) # states
    u = np.zeros((model.npar,sim_length)) # inputs

    # Set initial guess to start solver from
    x0i = np.zeros((model.nvar,1))
    x0 = np.transpose(np.tile(x0i, (1, model.N)))
    # Set initial condition
    xinit = np.transpose(np.array([0.8, 0., 0., 0., 0., 0.]))
    x[:,0] = xinit

    problem = {"x0": x0,
            "xinit": xinit}

    # Create 2D points on ellipse which the robot is supposed to follow
    num_points = 80
    path_points = calc_points_on_ellipse(num_points)

    start_pred = np.reshape(problem["x0"],(model.nvar,model.N)) # first prdicition corresponds to initial guess

    # generate plot with initial values
    createPlot(x,u,start_pred,sim_length,model,path_points,xinit)
   
    # Simulation
    for k in range(sim_length):
        
        # Set initial condition
        problem["xinit"] = x[:,k]

        # Set runtime parameters (here, the next N points on the path)
        next_path_points = extract_next_path_points(path_points, x[0:model.npar,k], model.N)
        problem["all_parameters"] = np.reshape(np.transpose(next_path_points), \
            (model.npar*model.N,1))

        # Time to solve the NLP!
        output, exitflag, info = solver.solve(problem)

        # Make sure the solver has exited properly.
        assert exitflag == 1, "bad exitflag"
        sys.stderr.write("FORCESPRO took {} iterations and {} seconds to solve the problem.\n"\
            .format(info.it, info.solvetime))

        # Extract output
        temp = np.zeros((np.max(model.nvar), model.N))
        for i in range(0, model.N):
            temp[:, i] = output['x{0:02d}'.format(i+1)]
        pred_u = temp[0:model.npar, :]
        pred_x = temp[model.npar:model.nvar, :]

        # Apply optimized input u of first stage to system and save simulation data
        u[:,k] = pred_u[:,0]
        x[:,k+1] = np.transpose(model.eq(np.concatenate((u[:,k],x[:,k]))))

        # plot results of current simulation step
        updatePlots(x,u,pred_x,pred_u,model,k)
       
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


if __name__ == "__main__":
    main()