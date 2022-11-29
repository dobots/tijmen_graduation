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
from matplotlib.gridspec import GridSpec
import casadi
from mpl_toolkits import mplot3d
from matplotlib.patches import Rectangle
from function_file_raybot_v7 import *


def generate_pathplanner(create_new_solver):
    """Generates and returns a FORCESPRO solver that calculates a path based on
    constraints and dynamics while minimizing an objective function
    """
    # Model Definition
    # ----------------

    # Problem dimensions
    model = forcespro.nlp.SymbolicModel()
    model.N = 20 # horizon length
    model.nvar = 20  # number of variables
    model.neq = 12  # number of equality constraints
    model.npar = 6 # number of runtime parameters

    # Objective function
    model.objective = obj
    model.objectiveN = objN # increased costs for the last stage
    # The function must be able to handle symbolic evaluation,
    # by passing in CasADi symbols. This means certain numpy funcions are not
    # available.

    # We use an explicit RK4 integrator here to discretize continuous dynamics
    integrator_stepsize = 0.5 ## decrease the stepsize!
    model.eq = lambda z: forcespro.nlp.integrate(continuous_dynamics, z[(model.nvar-model.neq):model.nvar], z[0:(model.nvar-model.neq)],
                                                integrator=forcespro.nlp.integrators.IRK4,
                                                stepsize=integrator_stepsize)
    # Indices on LHS of dynamical constraint - for efficiency reasons, make
    # sure the matrix E has structure [0 I] where I is the identity matrix.
    model.E = np.concatenate([np.zeros((model.neq,(model.nvar-model.neq))), np.eye(model.neq)], axis=1)

    # Inequality constraints
    #  upper/lower variable bounds lb <= z <= ub
    #                     inputs                 |  states
    #                     Fx    Fy    Fz     x      y    z    Vx    Vy    Vz  
    # model.lb = np.array([-100.,  -100.,  -100.,   -20.,  -20.,  -20.,  -40., -40., -40.])
    # model.ub = np.array([+100.,  +100.,   100.,    20.,   20.,   20.,   40.,  40.,  40.])
    # model.lb = np.array([-100.,  -100.,  -100., -100.,  -100., -100., -100.,  -100.,  -20.,  -20.,  -20., -20.,  -20.,  -20.,  -10., -10., -10., -10., -10., -10.])
    # model.ub = np.array([+100.,  +100.,   100., +100.,  +100.,  100., +100.,   100.,   20.,   20.,   20.,   20.,   20.,   20.,   10.,  10.,  10.,   10.,  10.,  10.])
    # model.lb = np.array([-np.inf,  -np.inf,  -np.inf, -np.inf,  -np.inf, -np.inf, -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    # model.ub = np.array([np.inf,  np.inf,   np.inf, np.inf,  np.inf,  np.inf, np.inf,   np.inf,    np.inf,   np.inf,   np.inf,   np.inf,   np.inf,   np.inf,   np.inf, np.inf,  np.inf,  np.inf, np.inf,  np.inf])
    model.lb = np.array([-28.,  -28.,  -28., -28.,  -28., -28., -28.,  -28.,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
    model.ub = np.array([+36.,  +36.,   36., +36.,  +36.,  36., +36.,   36.,    np.inf,   np.inf,   np.inf,   np.inf,   np.inf,   np.inf,   np.inf, np.inf,  np.inf,  np.inf, np.inf,  np.inf])
    # model.lb = np.array([-2000,  -2000,  -2000, -2000,  -2000, -2000, -2000,  -2000,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]) #for rexrov
    # model.ub = np.array([+2000,  +2000,   2000, +2000,  +2000,  2000, +2000,  2000,   np.inf,   np.inf,   np.inf,   np.inf,   np.inf,   np.inf,   np.inf, np.inf,  np.inf,  np.inf,  np.inf,  np.inf, np.inf]) # for rexrov
    # model.lb = np.array([-1.,  -1.,  -1., -1.,  -1., -1., -1.,  -1.,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -np.inf,  -0.5, -0.5, -0.5, -0.5, -0.5, -0.5])
    # model.ub = np.array([+1.,  +1.,   1., +1.,  +1.,  1., +1.,   1.,   np.inf,   np.inf,   np.inf,   np.inf,   np.inf,   np.inf,   0.5,  0.5,  0.5,   0.5,  0.5,  0.5])
    # model.lb = np.array([-28.,  -28.,  -28., -28.,  -28., -28., -28.,  -28.,  -20.,  -20.,  -20., -20.,  -20.,  -20.,  -0.25, -0.25, -0.25, -0.25, -0.25, -0.25])
    # model.ub = np.array([+36.,  +36.,   36., +36.,  +36.,  36., +36.,   36.,   20.,   20.,   20.,   20.,   20.,   20.,   0.25,  0.25,  0.25,   0.25,  0.25,  0.25])

    # Initial condition on vehicle states x
    model.xinitidx = range((model.nvar-model.neq),model.nvar) # use this to specify on which variables initial conditions
    # are imposed

    # Solver generation
    # -----------------

    # Set solver options
    codeoptions = forcespro.CodeOptions('FORCESNLPsolver')
    codeoptions.maxit = 200 #200     # Maximum number of iterations
    codeoptions.printlevel = 2  # Use printlevel = 2 to print progress (but 
    #                             not for timings)
    codeoptions.optlevel = 2    # 0 no optimization, 1 optimize for size, 
    #                             2 optimize for speed, 3 optimize for size & speed
    codeoptions.cleanup = False
    codeoptions.timing = 1
    codeoptions.overwrite = 1
    codeoptions.nlp.hessian_approximation = 'bfgs' # 'bfgs'
    codeoptions.solvemethod = 'SQP_NLP' # choose the solver method Sequential 
    #                              Quadratic Programming 'SQP_NLP' 
    # codeoptions.nlp.bfgs_init = 2.5*np.identity(model.neq)
    codeoptions.nlp.bfgs_init = 2.5*np.identity(model.neq)
    # codeoptions.nlp.bfgs_init = np.diag(np.array([1.23774255e+00, 1.23774255e+00, 1.27340741e+00, 1.27340741e+00,
    #                                               6.88696480e-01, 6.88696480e-01, 1.84223499e+00, 1.84223499e+00,
    #                                               1.00000000e+00, 1.00000000e+00, 2.50725148e+03, 1.00000000e+00,
    #                                               4.51614177e+01, 5.18369858e+01, 6.33790225e+03, 5.57089023e+01,
    #                                               2.13425158e+04, 1.00000000e+00, 2.47872333e+00, 1.00000000e+00]))
    codeoptions.exportBFGS = 1
    # codeoptions.nlp.parametricBFGSinit = 1  # Allows us to initialize the estimate at run time with the exported one
    # codeoptions.nlp.integrator.nodes = 5
    codeoptions.sqp_nlp.maxqps = 3   # maximum number of quadratic problems to be solved
    codeoptions.sqp_nlp.reg_hessian = 5e-3 # increase this if exitflag=-8
    # codeoptions.sqp_nlp.reg_hessian = 50 # increase this if exitflag=-8
    # codeoptions.sqp_nlp.reg_hessian = 500000 # increase this if exitflag=-8
    # codeoptions.exportBFGS = 1
    # codeoptions.sqp_nlp.TolStat = 0.1
    # codeoptions.sqp_nlp.TolEq = 0.1
    # codeoptions.forcenonconvex = 1
    codeoptions.sqp_nlp.qpinit = 0
    codeoptions.nlp.integrator.reuseNewtonJacobian = 0
    # codeoptions.nlp.integrator.newtonIter = 20
    # change this to your server or leave uncommented for using the 
    # standard embotech server at https://forces.embotech.com 
    # codeoptions.server = 'https://forces.embotech.com'
    
    # Creates code for symbolic model formulation given above, then contacts 
    # server to generate new solver
    print(f"create_new_solver: {create_new_solver}")
    if create_new_solver:
        solver = model.generate_solver(options=codeoptions)
    else:
        solver = forcespro.nlp.Solver.from_directory("/home/tijmen/plankton_ws/src/tijmen_graduation/Plankton/uuv_control/uuv_mpc_control/scripts/FORCESNLPsolver")
    return model,solver
