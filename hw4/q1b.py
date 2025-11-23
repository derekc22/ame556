import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
from cvxopt import matrix, solvers
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # disable runtime errors from cvxopt
os.environ["OMP_NUM_THREADS"] = "1"           # disable runtime errors from cvxopt
solvers.options['show_progress'] = False      # disable printing from cvxopt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.utils import *


def plot(t, data):
    plot_dir = "hw4/plots"
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(11, 9))

    plt.subplot(3,1,1)
    plt.plot(t, data[:, 0], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('x(t) [m]')
    plt.grid()
    
    plt.subplot(3,1,2)
    plt.plot(t,  data[:, 1], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('θ(t) [rad]')
    plt.grid()
    
    plt.subplot(3,1,3)
    plt.plot(t, data[:, 2], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('u(t) [Nm]')    
    plt.grid()
    
    plt.suptitle("x(t), θ(t), u(t) for cartpole under qp control")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/q1b.pdf")
    plt.close()
    

def lqr():
    # Define system matrices
    M = 1
    m = 0.2
    L = 0.3
    g = 9.81
    
    # Uses the state definition x = [x θ ẋ θ̇] 
    # See hw4/q1a.py for more
    A = np.array([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, (m*g)/M, 0, 0],
        [0, ((M+m)*g)/(L*M), 0, 0]
    ])
    B = np.array([
        [0],
        [0],
        [1/M],
        [1/(L*M)]
    ])

    # Define cost matrices 
    # i) Q = diag(10, 10, 10, 10); R = 1
    Q = np.diag([10, 10, 10, 10])
    R = np.array([[1]])
    
    # Solve for P
    P = solve_continuous_are(A, B, Q, R)
    
    # Form K_lqr    
    return np.linalg.inv(R) @ B.T @ P


def qp(u_lqr):
    # cvxopt solves:
    # x* = min_x (1/2) xTPx + qTx
    # s.t. Ax = b    (equality constraints)
    # s.t. Gx ≤ h    (inequality constraints)
    
    # The problem statement is:
    # u* = min_u (u - u_lqr)^2
    # s.t. |u| ≤ 10
    
    # Expand the optimization:
    # u* = min_u (u - u_lqr)^2
    #    = min_u u^2 - 2u*u_lqr + u_lqr^2
    #    = min_u u^2 - (2u_lqr)u
    # u_lqr^2 can be discarded because constant terms in the QP formulation do not affect the solution
    
    # Rewrite the inequality constraint:
    # s.t. |u| ≤ 10
    # s.t. -10 ≤ u ≤ 10
    
    # In matrix form...
    # [ 1] u ≤ [10]
    # [-1]     [10]
    # row 1 gives:      u ≤ 10
    # row 2 gives:     -u ≤ 10 (or equivalently:  -10 ≤ u)

    # Thus:
    # P = [2] 
    # q = [-2u_lqr]
    # G = [ 1]
    #     [-1]
    # h = [10]
    #     [10]
        
    # Generically
    # Let n: # decision variables, i: # of equality constraints, j: # of inequality constraints
    # P ∈ Rnxn
    # q ∈ Rnx1    
    # A ∈ Rixn
    # b ∈ Rix1
    # G ∈ Rjxn
    # h ∈ Rjx1
    
    # In this problem, specifically n = # control inputs = 1

    P = matrix([2], (1, 1), tc="d")                 # 1x1, double
    q = matrix([-2*u_lqr.item()], (1, 1), tc="d")   # 1x1, double

    G = matrix([1, -1], (2, 1), tc="d")             # 2x1, double
    h = matrix([10, 10], (2, 1), tc="d")            # 2x1, double

    sol = solvers.qp(P, q, G, h)
    u_star = np.array([sol['x'][0]])                # 1x1, double
    
    return u_star


def get_q(d):
    return np.concatenate([
                    d.qpos,  
                    d.qvel
                    ])


def q1b():
    m, d = load_model("hw4/assets/cartpole_q1b.xml")
    reset(m, d, "up")
    viewer = mujoco.viewer.launch_passive(m, d)
    camera_presets = {
                   "lookat": [0.0, 0.0, 0.1], 
                   "distance": 3, 
                   "azimuth": 90, 
                   "elevation": 0
                }
    set_cam(viewer, track=False, presets=camera_presets, show_world_csys=False, show_body_csys=False)

    tmax = 2
    dt = m.opt.timestep
    ts = round(tmax/dt)
    data = np.zeros((ts, 3))
    time = np.arange(0, ts*dt, dt)
    
    K_lqr = lqr()

    for t in range(ts):

        q = get_q(d)

        u_lqr = -K_lqr @ q
        u_qp = qp(u_lqr)

        d.ctrl = u_qp

        data[t] = np.concatenate([q[0:2], u_qp], axis=0)
        
        mujoco.mj_step(m, d)
        viewer.sync()
        
    viewer.close()
    plot(time, data)



if __name__ == "__main__":
    q1b()