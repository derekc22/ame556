import mujoco
import numpy as np
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
    
    plt.suptitle("x(t), θ(t), u(t) for cartpole under mpc")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/q1c.pdf")
    plt.close()


def mpc(x0, xf, dt):
    """Direct shooting mpc"""    

    # Define system matrices
    M = 1.0
    m = 0.2
    L = 0.3
    g = 9.81
    
    # State dimension
    nv = 4
    
    # Uses the state definition x = [x θ ẋ θ̇] 
    # See hw4/q1a.py for more
    A = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, (m*g)/M, 0, 0],
        [0.0, ((M+m)*g)/(L*M), 0.0, 0.0],
    ])
    B = np.array([
        [0.0],
        [0.0],
        [1.0/M],
        [1.0/(L*M)]
    ])
        
    # Define cost matrices
    Q = np.diag([100.0, 0.01, 100.0, 0.01])
    R = np.array([[1.0]])
    H = np.diag([200.0, 0.02, 200.0, 0.02]) # Terminal state cost

    # Setup optimization problem
    N = 20
    nx = A.shape[1]
    nm = B.shape[1]
    
    # Discretize continuous-time dynamics
    Ak, Bk = discretize(A, B, dt)

    # Reshape x0
    x0 = np.array(x0).reshape(-1, 1)
    
    # Repeat final state over N-step horizon
    Xf = np.tile(xf, reps=N).reshape(-1, 1)

    # Build state-transition prediction matrix (Phi) and control-input prediction matrix (Gamma)
    Phi = np.zeros(shape=(N*nx, nx))
    Gamma = np.zeros(shape=(N*nx, N*nm))
    
    for i in range(N):
        Phi[i*nx:(i+1)*nx, :] = np.linalg.matrix_power(Ak, i+1)
        for j in range(i+1):
            Gamma[i*nx:(i+1)*nx, j*nm:(j+1)*nm] = np.linalg.matrix_power(Ak, i-j) @ Bk
    

    # Build block diagonal cost matrices over N-step horizon
    Qbar = np.kron(np.eye(N), Q)
    Qbar[(N-1) * nv:, (N-1) * nv:] = H # Assign terminal state cost
    Rbar = np.kron(np.eye(N), R)
    
    # cvxopt solves: x* = min_x (1/2) xTPx + qTx
    # Build P and q
    P = 2 * (Gamma.T @ Qbar @ Gamma + Rbar)
    q = 2 * Gamma.T @ Qbar @ (Phi @ x0 - Xf)
    
    # Set max and min values for control inequality constraints
    u_max = 10.0
    u_min = -10.0
    
    # Set max and min values for state inequality constraints
    x_max = 0.8
    x_min = -0.8
    
    theta_max = np.pi/4
    theta_min = -np.pi/4
    
    # Tile constraints over horizon
    Umax = np.tile([u_max], reps=N).reshape(-1, 1)
    Umin = np.tile([u_min], reps=N).reshape(-1, 1)
    
    Xmax = np.tile([x_max, theta_max], reps=N).reshape(-1, 1)
    Xmin = np.tile([x_min, theta_min], reps=N).reshape(-1, 1)
    
    # Build control inequality constraint matrix (Gu) and inequality constraint vector (hu)
    Gu = np.vstack((
        np.eye(N*nm), 
        -np.eye(N*nm)
    ))
    hu = np.vstack((
        Umax, 
        -Umin
    ))

    # Build state inequality constraint matrix (Gx) and inequality constraint vector (hx)
    # Only [x θ] have constraints - build selection matrix Gx_select to select only these states when applying constraints
    Gx_select = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0]])
    
    # Build block diagonal cost matrices over N-step horizon    
    Gxbar = np.kron(np.eye(N), Gx_select)
    
    # Build state inequality constraint matrix (Gx) and inequality constraint vector (hx)
    # Note that the extra terms in each are due to the fact that this is a direct-shooting formulation, where u are the decision variables, NOT x
    # Thus, the N-step horizon state vector (X) must be written in terms of the N-step horizon control vector (U) as X = Φx0 ​+ ΓU
    # Rearranging then results in the expressions below for Gx and hx
    Gx = np.vstack((
        Gxbar @ Gamma, 
        -Gxbar @ Gamma
    ))
    hx = np.vstack((
        Xmax - Gxbar @ (Phi @ x0),
        -Xmin + Gxbar @ (Phi @ x0)
    ))

    # Building combined inequality constraint matrix and inequality constraint vector
    G = np.vstack((Gu, Gx))
    h = np.vstack((hu, hx))   
    
    # Convert numpy matrices to cvxopt matrices
    P_cvx = matrix(P.astype(float))
    q_cvx = matrix(q.astype(float))
    G_cvx = matrix(G.astype(float))
    h_cvx = matrix(h.astype(float))
    
    # Solve
    sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    u_star = np.array([sol['x'][0]])  
    
    return u_star
        

def get_q(d):
    return np.concatenate([
                    d.qpos,  
                    d.qvel
                    ])


def q1c():
    m, d = load_model("hw4/assets/cartpole_q1c.xml")
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
    
    xf = [0, 0, 0, 0]
    
    for t in range(ts):

        q = get_q(d)

        u_mpc = mpc(q, xf, dt)

        d.ctrl = u_mpc

        data[t] = np.concatenate([q[0:2], u_mpc], axis=0)
        
        mujoco.mj_step(m, d)
        viewer.sync()
        
    viewer.close()
    plot(time, data)



if __name__ == "__main__":
    q1c()