import mujoco
import numpy as np
np.set_printoptions(suppress=True)
from scipy.spatial.transform import Rotation as R
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
    
    plt.subplot(4,2,1)
    plt.plot(t, data[:, 0], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('x(t) [m]')
    plt.grid()

    plt.subplot(4,2,2)
    plt.plot(t, data[:, 1], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('y(t) [m]')    
    plt.grid()

    plt.subplot(4,2,3)
    plt.plot(t,  data[:, 2], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('θ(t) [rad]')
    plt.grid()
    
    plt.subplot(4,2,4)
    plt.plot(t,  data[:, 3], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('τ1(t) [Nm]')
    plt.grid()
    
    plt.subplot(4,2,5)
    plt.plot(t,  data[:, 4], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('τ2(t) [Nm]')
    plt.grid()
    
    plt.subplot(4,2,6)
    plt.plot(t,  data[:, 5], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('τ3(t) [Nm]')
    plt.grid()
    
    plt.subplot(4,2,7)
    plt.plot(t,  data[:, 6], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('τ4(t) [Nm]')
    plt.grid()
    
    plt.suptitle("x(t), θ(t), τ(t) for balancing biped under mpc control")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/q2b.pdf")
    plt.close()
    

def mpc(x0, xf, m, d, dt):
    
    if not get_feet_contact(m, d): return np.zeros(m.nu)
    
    # Define system matrices
    # M = get_body_mass(m, "torso") + get_body_mass(m, "l_thigh") + get_body_mass(m, "l_calf") + get_body_mass(m, "r_thigh") + get_body_mass(m, "r_calf") 
    M = get_M(m)
    # Izz = get_body_inertia(m, "torso")[1]
    Izz = get_Izz_com(m, d)
    g = np.abs(get_gravity(m))
    
    nq = m.nq
    
    # State dimension
    nv = 6

    q_pos = x0[:nq]
    
    xc = q_pos[0]
    yc = q_pos[2]
    
    PF1, PF2 = get_feet_xpos(m, d)[:, :]
    PF1x, PF1y = PF1[[0, 2]]
    PF2x, PF2y = PF2[[0, 2]]
    
    # Uses the state definition x = [x y θ ẋ ẏ θ̇] 
    # Uses the definition F = [F1x F1y F2x F2y] 
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    B = np.array([
        [0,                                 0,                 0,                 0],
        [0,                                 0,                 0,                 0],
        [0,                                 0,                 0,                 0],
        [1/M,                               0,               1/M,                 0],
        [0,                               1/M,                 0,               1/M],
        [(1/Izz)*(yc-PF1y), (1/Izz)*(PF1x-xc), (1/Izz)*(yc-PF2y), (1/Izz)*(PF2x-xc)]
    ])    
    w = np.array([
        [0],
        [0],
        [0],
        [0],
        [-g],
        [0]
    ])

    Q = np.array([
            [1000,   0,     0,    0,    0,    0],
            [0,    5000,    0,    0,    0,    0], 
            [0,      0,  8000,    0,    0,    0], 
            [0,      0,     0,   100,   0,    0],
            [0,      0,     0,    0,   100,    0],
            [0,      0,     0,    0,    0,  1000],
        ])

    # R Matrix: 
    # 0.01 provides enough regularization to prevent chattering, 
    # but is cheap enough to allow full force usage.
    R = np.eye(4) * 0.01

    # H Matrix (Terminal Cost):
    # RE-CALCULATED using Discrete Algebraic Riccati Equation (DARE)
    # specifically for: Izz=0.19, A[5,2]=20, dt=0.04.
    # This ensures the 10-step horizon is stable.
    H = np.array([
        [ 1.245e+04, -9.400e-12, -1.582e+02,  1.603e+03,  1.692e-13, -4.583e+01],
        [-9.400e-12,  2.990e+04, -1.972e-11, -1.289e-12,  1.728e+03, -7.574e-13],
        [-1.582e+02, -1.972e-11,  8.284e+04, -9.215e+01,  8.625e-13,  2.998e+03],
        [ 1.603e+03, -1.289e-12, -9.215e+01,  7.344e+02,  1.094e-14, -2.337e+01],
        [ 1.692e-13,  1.728e+03,  8.625e-13,  1.094e-14,  3.443e+02,  4.448e-14],
        [-4.583e+01, -7.574e-13,  2.998e+03, -2.337e+01,  4.448e-14,  1.121e+03]
    ])

    # Setup optimization problem
    N = 10
    nx = A.shape[0]
    nm = B.shape[1]
    
    # Discretize continuous-time dynamics
    Bbar = np.hstack([B, w])
    Ak, Bbark = discretize(A, Bbar, dt)
    Bk = Bbark[:, :4]
    wk = Bbark[:, 4]

    # Reshape x0
    x0 = np.array(x0).reshape(-1, 1)
    
    # Repeat final state over N-step horizon
    Xf = np.tile(xf, reps=N).reshape(-1, 1)

    # Repeat gravity over N-step horizon
    W = np.tile(wk, reps=N).reshape(-1, 1)

    # Build state-transition prediction matrix (Phi) and control-input prediction matrix (Gamma)
    Phi = np.zeros(shape=(N*nx, nx))
    Gamma = np.zeros(shape=(N*nx, N*nm))
    Omega = np.zeros(shape=(N*nx, N*nx))
    
    for i in range(N):
        Phi[i*nx:(i+1)*nx, :] = np.linalg.matrix_power(Ak, i+1)
        for j in range(i+1):
            Gamma[i*nx:(i+1)*nx, j*nm:(j+1)*nm] = np.linalg.matrix_power(Ak, i-j) @ Bk
            Omega[i*nx:(i+1)*nx, j*nx:(j+1)*nx] = np.linalg.matrix_power(Ak, i-j)
            
    # Build block diagonal cost matrices over N-step horizon
    Qbar = np.kron(np.eye(N), Q)
    Qbar[(N-1) * nv:, (N-1) * nv:] = H # Assign terminal state cost
    Rbar = np.kron(np.eye(N), R)
    
    PF1, PF2 = get_feet_xpos(m, d)[:, :]
    PF1x, PF1y = PF1[[0, 2]]
    PF2x, PF2y = PF2[[0, 2]]
            
    Fy_max = 250
    Fy_min = 10
    mu = 0.7

    P = 2 * (Gamma.T @ Qbar @ Gamma + Rbar)
    q = 2 * Gamma.T @ Qbar @ (Phi @ x0 + Omega @ W - Xf)
    
    Gu = np.array([
        [0,    1,   0,   0],
        [0,    0,   0,   1],
        [0,   -1,   0,   0],
        [0,    0,   0,  -1],
        [1,  -mu,   0,   0],
        [0,    0,   1, -mu],
        [-1, -mu,   0,   0],
        [0,    0,  -1, -mu]
    ])
    hu = np.array([
        [Fy_max],
        [Fy_max],
        [-Fy_min],
        [-Fy_min],
        [0],
        [0],
        [0],
        [0]
    ])
    
    G = np.kron(np.eye(N), Gu)
    h = np.tile(hu, (N, 1))

    # Convert numpy matrices to cvxopt matrices
    P_cvx = matrix(P.astype(float))
    q_cvx = matrix(q.astype(float))
    G_cvx = matrix(G.astype(float))
    h_cvx = matrix(h.astype(float))
    
    # Solve
    sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    F_GRF_star = np.array([sol['x']])[0]
    
    F_feet = -F_GRF_star
    
    F_foot_l = np.vstack([ F_feet[0], 0, F_feet[1] ])
    F_foot_r = np.vstack([ F_feet[2], 0, F_feet[3] ])

    jacp_l = np.zeros((3, m.nv)) # translational Jacobian
    mujoco.mj_jacGeom(m, d, jacp_l, None, get_geom_id(m, "l_foot"))

    jacp_r = np.zeros((3, m.nv)) # translational Jacobian
    mujoco.mj_jacGeom(m, d, jacp_r, None, get_geom_id(m, "r_foot"))
    
    
    tau_l_full = jacp_l.T @ F_foot_l
    tau_r_full = jacp_r.T @ F_foot_r
    
    tau_l = tau_l_full[-4:]
    tau_r = tau_r_full[-4:]
    
    tau = (tau_l + tau_r).flatten()
    
    return tau


def get_q(d):
    return np.concatenate([
                    d.qpos,  
                    d.qvel
                    ])


def q2b():
    
    # Initial conditions: 
    # x  = 0
    # q1 = -pi/3
    # q2 =  pi/2
    # q3 = -pi/6
    # q4 =  pi/2
    
    # Set y such that "feet are slightly above the ground"
    # Note that the legs of the biped are bent with the above IC..
    # So the biped's COM z-position can safely be initialized as less than the height from its COM to its feet
    # y = 0.52
    
    m, d = load_model("hw4/assets/biped.xml")
    reset(m, d, "init")
    viewer = mujoco.viewer.launch_passive(m, d)
    camera_presets = {
                   "lookat": [0.0, 0.0, 0.55], 
                   "distance": 2, 
                   "azimuth": 90, 
                   "elevation": -10
                }  
    set_cam(viewer, track=False, presets=camera_presets, show_world_csys=False, show_body_csys=False)

    tmax = 2
    dt = m.opt.timestep
    ts = round(tmax/dt)
    data = np.zeros((ts, 7))
    time = np.arange(0, ts*dt, dt)
    
    xf = [0, 0.5, 0, 0, 0, 0]
    
    # for t in range(ts):
    while True:

        q = get_q(d)
        
        q_pos = q[:m.nq]
        q_vel = q[m.nq:]
        
        xc = q_pos[0]
        yc = q_pos[2]
        theta_c_quat = q_pos[3:7]
        theta_c = R.from_quat(theta_c_quat, scalar_first=True).as_euler('zyx')[1]
        
        xc_dot = q_vel[0]
        yc_dot = q_vel[2]
        theta_c_dot = q_vel[4]

        x_com = get_x_com(m, d)
        xc, yc = x_com[[0, 2]]     
        
        v_com = get_v_com(m, d)
        xc_dot, yc_dot = v_com[[0, 2]]     
        
        x0 = np.array([xc, yc, theta_c, xc_dot, yc_dot, theta_c_dot])

        u_mpc = mpc(x0, xf, m, d, dt=0.04)
        d.ctrl = u_mpc

        # data[t] = np.concatenate([x0[:3], u_mpc], axis=0)
        mujoco.mj_step(m, d)
        viewer.sync()
        
    viewer.close()
    # plot(time, data)



if __name__ == "__main__":
    q2b()