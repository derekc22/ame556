import mujoco
import numpy as np
np.set_printoptions(suppress=True)
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
from cvxopt import matrix, solvers
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
solvers.options['show_progress'] = False

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
    
    M = get_body_mass(m, "torso") + get_body_mass(m, "l_thigh") + get_body_mass(m, "l_calf") + get_body_mass(m, "r_thigh") + get_body_mass(m, "r_calf") 
    Izz = get_body_inertia(m, "torso")[1]
    g = np.abs(get_gravity(m))
    
    nq = m.nq
    nv = 6

    q_pos = x0[:nq]
    
    xc = q_pos[0]
    yc = q_pos[2]
    
    PF1, PF2 = get_feet_xpos(m, d)[:, :]
    PF1x, PF1y = PF1[[0, 2]]
    PF2x, PF2y = PF2[[0, 2]]
    
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

    # SYSTEMATIC TUNING APPROACH FOR MPC
    # 
    # The key insight: This is an INVERTED PENDULUM problem
    # - Theta control is MOST critical (unstable mode)
    # - X position couples strongly with theta (move feet to balance)
    # - Y height should be maintained but is less critical
    # 
    # TUNING STEPS:
    # 1. Start with diagonal Q, high weight on theta (position 2)
    # 2. Add x-theta coupling Q[0,2] and Q[2,0] (typically 0.01 to 0.1 of theta weight)
    # 3. Tune velocity weights (usually 0.1x to 1x of position weights)
    # 4. Adjust R for smoother control (0.01 to 1.0)
    # 5. Set H = Q or use LQR solution if available
    
    # State: [x, y, theta, x_dot, y_dot, theta_dot]
    
    # OPTION 1: Conservative baseline (start here)
    Q = np.array([
        [100,      0,      1,      0,      0,      0],   # x position
        [0,       10,      0,      0,      0,      0],   # y position
        [1,        0,   1000,      0,      0,      0],   # theta - CRITICAL
        [0,        0,      0,     10,      0,      0],   # x_dot
        [0,        0,      0,      0,     10,      0],   # y_dot
        [0,        0,      0,      0,      0,    100],   # theta_dot
    ])
    R = 0.1 * np.eye(4)
    H = 2 * Q  # Terminal cost
    
    # OPTION 2: More aggressive (if Option 1 is too slow)
    # Q = np.array([
    #     [1000,     0,     10,      0,      0,      0],
    #     [0,      100,      0,      0,      0,      0],
    #     [10,       0,  10000,      0,      0,      0],
    #     [0,        0,      0,    100,      0,      0],
    #     [0,        0,      0,      0,    100,      0],
    #     [0,        0,      0,      0,      0,   1000],
    # ])
    # R = 0.01 * np.eye(4)
    # H = 5 * Q
    
    # OPTION 3: Very aggressive (if system needs fast response)
    # Q = np.array([
    #     [5000,     0,     50,      0,      0,      0],
    #     [0,      500,      0,      0,      0,      0],
    #     [50,       0, 100000,      0,      0,      0],
    #     [0,        0,      0,    500,      0,      0],
    #     [0,        0,      0,      0,   1000,      0],
    #     [0,        0,      0,      0,      0,   5000],
    # ])
    # R = 0.05 * np.eye(4)
    # H = np.array([
    #     [ 1.2e+04,  0.0e+00, -5.0e+01,  1.5e+03,  0.0e+00, -1.2e+01],
    #     [ 0.0e+00,  2.5e+04,  0.0e+00,  0.0e+00,  1.5e+03,  0.0e+00],
    #     [-5.0e+01,  0.0e+00,  7.0e+04, -2.5e+01,  0.0e+00,  2.5e+03],
    #     [ 1.5e+03,  0.0e+00, -2.5e+01,  7.0e+02,  0.0e+00, -6.0e+00],
    #     [ 0.0e+00,  1.5e+03,  0.0e+00,  0.0e+00,  1.5e+02,  0.0e+00],
    #     [-1.2e+01,  0.0e+00,  2.5e+03, -6.0e+00,  0.0e+00,  1.0e+03]
    # ])

    # Setup optimization problem
    N = 10
    nx = A.shape[0]
    nm = B.shape[1]
    
    # Discretize continuous-time dynamics
    Bbar = np.hstack([B, w])
    Ak, Bbark = discretize(A, Bbar, dt)
    Bk = Bbark[:, :4]
    wk = Bbark[:, 4]

    x0 = np.array(x0).reshape(-1, 1)
    Xf = np.tile(xf, reps=N).reshape(-1, 1)
    W = np.tile(wk, reps=N).reshape(-1, 1)

    # Build prediction matrices
    Phi = np.zeros(shape=(N*nx, nx))
    Gamma = np.zeros(shape=(N*nx, N*nm))
    Omega = np.zeros(shape=(N*nx, N*nx))
    
    for i in range(N):
        Phi[i*nx:(i+1)*nx, :] = np.linalg.matrix_power(Ak, i+1)
        for j in range(i+1):
            Gamma[i*nx:(i+1)*nx, j*nm:(j+1)*nm] = np.linalg.matrix_power(Ak, i-j) @ Bk
            Omega[i*nx:(i+1)*nx, j*nx:(j+1)*nx] = np.linalg.matrix_power(Ak, i-j)
            
    # Build block diagonal cost matrices
    Qbar = np.kron(np.eye(N), Q)
    Qbar[(N-1) * nv:, (N-1) * nv:] = H
    Rbar = np.kron(np.eye(N), R)
    
    # Contact constraints
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

    P_cvx = matrix(P.astype(float))
    q_cvx = matrix(q.astype(float))
    G_cvx = matrix(G.astype(float))
    h_cvx = matrix(h.astype(float))
    
    sol = solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
    F_GRF_star = np.array([sol['x']])[0]
    
    F_feet = -F_GRF_star
    
    F_foot_l = np.vstack([ F_feet[0], 0, F_feet[1] ])
    F_foot_r = np.vstack([ F_feet[2], 0, F_feet[3] ])

    jacp_l = np.zeros((3, m.nv))
    mujoco.mj_jacGeom(m, d, jacp_l, None, get_geom_id(m, "l_foot"))

    jacp_r = np.zeros((3, m.nv))
    mujoco.mj_jacGeom(m, d, jacp_r, None, get_geom_id(m, "r_foot"))
    
    tau_l_full = jacp_l.T @ F_foot_l
    tau_r_full = jacp_r.T @ F_foot_r
    
    tau_l = tau_l_full[-4:]
    tau_r = tau_r_full[-4:]
    
    tau = (tau_l + tau_r).flatten()
    
    return tau


def get_q(d):
    return np.concatenate([d.qpos, d.qvel])


def q2b():
    
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
    
    for t in range(ts):
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
        
        x0 = np.array([xc, yc, theta_c, xc_dot, yc_dot, theta_c_dot])
        print(x0)

        u_mpc = mpc(x0, xf, m, d, dt=0.04)
        d.ctrl = u_mpc

        data[t] = np.concatenate([x0[:3], u_mpc], axis=0)
        
        mujoco.mj_step(m, d)
        viewer.sync()
        
    viewer.close()
    plot(time, data)


if __name__ == "__main__":
    q2b()