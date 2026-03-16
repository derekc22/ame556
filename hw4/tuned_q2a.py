import mujoco
import numpy as np
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
    
    plt.suptitle("x(t), θ(t), τ(t) for balancing biped under qp control")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/q2a.pdf")
    plt.close()
    

def qp(x, m, d):
    
    if not get_feet_contact(m, d): return np.zeros(m.nu)
    
    M = get_body_mass(m, "torso") + get_body_mass(m, "l_thigh") + get_body_mass(m, "l_calf") + get_body_mass(m, "r_thigh") + get_body_mass(m, "r_calf") 
    Izz = get_body_inertia(m, "torso")[1]
    g = np.abs(get_gravity(m))
    
    # TUNING GUIDE:
    # - alpha: Lower = more aggressive control (0.0001-0.01)
    # - Q[0,0]: x position tracking weight (0.1-10)
    # - Q[1,1]: y position tracking weight (0.1-10)
    # - Q[2,2]: theta tracking weight (1-100)
    # - Q[0,2] and Q[2,0]: x-theta coupling (-1 to 1)
    
    alpha = 0.001
    Q = np.array([
        [1,    0,  0.1],
        [0,    1,    0],
        [0.1,  0,    5],
    ])

    q_pos = x[:m.nq]
    q_vel = x[m.nq:]
    
    xc = q_pos[0]
    yc = q_pos[2]
    theta_c_quat = q_pos[3:7]
    theta_c = R.from_quat(theta_c_quat, scalar_first=True).as_euler('zyx')[1]
    
    xc_dot = q_vel[0]
    yc_dot = q_vel[2]
    theta_c_dot = q_vel[4]
    
    xc_des = 0
    yc_des = 0.5
    theta_c_des = 0 
    xc_dot_des = 0
    yc_dot_des = 0
    theta_c_dot_des = 0
    
    # TUNING GUIDE FOR PD GAINS:
    # Start with low gains and increase gradually
    # If oscillating: reduce Kp or increase Kd
    # If too slow: increase Kp
    # If overshooting: increase Kd
    # Critical damping ratio: Kd = 2*sqrt(Kp*M) or Kd = 2*sqrt(Kp*Izz)
    
    # For reference, your original gains:
    # Kp_x = 5, Kd_x = 10
    # Kp_y = 5, Kd_y = 10  
    # Kp_theta = 0.1, Kd_theta = 0.1
    
    Kp_x = 5
    Kd_x = 10
    Kp_y = 5
    Kd_y = 10
    Kp_theta = 0.5      # Try: 0.2, 0.5, 1.0, 2.0, 5.0
    Kd_theta = 0.5      # Should roughly match Kp_theta

    x_ddot_des = Kp_x * (xc_des - xc) + Kd_x * (xc_dot_des - xc_dot)
    y_ddot_des = Kp_y * (yc_des - yc) + Kd_y * (yc_dot_des - yc_dot)
    theta_ddot_des = Kp_theta * (theta_c_des - theta_c) + Kd_theta * (theta_c_dot_des - theta_c_dot)
    
    PF1, PF2 = get_feet_xpos(m, d)[:, :]
    PF1x, PF1y = PF1[[0, 2]]
    PF2x, PF2y = PF2[[0, 2]]
    
    A = np.array([
        [1,       0,       1,       0      ],
        [0,       1,       0,       1      ],
        [yc-PF1y, PF1x-xc, yc-PF2y, PF2x-xc]
    ])
    
    b = np.array([
        [M * x_ddot_des],
        [M * (y_ddot_des + g)],
        [Izz * theta_ddot_des],
    ])
    
    n = A.shape[1]
    
    Fy_max = 250
    Fy_min = 10
    mu = 0.7

    P = 2 * (A.T @ Q @ A + alpha * np.eye(n))
    q = -2 * A.T @ Q @ b
    G = np.array([
        [0,    1,   0,   0],
        [0,    0,   0,   1],
        [0,   -1,   0,   0],
        [0,    0,   0,  -1],
        [1,  -mu,   0,   0],
        [0,    0,   1, -mu],
        [-1, -mu,   0,   0],
        [0,    0,  -1, -mu]
    ])
    h = np.array([
        [Fy_max],
        [Fy_max],
        [-Fy_min],
        [-Fy_min],
        [0],
        [0],
        [0],
        [0]
    ])
    
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


def q2a():
    
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

    tmax = 200
    dt = m.opt.timestep
    ts = round(tmax/dt)
    data = np.zeros((ts, 7))
    time = np.arange(0, ts*dt, dt)

    for t in range(ts):
        q = get_q(d)
        u_qp = qp(q, m, d)
        d.ctrl = u_qp

        xz = np.r_[q[0], q[2]]
        theta = R.from_quat(q[3:7], scalar_first=True).as_euler('zyx')[1:2]
        data[t] = np.concatenate([xz, theta, u_qp], axis=0)
        
        mujoco.mj_step(m, d)
        viewer.sync()
        
    viewer.close()
    plot(time, data)


if __name__ == "__main__":
    q2a()