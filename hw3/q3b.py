import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.utils import *


def plot(t, data):
    plot_dir = "hw3/plots"
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(11, 9))

    # 0.1 is the initial value for both x and θ, so we can reuse the same Ts and Mp lines
    init = 0.1
    
    # settling time, 5% criterion
    # for a 0 steady-state reference, use |y(t)| <= 0.05|y0|
    Ts_top_line = 0.05*init*np.ones_like(t)
    Ts_bottom_line = -0.05*init*np.ones_like(t)
    
    # overshoot, 20% max
    # for a 0 steady-state reference, use |y(t)| <= 1.2|y0|
    Mp_top_line = 1.2*init*np.ones_like(t)
    Mp_bottom_line = -1.2*init*np.ones_like(t)

    plt.subplot(3,1,1)
    plt.plot(t, data[:, 0], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('x(t) [m]')
    plt.grid()
    
    # controller requirements
    plt.plot(t, Ts_top_line, "--", color="C1", label="Ts, 5%", linewidth=2)
    plt.plot(t, Ts_bottom_line, "--", color="C1", linewidth=2)
    plt.plot(t, Mp_top_line, "--", color="C2", label="Mp, 20%", linewidth=2)
    plt.plot(t, Mp_bottom_line, "--", color="C2", linewidth=2)
    plt.legend(loc="upper right")

    plt.subplot(3,1,2)
    plt.plot(t,  data[:, 1], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('θ(t) [rad]')
    plt.grid()
    
    # controller requirements
    plt.plot(t, Ts_top_line, "--", color="C1", label="Ts, 5%", linewidth=2)
    plt.plot(t, Ts_bottom_line, "--", color="C1", linewidth=2)
    plt.plot(t, Mp_top_line, "--", color="C2", label="Mp, 20%", linewidth=2)
    plt.plot(t, Mp_bottom_line, "--", color="C2", linewidth=2)
    plt.legend(loc="upper right")

    plt.subplot(3,1,3)
    plt.plot(t, data[:, 2], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('u(t) [Nm]')    
    plt.grid()
    
    plt.suptitle("x(t), θ(t), u(t) for cartpole under lqr control")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/q3b.pdf")
    plt.close()
    

def lqr():
    # Define system matrices
    M = 1
    m = 0.2
    L = 0.3
    g = 9.81
    
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, (m*g)/M, 0],
        [0, 0, 0, 1],
        [0, 0, ((M+m)*g)/(L*M), 0]
    ])
    B = np.array([
        [0],
        [1/M],
        [0],
        [1/(L*M)]
    ])

    # Define cost matrices
    Q = np.array([
        [10000, 0, 0, 0],
        [0,     1, 0, 0],
        [0,     0, 1, 0],
        [0,     0, 0, 1],
    ])  
    
    R = 0.01*np.array([[1]])
    
    # Solve for P
    P = solve_continuous_are(A, B, Q, R)
    
    # Form K_lqr    
    return np.linalg.inv(R) @ B.T @ P


def get_q(d):
    return np.array([d.qpos[0], 
                     d.qvel[0], 
                     d.qpos[1], 
                     d.qvel[1]
                     ])


def q3b():
    m, d = load_model("hw3/cartpole.xml")
    reset(m, d, "up")
    viewer = mujoco.viewer.launch_passive(m, d)
    camera_presets = {
                   "lookat": [0.0, 0.0, 0.1], 
                   "distance": 1, 
                   "azimuth": 90, 
                   "elevation": 0
                }
    set_cam(viewer, camera_presets, show_world_csys=False, show_body_csys=False)

    tmax = 2
    dt = m.opt.timestep
    ts = round(tmax/dt)
    data = np.zeros((ts, 3))
    time = np.arange(0, ts*dt, dt)
    
    K_lqr = lqr()

    for t in range(ts):

        q = get_q(d)

        u = -K_lqr @ q

        d.ctrl = u

        data[t] = np.concatenate([[q[0], q[2]], u], axis=0)
        
        mujoco.mj_step(m, d)
        viewer.sync()
        
    viewer.close()
    plot(time, data)



if __name__ == "__main__":
    q3b()