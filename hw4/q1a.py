import mujoco
import numpy as np
from scipy.linalg import solve_continuous_are
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.utils import *


def plot(t, data, case):
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
    
    plt.suptitle("x(t), θ(t), u(t) for cartpole under lqr control")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/q1a_{case}.pdf")
    plt.close()
    

def lqr(Q, r):
    # Define system matrices
    M = 1
    m = 0.2
    L = 0.3
    g = 9.81
    
    # Code is reused from hw3/q3b.py
    # Note that hw3/q3b.py uses the state definition x = [x ẋ θ θ̇]
    # But this hw uses the state definition          x = [x θ ẋ θ̇]
    # That is, ẋ and θ are swapped
    # Thus, A and B must be re-organized accordingly
    # In the original state definition, ẋ is the 2nd element while θ is the 3rd element (1-indexed)
    # Thus:
    # swap rows 2 and 3 in A
    # swap rows 2 and 3 in B
    # swap columns 2 and 3 in A
    
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
    R = np.array([[r]])
    
    # Solve for P
    P = solve_continuous_are(A, B, Q, R)
    
    # Form K_lqr    
    return np.linalg.inv(R) @ B.T @ P


def get_q(d):
    return np.concatenate([
                    d.qpos,  
                    d.qvel
                    ])


def q1a(Q, r, case):
    m, d = load_model("hw4/assets/cartpole_q1a.xml")
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
    
    K_lqr = lqr(Q, r)

    for t in range(ts):

        q = get_q(d)

        u = -K_lqr @ q

        d.ctrl = u

        data[t] = np.concatenate([q[0:2], u], axis=0)
        
        mujoco.mj_step(m, d)
        viewer.sync()
        
    viewer.close()
    plot(time, data, case)



if __name__ == "__main__":
    
    # i) Q = diag(10, 10, 10, 10); R = 1
    q1a(Q=np.diag([10, 10, 10, 10]), r=1, case="i")
    
    # ii) Q = diag(100, 0.01, 100, 0.01); R = 10
    q1a(Q=np.diag([100, 0.01, 100, 0.01]), r=10, case="ii")
    
    # iii) Q = diag(100, 0.01, 100, 0.01); R = 1
    q1a(Q=np.diag([100, 0.01, 100, 0.01]), r=1, case="iii")