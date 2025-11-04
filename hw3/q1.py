import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
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

    plt.subplot(3,1,1)
    plt.plot(t, data[:, 0], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('x(t) [m]')
    plt.grid()

    plt.subplot(3,1,2)
    plt.plot(t, data[:, 1], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('y(t) [m]')    
    plt.grid()

    plt.subplot(3,1,3)
    plt.plot(t,  data[:, 2], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('θ(t) [rad]')
    plt.grid()
    
    plt.suptitle("x(t), y(t), θ(t) for free-falling bar")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/q1.pdf")
    plt.close()

def q1():
    
    m, d = load_model("hw3/bar.xml")
    reset(m, d, "init")
    viewer = mujoco.viewer.launch_passive(m, d)
    camera_presets = {
                   "lookat": [0.0, 0.0, 0.15], 
                   "distance": 1, 
                   "azimuth": 270, 
                   "elevation": -10
                }
    set_cam(viewer, camera_presets, show_world_csys=False, show_body_csys=False)

    tmax = 2
    dt = m.opt.timestep
    ts = round(tmax/dt)
    data = np.zeros((ts, 3))
    time = np.arange(0, ts*dt, dt)

    for t in range(ts):

        q = d.qpos
        xz = np.r_[q[0], q[2]]
        theta =  R.from_quat(q[3:]).as_euler('zyx')[1:2] # rad
        data[t] = np.concatenate([xz, theta], axis=0)
        
        mujoco.mj_step(m, d)
        viewer.sync()
        
    viewer.close()
    plot(time, data)



if __name__ == "__main__":
    q1()