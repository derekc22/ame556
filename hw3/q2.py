import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.utils import *


def plot(t, data, figname):
    plot_dir = "hw3/plots"
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
    plt.ylabel('q1(t) [rad]')
    plt.grid()
    
    plt.subplot(4,2,5)
    plt.plot(t,  data[:, 4], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('q2(t) [rad]')
    plt.grid()
    
    plt.subplot(4,2,6)
    plt.plot(t,  data[:, 5], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('q3(t) [rad]')
    plt.grid()
    
    plt.subplot(4,2,7)
    plt.plot(t,  data[:, 6], linewidth=2)
    plt.xlabel('t [s]')
    plt.ylabel('q4(t) [rad]')
    plt.grid()
    
    if figname == "q2a":
        plt.suptitle("x(t), y(t), θ(t), qi(t) for 2D biped")
    elif figname == "q2b":
        plt.suptitle("x(t), y(t), θ(t), qi(t) for 2D biped under pd control")
        
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{figname}.pdf")
    plt.close()
        

def q2a():
    
    m, d = load_model("hw3/biped.xml")
    reset(m, d, "init")
    viewer = mujoco.viewer.launch_passive(m, d)

    cam_presets = {
                   "lookat": [0.0, 0.0, 0.15], 
                   "distance": 1, 
                   "azimuth": 90, 
                   "elevation": -10
                }    
    set_cam(viewer, cam_presets, show_world_csys=False, show_body_csys=False)

    tmax = 2
    dt = m.opt.timestep
    ts = round(tmax/dt)
    data = np.zeros((ts, 7))
    time = np.arange(0, ts*dt, dt)

    for t in range(ts):

        qp = d.qpos
        xz = np.r_[qp[0], qp[2]]
        theta =  R.from_quat(qp[3:7]).as_euler('zyx')[1] # rad
        data[t] = np.concatenate([xz, [theta], qp[7:]], axis=0)
        
        mujoco.mj_step(m, d)
        viewer.sync()
        
    viewer.close()
    plot(time, data, "q2a")


def pd(qpos, qvel):
    qdes = np.array([-np.pi/3, np.pi/2, 0, np.pi/2])
    kp = 50
    kd = 2
    
    qp = qpos[7:]
    qv = qvel[6:]
    
    return kp*(qdes-qp) - kd*qv
    
    

def q2b():
    
    m, d = load_model("hw3/biped.xml")
    reset(m, d, "init")
    viewer = mujoco.viewer.launch_passive(m, d)
    cam_presets = {
                   "lookat": [0.0, 0.0, 0.55], 
                   "distance": 2, 
                   "azimuth": 90, 
                   "elevation": -10
                }  
    set_cam(viewer, cam_presets, show_world_csys=False, show_body_csys=True)

    tmax = 2
    dt = m.opt.timestep
    ts = round(tmax/dt)
    data = np.zeros((ts, 7))
    time = np.arange(0, ts*dt, dt)

    for t in range(ts):
        
        qp = d.qpos
        qv = d.qvel
        
        u = pd(qp, qv)
        d.ctrl = u

        xz = np.r_[qp[0], qp[2]]
        theta =  R.from_quat(qp[3:7]).as_euler('zyx')[1] # rad
        data[t] = np.concatenate([xz, [theta], qp[7:]], axis=0)
        
        mujoco.mj_step(m, d)
        viewer.sync()
        
    viewer.close()
    plot(time, data, "q2b")
    



if __name__ == "__main__":
    q2a()
    q2b()