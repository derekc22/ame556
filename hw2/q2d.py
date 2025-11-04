import mujoco
import numpy as np
from utils.utils import *

# pi/6 = 0.5235987756

def q2d():
    
    m, d = load_model("hw2/cartpole.xml")
    reset(m, d, "up")
    viewer = mujoco.viewer.launch_passive(m, d)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    
    tmax = 2
    ts = int(tmax/m.opt.timestep)
    data = np.zeros((ts, 2))

    for t in range(ts):
        data[t] = d.qpos
        # print(d.qpos)
        mujoco.mj_step(m, d)
        viewer.sync()


    viewer.close()
    np.savetxt("q2dmj.txt", data)


if __name__ == "__main__":
    q2d()