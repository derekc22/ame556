import mujoco
import numpy as np


def load_model(model_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    return m, d

def reset(m: mujoco.MjModel, 
          d: mujoco.MjData, 
          keyframe: str) -> None:
    init_qpos = m.keyframe(keyframe).qpos
    init_qvel = m.keyframe(keyframe).qvel
    mujoco.mj_resetData(m, d) 
    d.qpos = init_qpos
    d.qvel = init_qvel
    mujoco.mj_forward(m, d)


# pi/6 = 0.5235987756

def main():
    
    m, d = load_model("cartpole.xml")
    reset(m, d, "up")
    viewer = mujoco.viewer.launch_passive(m, d)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
    
    tmax = 2
    tsteps = int(tmax/m.opt.timestep)
    data = np.zeros((int(tsteps), 2))

    for t in range(tsteps):
        data[t] = d.qpos
        # print(d.qpos)
        mujoco.mj_step(m, d)
        viewer.sync()


    viewer.close()
    np.savetxt("q2dmj.txt", data)


if __name__ == "__main__":
    main()