import mujoco
import numpy as np
from scipy.linalg import expm


def load_model(model_path: str) -> tuple[mujoco.MjModel, mujoco.MjData]:    
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    return m, d

# def reset(m: mujoco.MjModel, 
#           d: mujoco.MjData, 
#           keyframe: str) -> None:
#     init_qpos = m.keyframe(keyframe).qpos
#     init_qvel = m.keyframe(keyframe).qvel
#     mujoco.mj_resetData(m, d) 
#     d.qpos = init_qpos
#     d.qvel = init_qvel
#     mujoco.mj_forward(m, d)

def reset(m: mujoco.MjModel, 
          d: mujoco.MjData, 
          keyframe: str) -> None:
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)
    
# def reset(m: mujoco.MjModel, 
#           d: mujoco.MjData, 
#           keyframe: str) -> None:
#     init_qpos = m.keyframe(keyframe).qpos
#     init_qvel = m.keyframe(keyframe).qvel
#     key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, keyframe)
#     mujoco.mj_resetDataKeyframe(m, d, key_id)
#     d.qpos = init_qpos
#     d.qvel = init_qvel
#     mujoco.mj_forward(m, d)

    
def set_cam(viewer: mujoco.viewer,
            track: bool = False,
            presets: dict = None,
            show_world_csys: bool = False,
            show_body_csys: bool = False) -> None:
    
    if track:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 0
    else:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    
    if presets is None:
        presets = {
                    "lookat": [0.0, 0.0, 0.0], 
                    "distance": 2.0, 
                    "azimuth": 90,
                    "elevation": -45
                }

    viewer.cam.lookat[:] = presets["lookat"]        # Point the camera is looking at
    viewer.cam.distance = presets["distance"]       # Distance from lookat point
    viewer.cam.azimuth = presets["azimuth"]         # Horizontal rotation angle [deg]
    viewer.cam.elevation = presets["elevation"]     # Vertical rotation angle [deg]
    
    if show_body_csys:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
    if show_world_csys:
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD

def discretize(A, B, Ts):
    # Applies a zero-order hold
    
    n = A.shape[0]
    m = B.shape[1]

    # Build augmented matrix
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B

    # Matrix exponential
    Mk = expm(M * Ts)

    # Extract discrete A and B
    Ak = Mk[:n, :n]
    Bk = Mk[:n, n:]

    return Ak, Bk

# def get_site_id(m: mujoco.MjModel,
#                 site: str) -> int:
#     return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)

# def get_site_xpos(m: mujoco.MjModel, 
#                   d: mujoco.MjData, 
#                   site: str) -> np.ndarray:
#     site_id = get_site_id(m, site)
#     return d.site(site_id).xpos

def get_geom_id(m: mujoco.MjModel, 
                geom: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, geom)

def get_geom_name(m: mujoco.MjModel, 
                  id: int) -> str:
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, id)

def get_feet_contact(m: mujoco.MjModel, 
                     d: mujoco.MjData) -> bool:
    
    contact_set = set()
    l_foot_id = get_geom_id(m, "l_foot")
    r_foot_id = get_geom_id(m, "r_foot")
    floor_id = get_geom_id(m, "floor")
    
    for k in range(d.ncon):
        c = d.contact[k]
        g1 = c.geom1
        g2 = c.geom2
        
        for foot in (l_foot_id, r_foot_id):
            if (foot in (g1, g2)) and (floor_id in (g1, g2)):
                contact_set.add(foot)
                
    return len(contact_set) == 2


def get_geom_xpos(m: mujoco.MjModel, 
                  d: mujoco.MjData, 
                  geom: str) -> np.ndarray:
    geom_id = get_geom_id(m, geom)
    return d.geom(geom_id).xpos

def get_body_id(m: mujoco.MjModel,
                body: str) -> int:
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)

def get_body_inertia(m: mujoco.MjModel,
                     body: str) -> int:
    return m.body_inertia[get_body_id(m, body)]

def get_body_mass(m: mujoco.MjModel,
                  body: str) -> int:
    return m.body_mass[get_body_id(m, body)]

def get_body_xpos(m: mujoco.MjModel, 
                  d: mujoco.MjData, 
                  body: str) -> np.ndarray:
    body_id = get_body_id(m, body)
    return d.xpos[body_id]

def get_gravity(m: mujoco.MjModel) -> float:
    return m.opt.gravity[-1]

def get_feet_xpos(m, d):
    return np.vstack([
        get_geom_xpos(m, d, "l_foot"),
        get_geom_xpos(m, d, "r_foot"),
    ])

def get_x_com(m, d):
    M = 0
    xyz_sum = np.zeros(3)
    
    for i in range(1, m.nbody):
        m_i  = m.body_mass[i]
        M += m_i
        xyz_sum += m_i * d.xpos[i]

    return xyz_sum / M

def get_M(m):
    M = 0
    for i in range(1, m.nbody):
        M += m.body_mass[i]
    return M


def get_v_com(m, d):
    M = 0
    v_sum = np.zeros(3)
    
    for i in range(1, m.nbody):
        m_i = m.body_mass[i]
        M += m_i
        # d.cvel[i] is the 6D spatial velocity (angular, linear)
        # We take the linear part [3:6]
        v_sum += m_i * d.cvel[i][3:] 
        
    return v_sum / M

def get_Izz_com(m, d):
    Izz_com = 0
    xyz_com = get_x_com(m, d)
    
    for i in range(1, m.nbody):
        Izz_com += m.body_inertia[i][1] + m.body_mass[i] * np.linalg.norm(d.xpos[i]-xyz_com)**2
        
    return Izz_com
    

