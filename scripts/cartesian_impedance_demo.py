import sys
import time
from pathlib import Path

import numpy as np
import mujoco
from mujoco import viewer
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from controllers.cartesian_impedance import CartesianImpedance

model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/models/scene.xml"))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
mujoco.mj_forward(model, data)

arm_dofs = [model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)]
arm_qpos = [model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)]

ci = CartesianImpedance(model, data, "attachment_site", arm_dofs, arm_qpos,
                        Kp=300.0, Kr=50.0, Kn=50.0,
                        Kd=np.array([119.5, 111.0, 148.5]),
                        Dr=np.array([5.5, 5.6, 2.45]),
                        Dn=15.0)

x0 = data.site_xpos[ci.site_id].copy()
R0 = data.site_xmat[ci.site_id].reshape(3, 3).copy()
q0 = data.qpos[arm_qpos].copy()

x_des = x0 + np.array([-0.8, -0.4, -0.2])
R_des = Rotation.from_euler("x", -40, degrees=True).as_matrix() @ R0

model.body("target_ball").pos[:] = x_des

with viewer.launch_passive(model, data) as v:
    wall0 = time.time()
    last_print = 0.0
    i = 0
    while v.is_running():
        mujoco.mj_forward(model, data)
        data.ctrl[arm_dofs] = ci.compute_torque(x_des, R_des, q0)
        mujoco.mj_step(model, data)

        if data.time - last_print >= 1.0:
            last_print = data.time
            pos_err = np.linalg.norm(x_des - data.site_xpos[ci.site_id]) * 1000
            rot_err = np.degrees(np.linalg.norm(ci.orientation_error(R_des)))
            print(f"t={data.time:4.1f}s  pos err {pos_err:6.2f} mm  rot err {rot_err:5.2f} deg")
        time.sleep(max(0.0, data.time - (time.time() - wall0)))

        i += 1
        if i % 20 == 0:
            v.sync()