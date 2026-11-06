import time
from pathlib import Path
import numpy as np
import mujoco, mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/models/scene.xml"))
data = mujoco.MjData(model)

arm_dofs = [model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)]
arm_qpos = [model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)]

mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
mujoco.mj_forward(model, data)
q0 = data.qpos[arm_qpos].copy()

with mujoco.viewer.launch_passive(model, data) as v:
    last_print = 0.0
    while v.is_running():
        t0 = time.time()
        data.ctrl[:] = data.qfrc_bias[arm_dofs]
        mujoco.mj_step(model, data)
        v.sync()

        if data.time - last_print >= 1.0:
            drift = np.abs(data.qpos[arm_qpos] - q0).max()
            print(f"t={data.time:5.1f}s  drift={drift:.2e}")
            last_print = data.time

        if (rest := model.opt.timestep - (time.time() - t0)) > 0:
            time.sleep(rest)