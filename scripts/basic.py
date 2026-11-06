from pathlib import Path
import sys
import time

import numpy as np
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

model = mujoco.MjModel.from_xml_path(str(ROOT / "assets/models/scene_with_table.xml"))
data = mujoco.MjData(model)
print(model.opt.timestep)

mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)  # sets qpos and ctrl to home
adr = model.joint("block_joint").qposadr[0]
data.qpos[adr:adr + 7] = [0.8, 0.0, 0.025, 1.0, 0.0, 0.0, 0.0]

mujoco.mj_forward(model, data)

with mujoco.viewer.launch(model, data):
    pass