import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import sys
import time
from pathlib import Path
from gymnasium.utils import seeding

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from controllers.cartesian_impedance import CartesianImpedance

class PushEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    MAX_DELTA = 1

    def __init__(self, seed=None, success_dist=None, n_substeps=25, max_steps = 400, render_mode=None):
        super().__init__()
        self.np_random, self.seed = seeding.np_random(seed)
        self.success_dist = success_dist
        self.n_substeps = n_substeps
        self.max_steps = max_steps
        self.steps = 0

        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(
            str(ROOT / "assets/models/scene_with_table.xml")
        )
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        self._init_geometry()
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self._init_obs_block()

        self.arm_dofs = [self.model.joint(f"joint{i}").dofadr[0] for i in range(1, 8)]
        self.arm_qpos = [self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)]

        self.q0 = self.model.key("home").qpos[self.arm_qpos].copy()

        self.ci = CartesianImpedance(self.model,
                                     self.data,
                                     "pusher_tip",
                                     self.arm_dofs,
                                     self.arm_qpos,
                                     Kp=300.0, Kr=50.0, Kn=50.0,
                                     Kd=np.array([119.5, 111.0, 148.5]),
                                     Dr=np.array([5.5, 5.6, 2.45]),
                                     Dn=15.0)

        self.pos_target = np.zeros(3)
        self.ori_target = np.zeros((3,3))

        self._viewer = None
        if render_mode is not None:
            self.render()

        self._prev_dist = 0
        self._prev_contact_dist = 0
        self._prev_angle = 0
        self._total_steps = 0

    def _init_geometry(self):
        mujoco.mj_forward(self.model, self.data)  # so geom_xpos is valid

        tid = self.model.geom("table").id  # geom names, not body names: check your XML
        half = self.model.geom_size[tid]
        c = self.data.geom_xpos[tid]
        assert np.allclose(self.data.geom_xmat[tid], np.eye(3).ravel()), "table is rotated"
        self.table_top = c[2] + half[2]
        self.table_x = (c[0] - half[0], c[0] + half[0])
        self.table_y = (c[1] - half[1], c[1] + half[1])

        block_half = self.model.geom("block").size
        pusher_r = self.model.geom("pusher").size[0]
        self.block_z = self.table_top + block_half[2]
        self.goal_z = self.table_top + 0.0005

        self.goal_geom_id = self.model.geom("goal_marker").id

        # block spawn region: shrink by the block's half-diagonal (random yaw) plus a margin
        r = np.hypot(block_half[0], block_half[1]) + 0.4
        self.spawn_x = (self.table_x[0] + r, self.table_x[1] - r)
        self.spawn_y = (self.table_y[0] + r, self.table_y[1] - r)

    def _init_obs_block(self):
        self.observation_space = spaces.Box(-np.inf, np.inf, (19,), dtype=np.float32)

        self._block_id = self.model.body("block").id
        self._tip_site_id = self.model.site("pusher_tip").id
        self._goal_mocap_id = self.model.body_mocapid[self.model.body("goal").id]
        self._vel6 = np.zeros(6)

        self.prev_action = np.zeros(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)  # sets qpos and ctrl to home
        # get idx of joint
        adr = self.model.joint("block_joint").qposadr[0]
        block_x = self.np_random.uniform(*self.spawn_x)
        block_y = self.np_random.uniform(*self.spawn_y)

        yaw = self.np_random.uniform(-np.pi, np.pi)
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self.data.qpos[adr:adr + 7] = np.concatenate([[block_x, block_y, self.block_z], quat])

        min_goal_dist = 0.15

        while True:
            goal_x = self.np_random.uniform(*self.spawn_x)
            goal_y = self.np_random.uniform(*self.spawn_y)

            if np.hypot(goal_x - block_x, goal_y - block_y) >= min_goal_dist:
                break
        self.data.mocap_pos[self._goal_mocap_id] = [goal_x, goal_y, self.goal_z]

        mujoco.mj_forward(self.model, self.data)
        self.pos_target = self.data.site_xpos[self._tip_site_id].copy()
        self.ori_target = self.data.site_xmat[self._tip_site_id].reshape(3, 3).copy()

        self.steps = 0
        self.prev_action = np.zeros(3)
        self.success_steps = 0

        self._prev_dist = self._block_goal_dist()
        self._prev_contact_dist = self._tip_contact_dist()
        self._prev_angle = self._goal_block_tip_angle()
        self._in_zone = False

        return self._get_obs(), {}

    def step(self, action):
        a_norm = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        action = a_norm * self.MAX_DELTA

        mujoco.mj_forward(self.model, self.data)
        self.pos_target = self.data.site_xpos[self._tip_site_id] + action
        for _ in range(self.n_substeps):
            self.data.ctrl[self.arm_dofs] = self.ci.compute_torque(self.pos_target, self.ori_target, self.q0)
            mujoco.mj_step(self.model, self.data)
            if self._viewer is not None and self._viewer.is_running():
                self._viewer.sync()
                time.sleep(self.model.opt.timestep)
        mujoco.mj_forward(self.model, self.data)

        self.prev_action = a_norm
        self.steps += 1

        dist = self._block_goal_dist()
        contact_dist = self._tip_contact_dist()
        angle = self._goal_block_tip_angle()
        success = False
        lost = False
        success_dist = self._get_success_dist()
        if dist < success_dist:
            self.success_steps += 1
            self._in_zone = True
            if self.success_steps >= 20:
                success = True
        else:
            self.success_steps = 0
            if self._in_zone: lost = True
            self._in_zone = False
        off_table = not self._is_on_table()

        z_pos = self.data.site_xpos[self._tip_site_id][2]

        in_zone = dist < success_dist
        # Per-term breakdown is logged to TensorBoard by InfoLoggerCallback.
        terms = {
            "progress": 2.0 * (self._prev_dist - dist),
            "contact_approach": 0.5 * (self._prev_contact_dist - contact_dist),
            "alignment": 0.001 * (self._prev_angle - angle),
            "in_zone": 0.05 * in_zone,
            "success": 10.0 * success,
            "lost": -1.0 * lost,
            "off_table": -10.0 * off_table,
            "tip_low": -0.02 * (z_pos < 0.01),
        }
        reward = sum(terms.values())

        self._prev_dist = dist
        self._prev_contact_dist = contact_dist
        self._prev_angle = angle

        self.model.geom_size[self.goal_geom_id, 0] = success_dist

        terminated = success or off_table
        truncated = self.steps > self.max_steps
        if terminated or truncated:
            self._total_steps += self.steps
        info = {
            "is_success": success,
            "off_table": off_table,
            "dist": dist,
            "success_dist": success_dist,
            "terms": terms,
        }
        return self._get_obs(), float(reward), terminated, truncated, info

    def _block_goal_dist(self):
        goal = self.data.mocap_pos[self._goal_mocap_id]
        block = self.data.xpos[self._block_id]
        return float(np.linalg.norm((goal - block)[:2]))

    def _tip_contact_dist(self):
        goal = self.data.mocap_pos[self._goal_mocap_id]
        block = self.data.xpos[self._block_id]
        tip = self.data.site_xpos[self._tip_site_id]
        block_to_goal = goal[:2] - block[:2]
        block_to_goal_dist = np.linalg.norm(block_to_goal)
        block_to_goal_normalised = block_to_goal / (block_to_goal_dist + 1e-8)

        contact_point = block.copy()
        contact_point[:2] -= 0.04 * block_to_goal_normalised

        return float(np.linalg.norm(tip - contact_point))

    def _goal_block_tip_angle(self):
        goal = self.data.mocap_pos[self._goal_mocap_id]
        block = self.data.xpos[self._block_id]
        tip = self.data.site_xpos[self._tip_site_id]
        v1 = goal - block
        v1[2] = 0 # assume xy only
        v2 = block - tip
        v1_normed = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_normed = v2 / (np.linalg.norm(v2) + 1e-8)
        return np.arccos(np.clip(np.dot(v1_normed, v2_normed), -1.0, 1.0))


    def _is_on_table(self, margin=0.0):
        p = self.data.xpos[self._block_id]
        in_xy = (self.table_x[0] + margin <= p[0] <= self.table_x[1] - margin and
                 self.table_y[0] + margin <= p[1] <= self.table_y[1] - margin)
        return bool(in_xy and p[2] > self.table_top - 0.01)

    def _get_success_dist(self):
        return self.success_dist if self.success_dist is not None else 0.01

    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def close(self):
        if self._viewer is not None:
            self._viewer.close()

    def _get_obs(self):
        # 19 dim:
        # vector between goal and block (xy):      2
        # vector between block and pusher (xyz):   3
        # block pose yaw as sin/cos pair:          2
        # block linear velocity (xy):              2
        # block yaw rate:                          1
        # pusher state absolute position (xyz):    3
        # pusher state absolute velocity (xyz):    3
        # previous action (xyz):                   3
        d = self.data

        goal = d.mocap_pos[self._goal_mocap_id]
        block = d.xpos[self._block_id]
        tip = d.site_xpos[self._tip_site_id]

        Rm = d.xmat[self._block_id].reshape(3, 3)
        yaw = np.arctan2(Rm[1, 0], Rm[0, 0])

        block_vel = self._obj_vel(mujoco.mjtObj.mjOBJ_BODY, self._block_id)
        tip_vel = self._obj_vel(mujoco.mjtObj.mjOBJ_SITE, self._tip_site_id)

        # TODO (perturbation harness): add observation noise to the block pose

        parts = [
            (goal - block)[:2],  # block -> goal, xy           (2)
            block - tip,  # pusher -> block, xyz               (3)
            [np.sin(yaw), np.cos(yaw)],  # block yaw           (2)
            block_vel[3:5],  # block linear velocity, xy       (2)
            [block_vel[2]],  # block yaw rate (world z)        (1)
            tip,  # pusher absolute position                   (3)
            tip_vel[3:6],  # pusher linear velocity            (3)
            self.prev_action, # previous action                (3)
        ]

        obs = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in parts])
        obs = obs.astype(np.float32)
        assert obs.shape == self.observation_space.shape, (obs.shape, self.observation_space.shape)

        return obs

    def _obj_vel(self, objtype, obj_id):
        mujoco.mj_objectVelocity(self.model, self.data, objtype, obj_id, self._vel6, 0)
        return self._vel6.copy()

    def _set_success_dist(self, d):
        self.success_dist = d
