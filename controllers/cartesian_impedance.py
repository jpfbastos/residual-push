import numpy as np
import mujoco


class CartesianImpedance:
    def __init__(self, model, data, site_name, arm_dof_ids, arm_qpos, Kp, Kr, Kn,
                 Kd=10, Dr=5, Dn=5, e_max=0.1):
        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        self.arm_qpos = arm_qpos
        self.dofs = arm_dof_ids
        self.Kp, self.Kd, self.Kr, self.Dr, self.Kn, self.Dn = Kp, Kd, Kr, Dr, Kn, Dn
        self.e_max = e_max

    def _jacobian(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        return jacp[:, self.dofs], jacr[:, self.dofs]

    def _arm_mass_matrix(self):
        M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.data, M)
        # filter by dof indices in case there is a free moving joint
        return M[np.ix_(self.dofs, self.dofs)]

    def compute_torque(self, x_des, R_des, q_des=None):
        Jp, Jr = self._jacobian()
        qd = self.data.qvel[self.dofs]
        ep = x_des - self.data.site_xpos[self.site_id]
        ep *= min(1., self.e_max / float(np.linalg.norm(ep)+1e-9))
        vp = Jp @ qd
        Fp = self.Kp * ep - self.Kd * vp

        er = self.orientation_error(R_des)
        omega = Jr @ qd
        Fr = self.Kr * er - self.Dr * omega
        F = np.concatenate([Fp, Fr])
        J = np.vstack([Jp, Jr])

        tau = self.data.qfrc_bias[self.dofs] + J.T @ F
        if q_des is not None:
            tau += self.nullspace_torque(q_des)
        return tau

    def nullspace_torque(self, q_des):
        J = np.vstack(self._jacobian())
        M_arm = self._arm_mass_matrix()
        Minv = np.linalg.inv(M_arm)
        Lam = np.linalg.inv(J @ Minv @ J.T + 1e-6 * np.eye(6))
        Jbar = Minv @ J.T @ Lam

        N = np.eye(len(self.dofs)) - J.T @ Jbar.T
        q = self.data.qpos[self.arm_qpos]
        qdot = self.data.qvel[self.dofs]
        tau0 = self.Kn * (q_des - q) - self.Dn * qdot
        return N @ tau0

    def effective_mass(self):
        Minv = np.linalg.inv(self._arm_mass_matrix())
        Jp, Jr = self._jacobian()
        Lambda_p = np.linalg.inv(Jp @ Minv @ Jp.T)
        Lambda_r = np.linalg.inv(Jr @ Minv @ Jr.T)
        m_eff = np.diag(Lambda_p)
        I_eff = np.diag(Lambda_r)
        return m_eff, I_eff

    def orientation_error(self, R_des):
        R = self.data.site_xmat[self.site_id]
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R)
        q_des = np.zeros(4)
        mujoco.mju_mat2Quat(q_des, R_des.flatten())

        q_err = np.zeros(4)
        q_conj = q.copy()
        q_conj[1:] *= -1
        mujoco.mju_mulQuat(q_err, q_des, q_conj)

        if q_err[0] < 0:
            q_err *= -1
        w = np.clip(q_err[0], -1.0, 1.0)

        v = q_err[1:]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-8:
            return np.zeros(3)
        theta = 2.0 * np.arctan2(v_norm, w)
        return theta * v / v_norm