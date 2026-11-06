import argparse
import sys
from pathlib import Path

import numpy as np
import mujoco
import yaml
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from controllers.cartesian_impedance import CartesianImpedance

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
SITE = "attachment_site"
KEYFRAME = "home"


class CITest:
    # simulates the true model, controller only sees a nominal copy

    def __init__(self, xml, Kp, Kr, Kn, Dn, Kd=None, Dr=None, tip_mass=0.0, e_max=1.0):
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.nom_model = mujoco.MjModel.from_xml_path(xml)
        self.nom_data = mujoco.MjData(self.nom_model)
        m = self.model

        self.dofs = [int(m.joint(n).dofadr[0]) for n in ARM_JOINTS]
        self.qpos_ids = [int(m.joint(n).qposadr[0]) for n in ARM_JOINTS]
        self.act = [m.joint(n).id for n in ARM_JOINTS]
        self.lo, self.hi = self._torque_limits()

        self.ci = CartesianImpedance(
            self.nom_model, self.nom_data, SITE, self.dofs, self.qpos_ids,
            Kp=Kp, Kr=Kr, Kn=Kn, Dn=Dn, e_max=e_max)
        self.site_id = self.ci.site_id
        self.tip_body = int(m.site_bodyid[self.site_id])
        self.Kp, self.Kr = Kp, Kr

        # extra payload goes on the true model only
        if tip_mass:
            self.model.body_mass[self.tip_body] += tip_mass
            mujoco.mj_setConst(self.model, self.data)

        self.reset()
        self.m_eff, self.I_eff = self.ci.effective_mass()

        if Kd is None:
            Kd = 2.0 * np.sqrt(Kp * self.m_eff)
        if Dr is None:
            Dr = 2.0 * np.sqrt(Kr * self.I_eff)
        self.ci.Kd, self.ci.Dr = np.asarray(Kd, float), np.asarray(Dr, float)

    def _torque_limits(self):
        m = self.model
        lo = np.full(len(self.act), -np.inf)
        hi = np.full(len(self.act), np.inf)
        for k, a in enumerate(self.act):
            if m.actuator_ctrllimited[a]:
                lo[k], hi[k] = m.actuator_ctrlrange[a]
        return lo, hi

    def _sync_nominal(self):
        self.nom_data.qpos[:] = self.data.qpos
        self.nom_data.qvel[:] = self.data.qvel
        mujoco.mj_forward(self.nom_model, self.nom_data)

    def reset(self):
        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME)
        if key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._sync_nominal()
        self.sat_steps = 0

    def home(self):
        return (self.nom_data.site_xpos[self.site_id].copy(),
                self.nom_data.site_xmat[self.site_id].reshape(3, 3).copy(),
                self.data.qpos[self.qpos_ids].copy())

    def run(self, T, x_des, R_des, q_des, force=None):
        n = int(round(T / self.model.opt.timestep))
        t = np.zeros(n)
        x = np.zeros((n, 3))
        e_rot = np.zeros((n, 3))
        for i in range(n):
            self._sync_nominal()
            t[i] = self.data.time
            x[i] = self.nom_data.site_xpos[self.site_id]
            e_rot[i] = self.ci.orientation_error(R_des)

            tau = self.ci.compute_torque(x_des, R_des, q_des)
            tau_c = np.clip(tau, self.lo, self.hi)
            self.sat_steps += int(np.any(tau != tau_c))
            self.data.ctrl[self.act] = tau_c

            self.data.qfrc_applied[:] = 0.0
            if force is not None:
                qfrc = np.zeros(self.model.nv)
                mujoco.mj_applyFT(self.nom_model, self.nom_data,
                                  np.asarray(force, float), np.zeros(3),
                                  self.nom_data.site_xpos[self.site_id],
                                  self.tip_body, qfrc)
                self.data.qfrc_applied[:] = qfrc
            mujoco.mj_step(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0
        return t, x, e_rot


def overshoot_zeta(y, target):
    Mp = max(0.0, (y.max() - target) / target)
    if Mp < 1e-3:
        return Mp, None
    lm = np.log(Mp)
    return Mp, -lm / np.sqrt(np.pi ** 2 + lm ** 2)


def settle_time(t, y, target, tol=0.02):
    bad = np.where(np.abs(y - target) > tol * abs(target))[0]
    if len(bad) == 0:
        return 0.0
    if bad[-1] == len(y) - 1:
        return float("nan")
    return float(t[bad[-1] + 1])


def flag(ok):
    return "OK   " if ok else "CHECK"


def save_plot(outdir, name, title, xlabel, ylabel, series, hlines=()):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    for label, xs, ys in series:
        plt.plot(xs, ys, label=label)
    for val, label in hlines:
        plt.axhline(val, ls="--", color="gray", lw=1, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.savefig(outdir / name, dpi=120)
    plt.close()


def test_step(args):
    print("\n=== 1. step response ===")
    ci_test = CITest(args.xml, args.kp, args.kr, args.kn, args.dn, Kd=args.Kd, Dr=args.Dr)
    print(f"m_eff (x,y,z) = {np.round(ci_test.m_eff, 2)} kg   I_eff = {np.round(ci_test.I_eff, 3)} kg m^2")
    print(f"Kd = {np.round(ci_test.ci.Kd, 1)}   Dr = {np.round(ci_test.ci.Dr, 2)}")

    # 10 cm step in x, orientation and posture held
    step = 0.10
    x0, R0, q0 = ci_test.home()
    t, x, _ = ci_test.run(args.T, x0 + np.array([step, 0, 0]), R0, q0)
    y = x[:, 0] - x0[0]
    Mp, zeta = overshoot_zeta(y, step)
    ts = settle_time(t, y, step)
    wn = np.sqrt(args.kp / ci_test.m_eff[0])
    ts_pred = 5.83 / wn
    off = np.abs(x[:, 1:] - x0[1:]).max() * 1000
    print(f"\nposition step {step*100:.0f} cm in x")
    print(f"  overshoot      {Mp*100:6.2f} %    zeta from overshoot: "
          f"{'>~1 (none)' if zeta is None else f'{zeta:.2f}'}")
    print(f"  2% settling    {ts:6.2f} s    predicted: {ts_pred:.2f} s")
    print(f"  final error    {abs(y[-1]-step)*1000:6.2f} mm   off-axis excursion (y,z): {off:.1f} mm")
    print(f"  torque-saturated steps: {ci_test.sat_steps}")
    print(f"  [{flag(Mp < 0.05 and np.isfinite(ts) and ts < 1.5 * ts_pred)}] "
          "overshoot < 5% and settling within 1.5x prediction")
    if args.plots:
        save_plot(args.plots, "step_position.png", "Position step (x)", "t [s]",
                  "displacement [m]", [("x", t, y)], [(step, "target")])

    # 20 deg about world x, position and posture held
    ci_test.reset()
    x0, R0, q0 = ci_test.home()
    th0 = np.deg2rad(20.0)
    R_des = Rotation.from_euler("x", 20, degrees=True).as_matrix() @ R0
    t, x, e = ci_test.run(args.T, x0, R_des, q0)
    y = th0 - e[:, 0]
    Mp, zeta = overshoot_zeta(y, th0)
    ts = settle_time(t, y, th0)
    wn = np.sqrt(args.kr / ci_test.I_eff[0])
    ts_pred = 5.83 / wn
    drift = np.abs(x - x0).max() * 1000
    print("\nrotation step 20 deg about world x")
    print(f"  initial |error|  {np.linalg.norm(e[0]):.4f} rad (expect {th0:.4f})")
    print(f"  overshoot      {Mp*100:6.2f} %    zeta from overshoot: "
          f"{'>~1 (none)' if zeta is None else f'{zeta:.2f}'}")
    print(f"  2% settling    {ts:6.2f} s    predicted: {ts_pred:.2f} s")
    print(f"  tip position drift during rotation: {drift:.1f} mm")
    print(f"  torque-saturated steps: {ci_test.sat_steps}")
    print(f"  [{flag(Mp < 0.05 and np.isfinite(ts) and ts < 1.5 * ts_pred)}] "
          "overshoot < 5% and settling within 1.5x prediction")
    if args.plots:
        save_plot(args.plots, "step_rotation.png", "Rotation step (x)", "t [s]",
                  "angle [rad]", [("angle", t, y)], [(th0, "target")])


def test_stiffness(args):
    print("\n=== 2. static stiffness (tip force along +y) ===")
    forces = np.array([5.0, 10.0, 15.0, 20.0])
    defl = []
    for F in forces:
        ci_test = CITest(args.xml, args.kp, args.kr, args.kn, args.dn, Kd=args.Kd, Dr=args.Dr)
        x0, R0, q0 = ci_test.home()
        t, x, _ = ci_test.run(args.T, x0, R0, q0, force=[0, F, 0])
        tail = t > t[-1] - 0.3
        d = x[tail, :].mean(axis=0) - x0
        defl.append(d[1])
        print(f"  F = {F:5.1f} N   deflection y = {d[1]*1000:7.2f} mm   "
              f"(predicted {F/args.kp*1000:6.2f})   off-axis x,z = "
              f"{d[0]*1000:.2f}, {d[2]*1000:.2f} mm")
    defl = np.array(defl)
    compliance = float(np.sum(forces * defl) / np.sum(forces ** 2))
    pred = 1.0 / args.kp
    print(f"\n  fitted compliance {compliance*1000:.3f} mm/N   predicted 1/Kp = {pred*1000:.3f} mm/N"
          f"   ratio {compliance/pred:.3f}")
    print(f"  [{flag(abs(compliance / pred - 1) < 0.1)}] slope within 10% of 1/Kp")
    if args.plots:
        save_plot(args.plots, "stiffness.png", "Static deflection vs force", "force [N]",
                  "deflection [m]", [("measured", forces, defl),
                                     ("1/Kp", forces, forces / args.kp)])


def test_mass(args):
    print(f"\n=== 3. unmodelled tip mass (+{args.mass} kg on the true model) ===")
    g = abs(mujoco.MjModel.from_xml_path(args.xml).opt.gravity[2])
    results = {}
    for dm in (0.0, args.mass):
        ci_test = CITest(args.xml, args.kp, args.kr, args.kn, args.dn, Kd=args.Kd, Dr=args.Dr,
                  tip_mass=dm)
        x0, R0, q0 = ci_test.home()
        t, x, e = ci_test.run(args.T, x0, R0, q0)
        tail = t > t[-1] - 0.5
        d = x[tail].mean(axis=0) - x0
        results[dm] = (t, x[:, 2] - x0[2])
        print(f"  extra mass {dm:4.1f} kg   steady offset (x,y,z) = "
              f"{d[0]*1000:7.2f}, {d[1]*1000:7.2f}, {d[2]*1000:7.2f} mm   "
              f"rot err {np.linalg.norm(e[tail].mean(axis=0)):.4f} rad   "
              f"saturated steps {ci_test.sat_steps}")
        if dm == 0.0:
            base_z = d[2]
            print(f"  [{flag(abs(d[2]) < 5e-4)}] baseline hold drift < 0.5 mm")
        else:
            pred = -dm * g / args.kp
            print(f"\n  predicted sag  -m g / Kp = {pred*1000:.2f} mm in z   "
                  f"measured {(d[2]-base_z)*1000:.2f} mm   ratio {(d[2]-base_z)/pred:.2f}")
            print(f"  [{flag(abs((d[2]-base_z) / pred - 1) < 0.15)}] sag within 15% of m g / Kp")
    if args.plots:
        save_plot(args.plots, "mass_sag.png", "Tip z vs time", "t [s]", "z displacement [m]",
                  [(f"+{dm} kg", *results[dm]) for dm in results])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default=str(ROOT / "assets/models/scene.xml"))
    p.add_argument("--kp", type=float, default=300.0)
    p.add_argument("--kr", type=float, default=50.0)
    p.add_argument("--kn", type=float, default=50.0)
    p.add_argument("--dn", type=float, default=15.0)
    p.add_argument("--mass", type=float, default=1.0, help="unmodelled tip mass [kg]")
    p.add_argument("--T", type=float, default=3.0, help="seconds per run")
    p.add_argument("--tests", default="step,stiffness,mass")
    p.add_argument("--plots", default=None, help="directory for PNGs")
    p.add_argument("--config", default=None,
                   help="yaml with frozen gains, overrides --kp/--kr/--kn/--dn")
    args = p.parse_args()

    args.Kd = args.Dr = None
    if args.config:
        c = yaml.safe_load(open(args.config))["controller"]
        args.kp, args.kr, args.kn, args.dn = c["Kp"], c["Kr"], c["Kn"], c["Dn"]
        args.Kd, args.Dr = c.get("Kd"), c.get("Dr")

    tests = {"step": test_step, "stiffness": test_stiffness, "mass": test_mass}
    for name in args.tests.split(","):
        tests[name.strip()](args)


if __name__ == "__main__":
    main()