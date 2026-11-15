from dataclasses import dataclass, fields, replace
from pathlib import Path

import mujoco
import numpy as np
import yaml

EVAL_CONFIG = Path(__file__).resolve().parent.parent / "configs/perturbations/eval.yaml"

PHYSICS_KEYS = (
    "block_mass",         # kg,
    "block_friction",     # sliding friction, block-table pair
    "pusher_friction",    # sliding friction, block-pusher pair
    "com_x",              # m, COM offset from block frame
    "com_y",              # m
    "contact_timeconst",  # contact stiffness (solref[0]) of both block contact pairs
    "contact_dampratio",  # damp ratio (solref[1])
)
NOISE_KEYS = (
    "obs_pos_std",        # m, Gaussian noise on observed block xy
    "obs_yaw_std",        # rad, Gaussian noise on observed block yaw
    "obs_vel_std",        # m/s and rad/s, noise on observed block velocities
)


def _coerce_spec(name, value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{name}: a range must be [lo, hi], got {value!r}")
        lo, hi = float(value[0]), float(value[1])
        if lo > hi:
            raise ValueError(f"{name}: lo > hi in {value!r}")
        return (lo, hi) if lo != hi else lo
    return float(value)


def _spec_bounds(spec):
    return spec if isinstance(spec, tuple) else (spec, spec)


@dataclass(frozen=True)
class PerturbationConfig:
    block_mass: object = None
    block_friction: object = None
    pusher_friction: object = None
    com_x: object = None
    com_y: object = None
    contact_timeconst: object = None
    contact_dampratio: object = None
    obs_pos_std: object = None
    obs_yaw_std: object = None
    obs_vel_std: object = None

    def __post_init__(self):
        for f in fields(self):
            object.__setattr__(self, f.name, _coerce_spec(f.name, getattr(self, f.name)))
        for name in ("block_mass", "contact_timeconst", "contact_dampratio"):
            spec = getattr(self, name)
            if spec is not None and _spec_bounds(spec)[0] <= 0:
                raise ValueError(f"{name} must be > 0, got {spec}")
        for name in ("block_friction", "pusher_friction") + NOISE_KEYS:
            spec = getattr(self, name)
            if spec is not None and _spec_bounds(spec)[0] < 0:
                raise ValueError(f"{name} must be >= 0, got {spec}")

    @classmethod
    def from_dict(cls, d):
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown perturbation keys: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    @classmethod
    def coerce(cls, value):
        """Accept None, a config, a dict, or a path to a YAML file."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls.from_dict(value)
        return cls.load(Path(value))

    def to_dict(self):
        """Plain dict (ranges as lists), safe for YAML/JSON and SubprocVecEnv pickling."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = list(v) if isinstance(v, tuple) else v
        return out

    def with_overrides(self, **overrides):
        unknown = set(overrides) - {f.name for f in fields(self)}
        if unknown:
            raise ValueError(f"Unknown perturbation keys: {sorted(unknown)}")
        return replace(self, **overrides)


class Perturber:

    def __init__(self, model, config, block_body="block", block_geom="block",
                 pairs=("block_table", "block_pusher")):
        self.model = model
        self.config = PerturbationConfig.coerce(config)

        self._block = model.body(block_body).id
        self._table_pair = model.pair(pairs[0]).id
        self._pusher_pair = model.pair(pairs[1]).id
        self._pair_ids = [self._table_pair, self._pusher_pair]

        self._mass0 = float(model.body_mass[self._block])
        self._inertia0 = model.body_inertia[self._block].copy()
        self._ipos0 = model.body_ipos[self._block].copy()
        self._pair_friction0 = model.pair_friction[self._pair_ids].copy()
        self._pair_solref0 = model.pair_solref[self._pair_ids].copy()
        self._block_half = model.geom(block_geom).size[:2].copy()
        self._timestep = float(model.opt.timestep)

        # The compiler marks the block as "simple" (COM at the body origin, diagonal
        # mass matrix) and skips work based on it. A COM offset breaks both
        # assumptions, so turn the shortcuts off for this body.
        model.body_sameframe[self._block] = 0
        model.body_simple[self._block] = 0
        dof = model.body_dofadr[self._block]
        model.dof_simplenum[dof:dof + model.body_dofnum[self._block]] = 0

        self.nominal = {
            "block_mass": self._mass0,
            "block_friction": float(self._pair_friction0[0, 0]),
            "pusher_friction": float(self._pair_friction0[1, 0]),
            "com_x": 0.0,
            "com_y": 0.0,
            "contact_timeconst": float(self._pair_solref0[0, 0]),
            "contact_dampratio": float(self._pair_solref0[0, 1]),
            "obs_pos_std": 0.0,
            "obs_yaw_std": 0.0,
            "obs_vel_std": 0.0,
        }
        self._validate(self.config)

    def set_config(self, config):
        config = PerturbationConfig.coerce(config)
        self._validate(config)
        self.config = config

    def _validate(self, config):
        for axis, half in zip(("com_x", "com_y"), self._block_half):
            spec = getattr(config, axis)
            if spec is not None and max(abs(b) for b in _spec_bounds(spec)) >= half:
                raise ValueError(f"{axis}={spec} puts the COM outside the block (half size {half:.3f} m)")
        spec = config.contact_timeconst
        if spec is not None and _spec_bounds(spec)[0] < 2 * self._timestep:
            raise ValueError(
                f"contact_timeconst={spec} is below 2*timestep={2 * self._timestep}; "
                "MuJoCo would clamp it"
            )

    def sample(self, rng, fixed=None):
        """One episode's parameters. ``fixed`` pins individual values for this call only."""
        fixed = fixed or {}
        unknown = set(fixed) - set(self.nominal)
        if unknown:
            raise ValueError(f"Unknown perturbation keys: {sorted(unknown)}")
        params = {}
        for key in PHYSICS_KEYS + NOISE_KEYS:  # fixed order keeps sampling reproducible
            spec = getattr(self.config, key)
            if key in fixed:
                params[key] = float(fixed[key])
            elif spec is None:
                params[key] = self.nominal[key]
            elif isinstance(spec, tuple):
                params[key] = float(rng.uniform(*spec))
            else:
                params[key] = spec
        return params

    def apply(self, params, data):
        """Write physics params into the model. Clobbers data, so call before resetting it."""
        m = self.model
        b = self._block

        scale = params["block_mass"] / self._mass0
        m.body_mass[b] = params["block_mass"]
        m.body_inertia[b] = self._inertia0 * scale
        m.body_ipos[b] = self._ipos0 + np.array([params["com_x"], params["com_y"], 0.0])

        m.pair_friction[self._pair_ids] = self._pair_friction0
        m.pair_friction[self._table_pair, 0:2] = params["block_friction"]
        m.pair_friction[self._pusher_pair, 0:2] = params["pusher_friction"]

        m.pair_solref[self._pair_ids, 0] = params["contact_timeconst"]
        m.pair_solref[self._pair_ids, 1] = params["contact_dampratio"]

        # Recompute derived constants (subtree masses, invweights) for the new body.
        mujoco.mj_setConst(m, data)
