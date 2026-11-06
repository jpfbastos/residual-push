# Third-party notices

## MuJoCo

MuJoCo is distributed under the Apache License 2.0. This repository depends on
the MuJoCo Python package; it does not include a copy of the MuJoCo engine
source. See the [official MuJoCo license](https://github.com/google-deepmind/mujoco/blob/main/LICENSE).

## Franka Emika Panda model

The Panda MJCF model and its mesh assets, including the hand/gripper, are
derived from the [Franka Emika Panda model in MuJoCo
Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda).
That model is distributed under Apache-2.0. Its license text is included at
[assets/models/LICENSE](assets/models/LICENSE), and its upstream model
documentation is at [assets/models/README.md](assets/models/README.md).

The hand/gripper is part of this Panda model package; this repository does not
include a separate hand-only license. The model license concerns the model and
mesh files and does not grant rights to the physical robot or the Franka
trademarks.
