# Residual Push

MuJoCo simulation and control code for pushing a block with a Franka Emika
Panda robot. The project includes a Cartesian impedance controller, a
Gymnasium pushing environment, and PPO training and evaluation scripts.

## Project layout

```text
assets/
  meshes/       Panda visual and collision meshes
  models/       MuJoCo XML models and the upstream Panda model documentation
configs/        Controller configuration
controllers/    Cartesian impedance and pushing controllers
envs/           Gymnasium pushing environment
rl/             PPO training and evaluation
scripts/        Interactive demos and controller analysis
tests/          Automated unit tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Interactive scripts require a desktop session with MuJoCo's viewer support.
On macOS, use `mjpython` for scripts that use the passive viewer.

```bash
python scripts/basic.py
mjpython scripts/cartesian_impedance_demo.py
mjpython scripts/eval_classical_push.py
```

Train PPO (this can take a long time):

```bash
python rl/train_ppo.py
```

Evaluate the saved best checkpoint after training:

```bash
mjpython scripts/eval_ppo.py
```

Run the controller analysis and optionally write plots:

```bash
python scripts/cartesian_impedance_analysis.py \
  --config configs/cartesian_controller.yaml \
  --plots plots
```

Run the automated tests:

```bash
python -m pytest -q
```

## Licensing

The MuJoCo engine and the Franka Emika Panda MJCF model (including its hand and
gripper meshes) are licensed under Apache-2.0. See
[Third-party notices](THIRD_PARTY_NOTICES.md) and the model's included
[license](assets/models/LICENSE).

No project-wide license has been selected for the original code in this
repository. The third-party licenses above apply to their respective
components, not automatically to the project's original code.