from pathlib import Path

import pytest
import numpy as np
import mujoco

from controllers.cartesian_impedance import CartesianImpedance


def make_controller():
    ROOT = Path(__file__).resolve().parent.parent

    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "assets" / "models" / "scene.xml")
    )
    data = mujoco.MjData(model)

    arm_dofs = [
        model.joint(f"joint{i}").dofadr[0]
        for i in range(1, 8)
    ]
    arm_qpos = [
        model.joint(f"joint{i}").qposadr[0]
        for i in range(1, 8)
    ]

    return CartesianImpedance(
        model=model,
        data=data,
        site_name="attachment_site",
        arm_dof_ids=arm_dofs,
        arm_qpos=arm_qpos,
        Kp=5.0,
        Kd=1.0,
        Kr=5.0,
        Dr=1.0,
        Kn=5.0,
    )


@pytest.fixture
def ci():
    return make_controller()

def Rz(theta):
    """Rotation about world z-axis."""
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ])


def Ry(theta):
    """Rotation about y-axis."""
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c],
    ])


def assert_close(actual, expected, tol=1e-6):
    assert np.allclose(actual, expected, atol=tol), (
        f"\nExpected: {expected}\n"
        f"Actual:   {actual}"
    )


def test_identity(ci):
    """R_des = R should give zero orientation error."""
    R = Rz(np.deg2rad(20))

    ci.data.site_xmat[ci.site_id] = R.flatten()

    error = ci.orientation_error(R)

    assert_close(error, np.zeros(3))


def test_world_frame_rotation(ci):
    """
    R_des = Rz(30°) @ R should produce
    [0, 0, 30°].

    R_des = R @ Rz(30°) represents a body-frame rotation,
    so its world-frame rotation vector should generally differ.
    """
    R = Ry(np.deg2rad(45))
    ci.data.site_xmat[ci.site_id] = R.flatten()

    theta = np.deg2rad(30)

    # World-frame rotation
    R_des_world = Rz(theta) @ R
    error_world = ci.orientation_error(R_des_world)

    assert_close(
        error_world,
        np.array([0, 0, theta])
    )

    # Body-frame rotation
    R_des_body = R @ Rz(theta)
    error_body = ci.orientation_error(R_des_body)

    expected_body = R @ np.array([0, 0, theta])

    assert_close(
        error_body,
        expected_body
    )

    # They should be different because R is not identity.
    assert not np.allclose(error_world, error_body)


def test_sign_and_double_cover(ci):
    """
    +270° is equivalent to -90°.

    The quaternion implementation should choose the shortest
    rotation and return -90° about z.
    """
    R = np.eye(3)
    ci.data.site_xmat[ci.site_id] = R.flatten()

    theta = np.deg2rad(270)

    R_des = Rz(theta)

    error = ci.orientation_error(R_des)

    expected = np.array([
        0,
        0,
        -np.pi / 2
    ])

    assert_close(error, expected)


def test_near_180_degrees(ci):
    """
    A 170° rotation should produce an error with magnitude
    close to 170° rather than collapsing toward zero.
    """
    R = np.eye(3)
    ci.data.site_xmat[ci.site_id] = R.flatten()

    theta = np.deg2rad(170)

    R_des = Rz(theta)

    error = ci.orientation_error(R_des)

    assert np.isclose(
        np.linalg.norm(error),
        theta,
        atol=1e-6
    )

    # Also check that the direction is correct.
    assert_close(
        error,
        np.array([0, 0, theta])
    )


if __name__ == "__main__":
    ci = make_controller()

    test_identity(ci)
    test_world_frame_rotation(ci)
    test_sign_and_double_cover(ci)
    test_near_180_degrees(ci)

    print("All orientation error tests passed.")