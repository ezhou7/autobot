"""Sign checks for ArduPilot (NED / FRD) <-> policy (z up / FLU) frames.

A flipped sign here flips the drone, so each axis is pinned with a
physical example.
"""
import numpy as np
import pytest

from autobot.mavlink.frames import (
    ModelFrame, gyro_frd_to_flu, quat_to_rot, rates_flu_to_frd)


def quat_from_euler_ned(roll, pitch, yaw):
    """ArduPilot convention: ZYX Euler, body FRD -> NED, (w, x, y, z)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def test_level_north_is_identity():
    f = ModelFrame.at(np.zeros(3), quat_to_rot([1, 0, 0, 0]))
    np.testing.assert_allclose(f.rotation(quat_to_rot([1, 0, 0, 0])), np.eye(3), atol=1e-12)
    assert f.heading == pytest.approx(0.0)


def test_position_axes_facing_north():
    f = ModelFrame.at(np.zeros(3), quat_to_rot([1, 0, 0, 0]))
    np.testing.assert_allclose(f.position([1, 0, 0]), [1, 0, 0])    # north = forward
    np.testing.assert_allclose(f.position([0, 1, 0]), [0, -1, 0])   # east = right
    np.testing.assert_allclose(f.position([0, 0, -1]), [0, 0, 1])   # up


def test_hand_over_heading_becomes_forward():
    # Drone facing east at hand-over (ArduPilot yaw +90 deg).
    q = quat_from_euler_ned(0, 0, np.radians(90))
    f = ModelFrame.at([5.0, -3.0, -2.0], quat_to_rot(q))
    np.testing.assert_allclose(f.position([5.0, -2.0, -2.0]), [1, 0, 0], atol=1e-9)   # 1 m east = forward
    np.testing.assert_allclose(f.position([6.0, -3.0, -2.0]), [0, 1, 0], atol=1e-9)   # 1 m north = left
    np.testing.assert_allclose(f.rotation(quat_to_rot(q)), np.eye(3), atol=1e-9)      # heading 0
    np.testing.assert_allclose(f.vector([0, 2.0, 0]), [2, 0, 0], atol=1e-9)


def test_nose_up_is_negative_policy_pitch():
    # ArduPilot +pitch = nose up. In the policy's FLU frame the nose
    # (body x) then points up: R[2, 0] > 0, which is a negative pitch.
    f = ModelFrame.at(np.zeros(3), quat_to_rot([1, 0, 0, 0]))
    R = f.rotation(quat_to_rot(quat_from_euler_ned(0, np.radians(10), 0)))
    assert R[2, 0] > 0.1


def test_right_bank_lowers_the_right_side():
    f = ModelFrame.at(np.zeros(3), quat_to_rot([1, 0, 0, 0]))
    R = f.rotation(quat_to_rot(quat_from_euler_ned(np.radians(10), 0, 0)))
    # body +y is LEFT in FLU; banking right raises the left side.
    assert R[2, 1] > 0.1


def test_rates():
    # Gyro: pitching nose up (FRD +q) is -pitch in FLU; yawing right
    # (FRD +r) is -yaw in FLU; roll right is + in both.
    np.testing.assert_allclose(gyro_frd_to_flu([0.5, 0.4, 0.3]), [0.5, -0.4, -0.3])
    # Commands: policy "pitch nose down" (FLU +) -> FRD negative.
    np.testing.assert_allclose(rates_flu_to_frd([0.2, 1.0, -1.0]), [0.2, -1.0, 1.0])
