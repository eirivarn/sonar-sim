from __future__ import annotations
import numpy as np

def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation matrix from roll/pitch/yaw (rad). Convention: R = Rz(yaw)*Ry(pitch)*Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr,  cr]], dtype=float)
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]], dtype=float)
    return Rz @ Ry @ Rx

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 0:
        return v
    return v / n
