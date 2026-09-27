"""Robot-neutral chunk execution; importing this module never opens hardware.

A hardware adapter implements ``send_action`` in the explicitly declared space,
frame and units. Camera acquisition and robot-specific calibration are external.
"""
from dataclasses import dataclass
import math
import time
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class ActionCalibration:
    space: str
    coordinate_frame: str
    scale: list[float]
    offset: list[float]
    lower: list[float]
    upper: list[float]

    def convert(self, actions) -> np.ndarray:
        if self.space not in {'joint_positions_rad', 'cartesian_pose_xyzrpy_m_rad'}:
            raise ValueError('space must explicitly describe absolute joint or Cartesian commands')
        if not isinstance(self.coordinate_frame, str) or not self.coordinate_frame.strip():
            raise ValueError('coordinate_frame is required')
        # Six joint/pose coordinates plus a gripper opening in metres.
        scale, offset, lower, upper = [np.asarray(x, dtype=float) for x in
                                      (self.scale, self.offset, self.lower, self.upper)]
        if any(x.shape != (7,) or not np.isfinite(x).all() for x in (scale, offset, lower, upper)):
            raise ValueError('calibration requires seven finite values per vector')
        if np.any(lower >= upper) or np.any(scale == 0):
            raise ValueError('invalid calibration bounds or zero scale')
        value = np.asarray(actions, dtype=float)
        if value.ndim != 2 or value.shape[1] != 7 or not len(value) or not np.isfinite(value).all():
            raise ValueError('actions must be a nonempty finite (chunk_length, 7) array')
        result = value * scale + offset
        if not np.isfinite(result).all() or np.any(result < lower) or np.any(result > upper):
            raise ValueError('command exceeds calibrated bounds')
        return result


class RobotBackend(Protocol):
    def send_action(self, action: np.ndarray, *, space: str, coordinate_frame: str) -> None:
        """Send one absolute command; final coordinate is gripper opening in metres."""
        ...


def model_observation(full_rgb, wrist_rgb, state) -> dict:
    images = [np.asarray(full_rgb), np.asarray(wrist_rgb)]
    if any(x.ndim != 3 or x.shape[-1] != 3 or x.dtype != np.uint8 or x.size == 0 for x in images):
        raise ValueError('both model inputs must be nonempty uint8 RGB images')
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 1 or not state.size or not np.isfinite(state).all():
        raise ValueError('state must be a finite vector in the checkpoint state convention')
    return {'full_image': images[0].copy(), 'wrist_image': images[1].copy(), 'state': state.copy()}


def execute_chunk(backend: RobotBackend, actions, calibration: ActionCalibration, *,
                  control_hz: float, clock=time.monotonic, sleep=time.sleep) -> list[dict]:
    """Execute one validated chunk with absolute deadlines and command timestamps.

    A delta-action policy needs its own state-aware decoder before this function.
    Returned timestamps describe commands, not measured robot motion.
    """
    if not math.isfinite(control_hz) or control_hz <= 0:
        raise ValueError('control_hz must be positive and finite')
    commands = calibration.convert(actions)
    start = clock()
    trace = []
    for index, command in enumerate(commands):
        deadline = start + index / control_hz
        now = clock()
        if now < deadline:
            sleep(deadline - now)
        backend.send_action(command.copy(), space=calibration.space,
                            coordinate_frame=calibration.coordinate_frame)
        sent = clock()
        trace.append({'action_index': index, 'command': command.tolist(),
                      'scheduled_s': deadline - start, 'sent_s': sent - start,
                      'lateness_ms': max(0.0, sent - deadline) * 1000})
    return trace
