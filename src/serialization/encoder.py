"""Protobuf encoder — converts internal Python objects to protobuf binary.

Falls back to a manual dict-based encoding if the generated protobuf
modules are not available (e.g., protoc hasn't been run yet).
This ensures the server can start even without a proto compile step.
"""

from __future__ import annotations

import logging
import struct
from typing import Any

import numpy as np

logger = logging.getLogger("tars.serialization")

# Try importing generated protobuf modules
_PROTO_AVAILABLE = False
try:
    from serialization.proto import trajectory_pb2 as pb
    _PROTO_AVAILABLE = True
    logger.info("Protobuf generated modules loaded")
except ImportError:
    logger.warning("Protobuf modules not found — using manual binary encoding")


# --------------------------------------------------------------------------- #
#  Protobuf-based encoding (preferred)
# --------------------------------------------------------------------------- #

def encode_planetary_snapshot(
    epoch_jd: float,
    bodies: list[dict],
) -> bytes:
    """Encode a planetary position snapshot to protobuf binary.

    bodies: list of {"body_id": int, "name": str, "position": (x, y, z)}
    """
    if _PROTO_AVAILABLE:
        snap = pb.PlanetarySnapshot()
        snap.epoch_jd = epoch_jd
        for b in bodies:
            state = snap.bodies.add()
            state.body_id = b["body_id"]
            state.name = b["name"]
            state.position.x = b["position"][0]
            state.position.y = b["position"][1]
            state.position.z = b["position"][2]
            state.epoch_jd = epoch_jd
        return snap.SerializeToString()

    # Fallback: custom compact binary
    return _encode_snapshot_binary(epoch_jd, bodies)


def encode_trajectory(
    segments: list[dict],
    total_dv: float,
    departure_jd: float,
    arrival_jd: float,
    tof_days: float,
) -> bytes:
    """Encode a trajectory to protobuf binary.

    segments: list of {
        "type": int (0=cruise, 1=departure, 2=arrival, 3=gravity_assist),
        "points": np.ndarray shape (N, 3) in scene units,
        "epochs": np.ndarray shape (N,),
        "origin_body_id": int,
        "target_body_id": int,
    }
    """
    if _PROTO_AVAILABLE:
        traj = pb.Trajectory()
        traj.total_dv = total_dv
        traj.departure_epoch_jd = departure_jd
        traj.arrival_epoch_jd = arrival_jd
        traj.tof_days = tof_days

        for seg_data in segments:
            seg = traj.segments.add()
            seg.type = seg_data["type"]
            seg.origin_body_id = seg_data.get("origin_body_id", 0)
            seg.target_body_id = seg_data.get("target_body_id", 0)

            points = seg_data["points"]
            epochs = seg_data["epochs"]
            for i in range(len(points)):
                pt = seg.points.add()
                pt.position.x = float(points[i, 0])
                pt.position.y = float(points[i, 1])
                pt.position.z = float(points[i, 2])
                pt.epoch_jd = float(epochs[i])

        return traj.SerializeToString()

    return _encode_trajectory_binary(segments, total_dv, departure_jd, arrival_jd, tof_days)


def encode_optimization_update(
    job_id: str,
    status: str,
    progress: dict,
    trajectory_bytes: bytes | None = None,
) -> bytes:
    """Encode an optimization progress update to protobuf binary."""
    if _PROTO_AVAILABLE:
        update = pb.OptimizationUpdate()
        update.job_id = job_id
        update.status = status
        update.iteration = progress.get("iteration", 0)
        update.max_iterations = progress.get("max_iterations", 0)
        update.best_dv_total = progress.get("best_dv_total", 0.0) or 0.0
        update.best_departure_jd = progress.get("best_departure_jd", 0.0)
        update.best_tof_days = progress.get("best_tof_days", 0.0)
        update.best_dv_departure = progress.get("best_dv_departure", 0.0) or 0.0
        update.best_dv_arrival = progress.get("best_dv_arrival", 0.0) or 0.0
        update.converged = progress.get("converged", False)
        return update.SerializeToString()

    # Fallback: just use JSON bytes
    import json
    return json.dumps({"job_id": job_id, "status": status, **progress}).encode()


def encode_porkchop(
    departure_jds: np.ndarray,
    tof_days: np.ndarray,
    dv_grid: np.ndarray,
) -> bytes:
    """Encode pork-chop plot data to protobuf binary."""
    if _PROTO_AVAILABLE:
        pc = pb.PorkChopData()
        pc.departure_jds.extend(departure_jds.tolist())
        pc.tof_days.extend(tof_days.tolist())
        # Flatten grid row-major, replace inf with -1
        flat = dv_grid.flatten()
        flat = np.where(np.isinf(flat), -1.0, flat)
        pc.dv_values.extend(flat.tolist())
        pc.dep_steps = len(departure_jds)
        pc.tof_steps = len(tof_days)
        return pc.SerializeToString()

    return _encode_porkchop_binary(departure_jds, tof_days, dv_grid)


# --------------------------------------------------------------------------- #
#  Fallback binary encoding (compact, no protobuf dependency)
#  Format: [header][data]
#  These can be read directly into Float64Array on the frontend.
# --------------------------------------------------------------------------- #

def _encode_snapshot_binary(epoch_jd: float, bodies: list[dict]) -> bytes:
    """Pack snapshot as: [epoch_jd:f64][n_bodies:u32][(body_id:i32, x:f64, y:f64, z:f64)*n]"""
    n = len(bodies)
    buf = struct.pack("<dI", epoch_jd, n)
    for b in bodies:
        pos = b["position"]
        buf += struct.pack("<iddd", b["body_id"], pos[0], pos[1], pos[2])
    return buf


def _encode_trajectory_binary(
    segments: list[dict], total_dv: float,
    departure_jd: float, arrival_jd: float, tof_days: float,
) -> bytes:
    """Pack trajectory as header + per-segment point arrays."""
    n_seg = len(segments)
    buf = struct.pack("<ddddi", total_dv, departure_jd, arrival_jd, tof_days, n_seg)

    for seg in segments:
        points = seg["points"]
        epochs = seg["epochs"]
        n_pts = len(points)
        buf += struct.pack("<iiiI", seg["type"], seg.get("origin_body_id", 0),
                           seg.get("target_body_id", 0), n_pts)
        # Pack points as contiguous float64 array
        buf += np.ascontiguousarray(points, dtype=np.float64).tobytes()
        buf += np.ascontiguousarray(epochs, dtype=np.float64).tobytes()

    return buf


def _encode_porkchop_binary(
    departure_jds: np.ndarray, tof_days: np.ndarray, dv_grid: np.ndarray,
) -> bytes:
    """Pack porkchop as: [dep_steps:u32][tof_steps:u32][dep_jds][tof_days][dv_flat]"""
    m, n = len(departure_jds), len(tof_days)
    buf = struct.pack("<II", m, n)
    buf += np.ascontiguousarray(departure_jds, dtype=np.float64).tobytes()
    buf += np.ascontiguousarray(tof_days, dtype=np.float64).tobytes()
    flat = dv_grid.flatten()
    flat = np.where(np.isinf(flat), -1.0, flat)
    buf += np.ascontiguousarray(flat, dtype=np.float64).tobytes()
    return buf
