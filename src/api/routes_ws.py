"""WebSocket endpoints for real-time streaming.

- /ws/trajectory/{job_id}  — Stream optimization progress updates
- /ws/ephemeris/stream      — Stream planetary positions for animation
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ephemeris.bodies import ALL_BODIES, BODY_BY_ID
from ephemeris.spline_cache import EphemerisCache
from mechanics.transforms import iso_to_jd, jd_to_iso, positions_to_scene, km_to_scene
from optimizer.dispatcher import stream_progress
from serialization.encoder import encode_planetary_snapshot

logger = logging.getLogger("tars.ws")
router = APIRouter()


@router.websocket("/ws/trajectory/{job_id}")
async def ws_trajectory_stream(websocket: WebSocket, job_id: str):
    """Stream optimization progress for a job.

    The client connects after submitting a POST /optimize request.
    Messages are JSON dicts with optimization progress (iteration,
    best delta-v, departure date, etc.).

    When the job completes, a final message with status="complete" is sent
    and the connection is closed.
    """
    await websocket.accept()
    logger.info("WebSocket connected for job %s", job_id)

    try:
        async for progress in stream_progress(job_id):
            await websocket.send_json(progress)

            if progress.get("status") in ("complete", "failed"):
                break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for job %s", job_id)
    except Exception as e:
        logger.error("WebSocket error for job %s: %s", job_id, e)
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/ws/ephemeris/stream")
async def ws_ephemeris_stream(websocket: WebSocket):
    """Stream planetary positions for real-time 3D animation.

    Protocol:
    1. Client sends a JSON config message:
       {
         "body_ids": [399, 499, ...],     // NAIF IDs to track (optional, default=all planets)
         "start_jd": 2460000.0,           // start epoch (optional)
         "speed": 1.0,                    // days per second of real time (optional, default=1)
         "fps": 30,                       // target frames per second (optional, default=30)
         "scene_units": true,             // convert to scene units (optional, default=true)
         "binary": false                  // use binary encoding (optional, default=false)
       }
    2. Server streams position snapshots at the requested FPS.
    3. Client can send "pause", "resume", "stop", or a new config to change params.
    """
    await websocket.accept()
    logger.info("Ephemeris stream WebSocket connected")

    cache: EphemerisCache = websocket.app.state.ephemeris

    # Defaults
    body_ids: list[int] = cache.available_bodies()
    current_jd: float = iso_to_jd("2026-01-01")
    speed: float = 1.0  # days per real-time second
    fps: int = 30
    scene_units: bool = True
    use_binary: bool = False
    paused: bool = False

    try:
        # Wait for initial config
        try:
            config_raw = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            config = json.loads(config_raw)

            body_ids = config.get("body_ids", body_ids)
            current_jd = config.get("start_jd", current_jd)
            speed = config.get("speed", speed)
            fps = min(60, max(1, config.get("fps", fps)))
            scene_units = config.get("scene_units", scene_units)
            use_binary = config.get("binary", use_binary)

        except asyncio.TimeoutError:
            # Use defaults if no config received within 5s
            pass

        dt_per_frame = speed / fps  # days per frame
        frame_interval = 1.0 / fps  # seconds between frames

        while True:
            # Check for control messages (non-blocking)
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                msg = msg.strip().lower()

                if msg == "pause":
                    paused = True
                    continue
                elif msg == "resume":
                    paused = False
                    continue
                elif msg == "stop":
                    break
                else:
                    # Try to parse as new config
                    try:
                        new_config = json.loads(msg)
                        body_ids = new_config.get("body_ids", body_ids)
                        speed = new_config.get("speed", speed)
                        fps = min(60, max(1, new_config.get("fps", fps)))
                        scene_units = new_config.get("scene_units", scene_units)
                        use_binary = new_config.get("binary", use_binary)

                        if "start_jd" in new_config:
                            current_jd = new_config["start_jd"]

                        dt_per_frame = speed / fps
                        frame_interval = 1.0 / fps
                    except (json.JSONDecodeError, TypeError):
                        pass

            except asyncio.TimeoutError:
                pass

            if paused:
                await asyncio.sleep(frame_interval)
                continue

            # Build snapshot
            bodies_data = []
            for bid in body_ids:
                if bid not in cache:
                    continue
                body_info = BODY_BY_ID.get(bid)
                if body_info is None:
                    continue

                pos = cache.get_position(bid, current_jd)
                if scene_units:
                    pos = km_to_scene(pos)

                bodies_data.append({
                    "body_id": bid,
                    "name": body_info.name,
                    "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                })

            # Send snapshot
            if use_binary:
                data = encode_planetary_snapshot(current_jd, bodies_data)
                await websocket.send_bytes(data)
            else:
                snapshot = {
                    "epoch_jd": current_jd,
                    "epoch_iso": jd_to_iso(current_jd),
                    "bodies": bodies_data,
                }
                await websocket.send_json(snapshot)

            # Advance time
            current_jd += dt_per_frame
            await asyncio.sleep(frame_interval)

    except WebSocketDisconnect:
        logger.info("Ephemeris stream WebSocket disconnected")
    except Exception as e:
        logger.error("Ephemeris stream error: %s", e)
        try:
            await websocket.send_json({"status": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
