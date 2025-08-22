#!/usr/bin/env python3
"""
MuJoCo offscreen recorder (macOS-safe) that uses the wrist camera 'ee_cam'.
- Offscreen GL context (no window)
- Proper OFFSCREEN buffer selection + resize
- Scene updated each frame with mjCAT_ALL
- Saves H.264 MP4 via imageio/ffmpeg
"""

import os
# Important: set before importing mujoco on macOS
os.environ.setdefault("MUJOCO_GL", "glfw")

import time
import numpy as np
import imageio
import mujoco


def main():
    xml_path = "manipulator_scene.xml"

    # --- Load model & data ---
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # Pose (so the arm is in-frame even before motion)
    d.qpos[:] = [0.0, 0.3, 0.5, 0.0]
    mujoco.mj_forward(m, d)

    # --- Video params ---
    W, H = 640, 480
    FPS = 30
    DURATION_S = 5
    N = FPS * DURATION_S
    out_mp4 = "ee_cam_recording.mp4"

    # --- Offscreen GL context ---
    ctx = mujoco.GLContext(W, H)
    ctx.make_current()

    # --- Visualization objects (create ONCE) ---
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)

    # Resolve wrist camera by name (do NOT hardcode ids)
    cam_name = "ee_cam"
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id == -1:
        raise RuntimeError(
            f"Camera '{cam_name}' not found. Available cameras: "
            f"{[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]}"
        )
    print(f"Using camera '{cam_name}' with id {cam_id}")
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    cam.fixedcamid = cam_id

    # Scene + rendering context
    scn = mujoco.MjvScene(m, maxgeom=1_000_000)
    rcon = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)

    # --- Select OFFSCREEN buffer and size it ---
    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, rcon)
    mujoco.mjr_resizeOffscreen(W, H, rcon)
    rect = mujoco.MjrRect(0, 0, W, H)

    # --- Frame capture loop ---
    frames = []
    rgb = np.empty((H, W, 3), dtype=np.uint8)
    t0 = time.time()

    for i in range(N):
        # simple periodic motion so the view changes
        t = i / N
        ang = 2 * np.pi * t
        d.qpos[0] = 0.3 * np.sin(ang)
        d.qpos[1] = 0.4 + 0.2 * np.sin(2 * ang)
        d.qpos[2] = 0.5 + 0.2 * np.cos(ang)
        d.qpos[3] = 0.1 * np.sin(3 * ang)

        mujoco.mj_step(m, d)

        # Update → render → read
        mujoco.mjv_updateScene(
            m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn
        )
        mujoco.mjr_render(rect, scn, rcon)
        mujoco.mjr_readPixels(rgb, None, rect, rcon)

        # Flip vertically (OpenGL origin is bottom-left)
        frames.append(np.flipud(rgb.copy()))

        if i == 0:
            print("First frame stats:",
                  "min", rgb.min(), "max", rgb.max(), "mean", float(rgb.mean()))
        if i % FPS == 0:
            print(f"{i}/{N} frames, elapsed {time.time()-t0:.1f}s")

    # --- Encode MP4 ---
    imageio.mimwrite(
        out_mp4, frames, fps=FPS, quality=8, codec="libx264", pixelformat="yuv420p"
    )
    print(f"Saved {out_mp4} ({len(frames)} frames at {FPS} FPS, {W}x{H})")

    # --- Cleanup ---
    rcon.free()
    ctx.free()


if __name__ == "__main__":
    main()
