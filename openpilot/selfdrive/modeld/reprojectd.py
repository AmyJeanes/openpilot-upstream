#!/usr/bin/env python3
"""Fits this unit's narrow->wide camera rotation for the 3X->comma 4 reprojection stage: a one-off, from frame pairs taken
under calibrationd's own conditions (straight road, above 15 mph) while it is calibrating, or until a first fit exists.
The result lands in rotation.json and modeld applies it at its next start; afterwards this process only watches for a
recalibration (device remounted), which starts a new fit. Numpy only, low priority, one frame pair every few seconds."""
import os
import time

import numpy as np

from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.locationd.calibrationd import MIN_SPEED_FILTER, MAX_YAW_RATE_FILTER
from openpilot.selfdrive.modeld import reproject_c4 as RC

N_FITS = 8            # frame pairs per fit (~25 s of straight driving, like calibrationd's 5 blocks)
PERIOD = 3.0          # s between frame pairs
MIN_MATCHES = 15
MAX_RMS_DEG = 0.25    # per-frame residual after trimming (good frames 0.10-0.17 on the 3X)
PROGRESS_FILE = '/data/reproject_c4/fit.json'
Status = log.ExtrinsicsCalibration.Status


def luma(buf) -> np.ndarray:
  return np.array(buf.data[:buf.uv_offset], dtype=np.uint8).reshape(-1, buf.stride)[:buf.height, :buf.width]


def write_progress(**kw) -> None:
  try:
    import json
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    tmp = PROGRESS_FILE + '.tmp'; json.dump(kw, open(tmp, 'w')); os.replace(tmp, PROGRESS_FILE)
  except OSError:
    pass


def main():
  os.nice(10)
  sm = messaging.SubMaster(['carState', 'extrinsicsCalibration', 'cameraOdometry'])
  clients = {'narrow': VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True),
             'wide': VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, True)}
  applied = RC.load_rotation(); calib = RC.calib_from_rotvec(applied)
  state = RC.read_rotation_file()
  fits: list[tuple] = []
  last = 0.0
  cloudlog.warning(f"reprojectd: applied rotation {np.degrees(applied).round(3)} deg, {'fitted' if state.get('fitted') else 'not fitted yet'}")
  write_progress(n=0, of=N_FITS, fitted=bool(state.get('fitted')))
  while True:
    sm.update(100)
    if not all(c.is_connected() for c in clients.values()):
      for c in clients.values():
        c.connect(False)
      continue
    if not sm.all_checks(['extrinsicsCalibration']):
      continue
    cal = sm['extrinsicsCalibration'].calStatus
    if state.get('fitted'):
      if cal in (Status.uncalibrated, Status.recalibrating) and sm.updated['extrinsicsCalibration']:
        cloudlog.warning("reprojectd: calibrationd is recalibrating: refitting the rotation")
        state = {}; fits = []; write_progress(n=0, of=N_FITS, fitted=False)
      else:
        time.sleep(0.5)
        continue
    now = time.monotonic()
    straight_and_fast = (sm.all_checks(['carState', 'cameraOdometry']) and sm['carState'].vEgo > MIN_SPEED_FILTER
                         and abs(sm['cameraOdometry'].rot[2]) < MAX_YAW_RATE_FILTER)
    if not straight_and_fast or now - last < PERIOD:
      continue
    last = now
    bn = clients['narrow'].recv(50); bw = clients['wide'].recv(50)
    if bn is None or bw is None or clients['narrow'].frame_id != clients['wide'].frame_id:
      continue
    narrow_y, wide_y = luma(bn), luma(bw)
    t0 = time.monotonic()
    r = RC.fit_rotation(narrow_y, wide_y, calib)
    if r is None or r[1] < MIN_MATCHES or r[2] > MAX_RMS_DEG:
      cloudlog.warning(f"reprojectd: frame {clients['narrow'].frame_id} rejected ({'no fit' if r is None else f'{r[1]} matches, rms {r[2]:.3f} deg'}), {time.monotonic() - t0:.1f} s")
      continue
    fits.append(r)
    cloudlog.warning(f"reprojectd: fit {len(fits)}/{N_FITS}: {np.degrees(r[0]).round(3)} deg, {r[1]} matches, rms {r[2]:.3f} deg, {time.monotonic() - t0:.1f} s")
    write_progress(n=len(fits), of=N_FITS, fitted=False, last_deg=[round(float(x), 3) for x in np.degrees(r[0])], rms=round(r[2], 3))
    if len(fits) < N_FITS:
      continue
    # drop the two frames farthest from the mean, then average the rest
    rv = [f[0] for f in fits]; m = RC.mean_rotvec(rv)
    dev = [np.linalg.norm(RC.matrix_to_rotvec(RC.rotvec_to_matrix(m).T @ RC.rotvec_to_matrix(v))) for v in rv]
    keep = np.argsort(dev)[:len(rv) - 2]
    final = RC.mean_rotvec([rv[i] for i in keep]); spread = float(np.degrees(max(dev[i] for i in keep)))
    RC.save_rotation(final, fitted=True, n=int(len(keep)), spread_deg=round(spread, 3), applied=[float(v) for v in applied])
    state = RC.read_rotation_file(); fits = []
    write_progress(n=N_FITS, of=N_FITS, fitted=True, deg=[round(float(x), 3) for x in np.degrees(final)], spread_deg=round(spread, 3))
    cloudlog.warning(f"reprojectd: rotation fitted {np.degrees(final).round(3)} deg (was {np.degrees(applied).round(3)}, spread {spread:.3f} deg): applies at the next modeld start")


if __name__ == "__main__":
  main()
