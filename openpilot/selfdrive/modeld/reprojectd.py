#!/usr/bin/env python3
"""Fits this unit's narrow->wide camera rotation for the 3X->comma 4 reprojection stage: a one-off, from frame pairs taken
under calibrationd's own conditions (straight road, above 15 mph), about one frame a second until the estimate converges.
The tables are built here and the result lands in rotation.json; modeld swaps it in (calibrationd holds until then, so the
car cannot be engaged around the swap); afterwards this process only watches for a recalibration (device remounted),
which starts a new fit. Numpy only, low priority, one frame pair every few seconds."""
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

MIN_N, MAX_N = 6, 40  # frames: stop when the mean has converged (SE below SE_STOP) or at MAX_N
SE_STOP = 0.02        # deg, standard error of the trimmed mean's pitch/yaw
MIN_MATCHES = 15
MAX_RMS_DEG = 0.25    # per-frame residual after trimming (good frames 0.10-0.17 on the 3X)
PROGRESS_FILE = '/data/reproject_c4/fit.json'
C4_CAM = (1344, 760)
CACHE_DIR = os.environ.get('XDG_CACHE_HOME', '/data/tgcache')
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
  clients = {'narrow': VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_NARROW_ROAD, True),
             'wide': VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, True)}
  applied = tuple(RC.read_applied().get('rotvec') or RC.load_rotation()); calib = RC.calib_from_rotvec(applied)  # modeld's, once it has started
  state = RC.read_rotation_file()
  fits: list[tuple] = []; mean = applied; prev_cal = None
  cloudlog.warning(f"reprojectd: applied rotation {np.degrees(applied).round(3)} deg, {'fitted' if state.get('fitted') else 'not fitted yet'}")
  write_progress(n=0, of=MAX_N, fitted=bool(state.get('fitted')))
  while True:
    sm.update(100)
    if not all(c.is_connected() for c in clients.values()):
      for c in clients.values():
        c.connect(False)
      continue
    if not sm.all_checks(['extrinsicsCalibration']):
      continue
    cal = sm['extrinsicsCalibration'].calStatus
    if prev_cal is None:
      prev_cal = cal
    if state.get('fitted'):
      # only a calibration that was complete and got reset (user, or calibrationd's own mount check) means a refit; the
      # reset calibrationd does for our own swap must not, or fit -> swap -> reset -> refit loops forever
      if prev_cal == Status.calibrated and cal in (Status.uncalibrated, Status.recalibrating):
        cloudlog.warning("reprojectd: calibration was reset: refitting the rotation")
        state = {}; fits = []; write_progress(n=0, of=MAX_N, fitted=False)
      else:
        prev_cal = cal
        time.sleep(0.5)
        continue
    prev_cal = cal
    now = time.monotonic()
    straight_and_fast = (sm.all_checks(['carState', 'cameraOdometry']) and sm['carState'].vEgo > MIN_SPEED_FILTER
                         and abs(sm['cameraOdometry'].rot[2]) < MAX_YAW_RATE_FILTER)
    if not straight_and_fast:
      continue
    bn = clients['narrow'].recv(50); bw = clients['wide'].recv(50)
    if bn is None or bw is None or clients['narrow'].frame_id != clients['wide'].frame_id:
      continue
    narrow_y, wide_y = luma(bn), luma(bw)
    t0 = time.monotonic()
    # continuous: the first frame from the applied rotation with the coarse pass, later frames refined from the running
    # mean (a quarter of the work, ~1 frame/s on the 3X), until the mean has converged
    r = RC.fit_rotation(narrow_y, wide_y, calib) if not fits else RC.fit_rotation(narrow_y, wide_y, RC.calib_from_rotvec(mean), iters=1, coarse=False)
    if r is None or r[1] < MIN_MATCHES or r[2] > MAX_RMS_DEG:
      cloudlog.warning(f"reprojectd: frame {clients['narrow'].frame_id} rejected ({'no fit' if r is None else f'{r[1]} matches, rms {r[2]:.3f} deg'}), {time.monotonic() - t0:.1f} s")
      continue
    fits.append(r)
    mean, keep, spread, se = RC.combine_fits([f[0] for f in fits])
    cloudlog.warning(f"reprojectd: fit {len(fits)}: {np.degrees(r[0]).round(3)} deg, {r[1]} matches, rms {r[2]:.3f} deg, {time.monotonic() - t0:.1f} s; "
                     f"mean {np.degrees(mean).round(3)} spread {spread:.3f} se {se:.3f} deg")
    write_progress(n=len(fits), of=MAX_N, fitted=False, last_deg=[round(float(x), 3) for x in np.degrees(r[0])], rms=round(r[2], 3),
                   mean_deg=[round(float(x), 3) for x in np.degrees(mean)], se_deg=round(se, 3))
    if not (len(fits) >= MIN_N and se < SE_STOP) and len(fits) < MAX_N:
      continue
    final = np.array(mean)
    # build the tables here (~6 s of numpy at low priority) so modeld's swap is just a load; then publish the fit
    t0 = time.monotonic(); write_progress(n=len(fits), of=MAX_N, fitted=False, building=True, deg=[round(float(x), 3) for x in np.degrees(final)])
    RC.load_tables((bn.width, bn.height), C4_CAM, CACHE_DIR, RC.calib_from_rotvec(final))
    cloudlog.warning(f"reprojectd: tables built in {time.monotonic() - t0:.0f} s")
    RC.save_rotation(final, fitted=True, n=int(len(keep)), spread_deg=round(spread, 3), se_deg=round(se, 3), applied=[float(v) for v in applied])
    state = RC.read_rotation_file(); fits = []; applied = tuple(float(v) for v in final); calib = RC.calib_from_rotvec(applied)
    write_progress(n=len(keep), of=len(keep), fitted=True, deg=[round(float(x), 3) for x in np.degrees(final)], spread_deg=round(spread, 3), se_deg=round(se, 3))
    cloudlog.warning(f"reprojectd: rotation fitted {np.degrees(final).round(3)} deg (was {np.degrees(applied).round(3)}, spread {spread:.3f} deg)")


if __name__ == "__main__":
  main()
