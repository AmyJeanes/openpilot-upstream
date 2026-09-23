#!/usr/bin/env python3
"""The 3X's cameras as a comma 4's. Reprojects camerad's narrow and wide frames into the comma 4 geometry on the QCOM GPU
(reproject_c4: the wide as the surround, the narrow as the sharp inset) and serves the pair as VisionIPC streams "reproject"
(narrow = the composite, wide), with camerad's frame ids and timestamps, so modeld runs comma's big model on them as it would
on a mici and the ui can show them (ReprojectionView). The seam exposure meter lives here, and a rotation reprojectcalibd
publishes as ready is swapped in here. It publishes nothing: reprojectCalibration carries the rotation, and the stage's timing
goes to the log as a summary a minute (reprojectd.stage)."""
import os
os.environ.setdefault('QCOM_PRIORITY', '1')  # KGSL context priority: the stage preempts the driver-monitoring model's 20 ms
os.environ.setdefault('DEV', 'QCOM')  # tinygrad's default device: never probe the eGPU, modeld owns it (its USB lock)
import threading
import time

import numpy as np
from tinygrad import Tensor, Device

from openpilot.cereal import messaging, custom
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient, VisionIpcServer
from openpilot.common.realtime import config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld import reproject_c4 as RC
from openpilot.selfdrive.modeld.reproject_c4.kernel import Reprojector
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

C4_CAM = RC.C4_CAM
NARROW, WIDE = VisionStreamType.VISION_STREAM_NARROW_ROAD, VisionStreamType.VISION_STREAM_WIDE_ROAD


class Stage:
  def __init__(self, src_wh: tuple[int, int]):
    self.src_wh, self.cache_dir = src_wh, RC.table_cache_dir()
    self.src_size = get_nv12_info(*src_wh)[3]
    self.rotation = RC.load_rotation()
    self.fitted = bool(RC.read_rotation())
    cloudlog.warning(f"reprojectd: rotation {np.degrees(self.rotation).round(3)} deg, "
                     + ('fitted' if self.fitted else 'seeded from CalibrationParams / the fleet median'))
    self.calib = RC.calib_from_rotvec(self.rotation)
    T = RC.load_tables(src_wh, C4_CAM, self.cache_dir, self.calib)
    self.rp = Reprojector(T, C4_CAM, 'QCOM')
    self.meter = RC.SeamMeter(T["meter"])
    self.pending = None
    self.loader: threading.Thread | None = None
    self.saver: threading.Thread | None = None
    self._src_tensors: dict[int, Tensor] = {}
    self.out = np.zeros((2, self.rp.body), np.uint8)  # composite, wide: the kernel writes here, the server copies into its ring
    self.rp.bind(self.out[1], self.out[0])
    self.time = 0.0

  def src_tensor(self, buf) -> Tensor:
    ptr = np.frombuffer(buf.data, dtype=np.uint8).ctypes.data
    if ptr not in self._src_tensors:  # one mapping per VisionIPC ring slot
      self._src_tensors[ptr] = Tensor.from_blob(ptr, (self.src_size,), dtype='uint8', device='QCOM')
    return self._src_tensors[ptr]

  def run(self, wide, narrow, match: np.ndarray) -> None:
    t0 = time.perf_counter()
    self.rp(self.src_tensor(wide), self.src_tensor(narrow), match)
    Device['QCOM'].synchronize()
    self.time = time.perf_counter() - t0

  def follow_fit(self, fit) -> None:
    """A rotation reprojectcalibd has published as ready (its tables already built) is loaded in a thread and swapped in
    between frames (one upload), then kept in ReprojectRotation: that is what tells reprojectcalibd it is applied, so it
    is saved again while reprojectcalibd still says ready. calibrationd holds meanwhile, so the car cannot be engaged
    around the swap."""
    if self.pending is not None:
      T, calib, meter, rot = self.pending
      self.pending = None
      self.loader = None
      t0 = time.perf_counter()
      self.rp.reload(T)
      self.calib, self.meter, self.rotation, self.fitted = calib, meter, rot, True
      cloudlog.warning(f"reprojectd: fitted rotation {np.degrees(rot).round(3)} deg swapped in ({(time.perf_counter() - t0) * 1e3:.0f} ms)")
      return
    if self.loader is not None or fit is None or fit.status != custom.ReprojectCalibration.Status.ready or len(fit.target) != 3:
      return
    rot = tuple(float(v) for v in fit.target)
    if rot == self.rotation:
      if self.saver is None or not self.saver.is_alive():
        def save():  # a param write, off the frame loop
          os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))  # inherited SCHED_FIFO otherwise
          RC.save_rotation(rot)
        self.saver = threading.Thread(target=save, daemon=True)
        self.saver.start()
      return
    calib = RC.calib_from_rotvec(rot)
    def load():  # one 19 MB read off the frame loop (a build, if the fitter's tables are gone)
      os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))
      T = RC.load_tables(self.src_wh, C4_CAM, self.cache_dir, calib)
      self.pending = (T, calib, RC.SeamMeter(T["meter"]), rot)
    self.loader = threading.Thread(target=load, daemon=True)
    self.loader.start()


def main():
  sm = messaging.SubMaster(['narrowRoadCameraState', 'wideRoadCameraState', 'reprojectCalibration'])
  narrow = VisionIpcClient("camerad", NARROW, True)
  wide = VisionIpcClient("camerad", WIDE, False)
  while not narrow.connect(False):
    time.sleep(0.1)
  while not wide.connect(False):
    time.sleep(0.1)
  src_wh = (narrow.width, narrow.height)
  cloudlog.warning(f"reprojectd: cameras {src_wh[0]}x{src_wh[1]} -> comma 4 {C4_CAM[0]}x{C4_CAM[1]}")
  t0 = time.monotonic()
  stage = Stage(src_wh)
  blank = [Tensor.zeros(stage.src_size, dtype='uint8', device='QCOM').contiguous().realize() for _ in range(2)]
  for _ in range(3):  # jit capture before the first real frame
    stage.rp(blank[0], blank[1])
    Device['QCOM'].synchronize()
  stride, y_height, _, _ = get_nv12_info(*C4_CAM)
  server = VisionIpcServer("reproject")
  for tp in (NARROW, WIDE):
    server.create_buffers_with_sizes(tp, 4, C4_CAM[0], C4_CAM[1], stage.rp.body, stride, stride * y_height)
  server.start_listener()
  cloudlog.warning(f"reprojectd: serving after {time.monotonic() - t0:.1f} s")
  # real-time only from here: the table load and the jit capture above are seconds of CPU, and a real-time task holding a
  # core for a second trips the kernel's throttle, which this kernel turns into a panic (CONFIG_PANIC_ON_RT_THROTTLING)
  config_realtime_process(6, 53)

  n = 0
  stage_ms: list[float] = []
  summary_t = time.monotonic()
  while True:
    pair = RC.recv_pair(narrow, wide)
    if pair is None:
      continue
    buf_n, buf_w = pair
    sm.update(0)
    # match the wide surround to the narrow inset: measured in the seam ring of these frames, the sensors' exposure
    # settings as the fallback when the ring is unusable
    ncs, wcs = sm['narrowRoadCameraState'], sm['wideRoadCameraState']
    g = RC.exposure_gain(ncs.gain * ncs.integLines, wcs.gain * wcs.integLines) if sm.seen['narrowRoadCameraState'] and sm.seen['wideRoadCameraState'] else 1.0
    match = stage.meter.update(np.frombuffer(buf_w.data, dtype=np.uint8), np.frombuffer(buf_n.data, dtype=np.uint8), g)
    stage.run(buf_w, buf_n, match)
    server.send(NARROW, stage.out[0], narrow.frame_id, narrow.timestamp_sof, narrow.timestamp_eof)
    server.send(WIDE, stage.out[1], wide.frame_id, wide.timestamp_sof, wide.timestamp_eof)
    n += 1
    stage_ms.append(stage.time * 1e3)
    if time.monotonic() - summary_t >= 60.0:
      ms = np.array(stage_ms)
      cloudlog.event("reprojectd.stage", frames=len(ms), p50_ms=round(float(np.percentile(ms, 50)), 2),
                     p95_ms=round(float(np.percentile(ms, 95)), 2), max_ms=round(float(ms.max()), 2), over_8ms=int((ms > 8).sum()))
      stage_ms.clear()
      summary_t = time.monotonic()
    if n % 20 == 0:
      stage.follow_fit(sm['reprojectCalibration'] if sm.seen['reprojectCalibration'] else None)


if __name__ == "__main__":
  main()
