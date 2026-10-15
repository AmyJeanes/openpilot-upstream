"""reprojectd's stage on this PC's GPU."""
import os
os.environ.setdefault('DEV', 'CUDA')  # before tinygrad is imported
import logging
import threading

import numpy as np

from openpilot.selfdrive.modeld import reproject_c4 as RC
from openpilot.selfdrive.modeld.reproject_c4.tables import ALPHA_SHIFT

DEV = os.environ['DEV']

logger = logging.getLogger(__name__)


class Stage:
  """reprojectd's stage on this PC's GPU at a rotation, run three ways each frame: as on the device, and with the blend forced
  to all narrow and all wide inside the inset (the seam check's two sides, both matched as the device matches the wide).
  Tables come from the cache or a build (seconds of numpy, once per rotation); a rotation the log says is coming (building or
  ready) is built ahead in a thread."""
  def __init__(self, src_wh, rotation):
    self.src_wh = src_wh
    self.rp: dict = {}
    self.ahead: dict[tuple, threading.Thread] = {}
    self.rotation: tuple = ()
    self.follow(rotation)

  def _tables(self, rotation):
    calib = RC.calib_from_rotvec(rotation)
    if not os.path.exists(RC.table_path(self.src_wh, RC.C4_CAM, RC.table_cache_dir(), calib)):
      logger.info(f'tables for rotation {np.degrees(rotation).round(2)} deg: building (seconds, once per rotation)')
    return RC.load_tables(self.src_wh, RC.C4_CAM, RC.table_cache_dir(), calib)

  def build_ahead(self, rotation) -> None:
    key = tuple(float(v) for v in rotation)
    if key != self.rotation and key not in self.ahead:
      self.ahead[key] = threading.Thread(target=self._tables, args=(key,), daemon=True)
      self.ahead[key].start()

  def follow(self, rotation) -> None:
    """Swap to the rotation the log says is applied now."""
    from openpilot.selfdrive.modeld.reproject_c4.kernel import Reprojector
    rot = tuple(float(v) for v in rotation)
    if rot == self.rotation:
      return
    building = self.ahead.pop(rot, None)
    if building is not None:
      building.join()
    T = self._tables(rot)
    pw = T['narrow']['pw'].astype(np.int64)
    alpha = (pw >> ALPHA_SHIFT) & 0xff
    def forced(a):
      return {'wide': T['wide'], 'narrow': dict(T['narrow'], pw=((pw & ~(0xff << ALPHA_SHIFT)) | (a << ALPHA_SHIFT)).astype(np.int32))}
    tables = {'narrow': T, 'narrow_only': forced(np.where(alpha > 0, 255, 0)), 'wide_only': forced(np.zeros_like(alpha))}
    for k, tab in tables.items():
      if k in self.rp:
        self.rp[k].reload(tab)
      else:
        self.rp[k] = Reprojector(tab, RC.C4_CAM, DEV)
    self.meter_table = T['meter']
    self.meter = RC.SeamMeter(self.meter_table)
    if self.rotation:
      logger.info(f'stage: now applying {np.degrees(rot).round(2)} deg')
    self.rotation = rot

  def run(self, wide: np.ndarray, narrow: np.ndarray, gain: float) -> dict[str, np.ndarray]:
    """The comma 4 narrow and wide frames and the narrow-only and wide-only narrow frames, as strided NV12."""
    from tinygrad import Tensor
    match = self.meter.update(wide, narrow, gain)
    tw, tn = Tensor(wide, device=DEV).realize(), Tensor(narrow, device=DEV).realize()
    c4_wide, c4_narrow = self.rp['narrow'](tw, tn, match)
    out = {'narrow': c4_narrow.numpy(), 'wide': c4_wide.numpy()}
    for k in ('narrow_only', 'wide_only'):
      out[k] = self.rp[k](tw, tn, match)[1].numpy()
    return out
