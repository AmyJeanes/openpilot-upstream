"""The debug clip's drawing processes: one frame's picture from the stage's renders, the 3X frames and what the main
process read from the log for it. Threads would fight the stage for the GIL, so this runs in worker processes."""
import signal

import numpy as np

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.selfdrive.modeld import reproject_c4 as RC
from openpilot.selfdrive.modeld.reproject_c4.tables import ALPHA_SHIFT
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.reproject_c4 import stats
from openpilot.tools.reproject_c4 import view as V

DW, DH = RC.C4_CAM
C4_CAMERA = DEVICE_CAMERAS['mici', 'os04c10']


def blend_weights(T: dict) -> np.ndarray:
  """The stage's narrow/wide blend weights (0 all wide - 255 all narrow) over the comma 4 narrow frame's px."""
  stride = get_nv12_info(DW, DH)[0]
  return ((T['narrow']['pw'][:stride * DH] >> ALPHA_SHIFT) & 0xff).reshape(DH, stride)[:, :DW].astype(np.uint8)


def input_coverage(alpha: np.ndarray, M: np.ndarray) -> tuple[float, float, float]:
  """Shares of the big model's narrow input (512x256 px, through modeld's warp M) that are pure 3X narrow, in the blend band
  and pure 3X wide."""
  ys, xs = np.mgrid[0:256, 0:512] + 0.5
  p = np.stack([xs, ys, np.ones_like(xs)], -1) @ M.T
  a = alpha[(p[..., 1] / p[..., 2]).astype(int).clip(0, DH - 1), (p[..., 0] / p[..., 2]).astype(int).clip(0, DW - 1)]
  return float((a == 255).mean()), float(((a > 0) & (a < 255)).mean()), float((a == 0).mean())


_W: dict = {}  # this process's slots and cameras, and the geometry for the rotation and calibration it last drew


def init_worker(cfg: dict) -> None:
  signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C reaches the whole process group; the main process stops the workers
  _W.update(cfg, geometry=None, alpha=None)


def _geometry(rotation: tuple, rpy: tuple) -> dict:
  """The inset and the input coverage: they move only with the rotation and calibration."""
  if _W['geometry'] is None or _W['geometry']['key'] != (rotation, rpy):
    if _W['alpha'] is None or _W['alpha'][0] != rotation:
      T = RC.load_tables(_W['src_wh'], RC.C4_CAM, RC.table_cache_dir(), RC.calib_from_rotvec(rotation))  # built by the stage
      _W['alpha'] = (rotation, blend_weights(T))
    alpha = _W['alpha'][1]
    _W['geometry'] = {'key': (rotation, rpy), 'masks': V.inset_masks(alpha, (V.PW, V.PH)),
                      'coverage': input_coverage(alpha, get_warp_matrix(np.asarray(rpy, np.float32), C4_CAMERA.narrow_road.intrinsics, False))}
  return _W['geometry']


def _panels(r: dict, device: dict, rpy: np.ndarray) -> dict:
  """The frame panels' pictures, from the stage's renders (r) and the 3X frames (packed NV12)."""
  sw, sh = _W['src_wh']
  Kn, Kw = C4_CAMERA.narrow_road.intrinsics, C4_CAMERA.wide_road.intrinsics
  st, y_height, _, _ = get_nv12_info(DW, DH)
  c4 = {k: V.nv12_image(r[k.removeprefix('c4_')], DW, DH, st, st * y_height) for k in ('c4_narrow', 'c4_wide')}
  panels = {k: V.bare(img, V.PW, V.PH) for k, img in c4.items()}
  for k in ('device_narrow', 'device_wide'):
    panels[k] = V.nv12_image(device[k], sw, sh, sw, sw * sh, (V.XW, V.PH))
  S_in = np.diag([V.PH / 512, V.IH / 256, 1.0])
  M = {k: get_warp_matrix(rpy, K, k == 'input_wide') @ np.linalg.inv(S_in)  # panel px -> model px -> comma 4 px, as modeld warps
       for k, K in (('input_narrow', Kn), ('input_wide', Kw))}
  panels['input_narrow'] = V.perspective(c4['c4_narrow'], M['input_narrow'], (V.PH, V.IH))
  panels['input_wide'] = V.perspective(c4['c4_wide'], M['input_wide'], (V.PH, V.IH))
  return panels


def frame(job: dict) -> None:
  """One frame's picture, into its slot."""
  slots, slot = _W['slots'], job['slot']
  r = {k: slots.view(slot, k) for k in ('narrow', 'wide', 'narrow_only', 'wide_only')}
  g = _geometry(job['rotation'], tuple(np.round(job['rpy'], 3)))
  panels = _panels(r, {k: slots.view(slot, k) for k in ('device_narrow', 'device_wide')}, job['rpy'])
  st = get_nv12_info(DW, DH)[0]
  n, n_only, w_only = (V.luma(r[k], DW, DH, st, (V.PW, V.PH)) for k in ('narrow', 'narrow_only', 'wide_only'))
  picture = V.compose(dict(panels, seam=V.seam_view(n, n_only, w_only, *g['masks']), speed=job['speed'],
                           alert=job['alert'], stats=stats.stats_panel(*V.PANELS['stats'][2:], dict(job['stats'], coverage=g['coverage']))))
  slots.view(slot, 'picture')[:] = np.asarray(picture).ravel()

