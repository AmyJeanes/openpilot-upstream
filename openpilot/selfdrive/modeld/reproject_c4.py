"""Reproject the comma 3X cameras into comma 4 camera geometry, on the GPU, in front of an untouched comma 4 model.

Every comma 4 output pixel is traced through the comma 4 lens to a ray, rotated onto the 3X device axis and looked up in
the 3X wide (fisheye) or, for the narrow output, composited from the 3X narrow inset and the 3X wide surround. None of it
depends on the calibration, so the lookup tables are constants built once per process and the per-frame work is one
gather per output byte (two plus a blend for the composite) - the same cost as the model's own crop warp.

Outputs are NV12 buffers in the comma 4 camerad layout so the model pkl (keyed by its camera size) sees exactly what a
comma 4 would have given it. Lens numbers come from the bench + multi-device work in ~/tizi-to-mici (see the memory file).
"""
import hashlib
import json
import os

import numpy as np
from tinygrad import Tensor, TinyJit

from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

# measured lenses (board-calibrated 3X, self-calibrated comma 4 population); f/cx/cy in px, angles in rad
X3_WIDE = dict(f=596.669, cx=957.566, cy=581.537, k=(-0.014942, -0.0023814, -0.00064424), tc=1.51354)  # fisheye r=f·θ(1+k1θ²+k2θ⁴+k3θ⁶), linear past tc
X3_NARROW = dict(f=2600.85, cx=964.0, cy=604.0, k1=-0.36400)  # pinhole with radial barrel: r=f·x(1+k1·x²)
C4_WIDE = dict(f=442.555, cx=672.380, cy=378.718, k=(0.0089611, 0.029156, -0.015066), tc=1.39626)
R_NARROW_FROM_WIDE = (-0.0167946, 0.0022473, -0.0011422)  # rotvec: the 3X wide points ~1° above the narrow
# the wide lens + narrow->wide rotation are per unit (the cameras' relative yaw spreads +-1.5 deg across the fleet); these are the board-calibrated reference unit's
DEFAULT_CALIB = dict(wide=X3_WIDE, narrow=X3_NARROW, R=R_NARROW_FROM_WIDE)
# fleet median of 40 self-calibrated 3X units: the lens a unit gets when only its rotation is calibrated (the model's and the
# self-cal's rotations pair with this nominal centre, not with the board lens above)
X3_WIDE_POP = dict(f=597.732, cx=963.936, cy=603.959, k=(-0.011968, 0.024043, -0.0091132), tc=1.51354)
POP_ROTATION = (0.0154781, -0.0225994, -0.00068013)


def calib_from_rotvec(rotvec):
  return dict(wide=X3_WIDE_POP, narrow=X3_NARROW, R=tuple(float(v) for v in rotvec))


def rotvec_from_wide_from_device_euler(euler):
  """The model's wide_from_device_euler (device frame: roll, pitch, yaw) as our camera-axes rotvec (x=pitch, y=yaw, z=roll)."""
  r, p, y = euler
  return (float(p), float(y), float(r))


ROTATION_FILE = os.environ.get('REPROJECT_C4_ROTATION', '/data/reproject_c4/rotation.json')


def read_rotation_file() -> dict:
  try:
    d = json.load(open(ROTATION_FILE))
    if len(d["rotvec"]) == 3 and np.isfinite(d["rotvec"]).all():
      return d
  except (OSError, ValueError, KeyError, TypeError):
    pass
  return {}


def load_rotation():
  """The unit's narrow->wide rotation: the fitted/estimated value on disk, else seeded from the stock calibration's
  persisted wideFromDeviceEuler (calibrationd's block average of the model output), else the fleet median."""
  d = read_rotation_file()
  if d:
    return tuple(float(v) for v in d["rotvec"])
  try:
    from cereal import log
    from openpilot.common.params import Params
    with log.Event.from_bytes(Params().get("CalibrationParams")) as msg:
      e = list(msg.extrinsicsCalibration.wideFromDeviceEuler)
    if len(e) == 3 and np.isfinite(e).all():
      return rotvec_from_wide_from_device_euler(e)
  except Exception:
    pass
  return POP_ROTATION


APPLIED_FILE = os.path.join(os.path.dirname(ROTATION_FILE), 'applied.json')  # what modeld is running with; calibrationd waits on it


def save_applied(rotvec, fitted, stage=True) -> None:
  import time
  os.makedirs(os.path.dirname(APPLIED_FILE), exist_ok=True)
  tmp = APPLIED_FILE + '.tmp'
  json.dump({'stage': stage, 'fitted': bool(fitted), 'rotvec': [float(v) for v in rotvec], 'at': time.time()}, open(tmp, 'w')); os.replace(tmp, APPLIED_FILE)


def read_applied() -> dict:
  try:
    return json.load(open(APPLIED_FILE))
  except (OSError, ValueError):
    return {}


def save_rotation(rotvec, **extra) -> None:
  import time
  os.makedirs(os.path.dirname(ROTATION_FILE), exist_ok=True)
  tmp = ROTATION_FILE + '.tmp'
  json.dump({'rotvec': [float(v) for v in rotvec], 'at': time.time(), **extra}, open(tmp, 'w')); os.replace(tmp, ROTATION_FILE)


# --- one-shot rotation fit from a narrow+wide frame pair (numpy only: runs on the device during the calibration phase) ---

def render_layers(narrow_y, wide_y, calib, dst_wh=(1344, 760)):
  """The comma 4 narrow view's luma twice, nearest-sampled: inset from the 3X narrow, surround from the 3X wide, plus the
  wide sample coordinates and the mask where both exist. With the right rotation the two layers coincide."""
  sh, sw = narrow_y.shape
  mw, mn = sample_coords("narrow", dst_wh[0], dst_wh[1], 1.0, calib)
  xn = np.round(mn[..., 0] - 0.5).astype(int); yn = np.round(mn[..., 1] - 0.5).astype(int)
  xw = np.round(mw[..., 0] - 0.5).astype(int); yw = np.round(mw[..., 1] - 0.5).astype(int)
  vn = (xn >= 0) & (xn < sw) & (yn >= 0) & (yn < sh); vw = (xw >= 0) & (xw < sw) & (yw >= 0) & (yw < sh)
  inset = np.where(vn, narrow_y[yn.clip(0, sh - 1), xn.clip(0, sw - 1)], 0).astype(np.float32)
  surround = np.where(vw, wide_y[yw.clip(0, sh - 1), xw.clip(0, sw - 1)], 0).astype(np.float32)
  return inset, surround, mw, vn & vw


def phase_shift(a, b):
  """Shift of b relative to a (float32 patches, same shape) by phase correlation: (dx, dy, peak-to-sidelobe ratio)."""
  h, w = a.shape
  win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
  A = np.fft.rfft2((a - a.mean()) * win); B = np.fft.rfft2((b - b.mean()) * win)
  R = A * np.conj(B); R /= np.abs(R) + 1e-6
  r = np.fft.irfft2(R, s=(h, w))
  py, px = divmod(int(np.argmax(r)), w); peak = r[py, px]
  m = np.ones_like(r, bool); m[max(0, py - 5):py + 6, max(0, px - 5):px + 6] = False
  side = r[m]; psr = (peak - side.mean()) / (side.std() + 1e-9)
  def sub(c, l, rr):  # parabolic sub-pixel peak
    d = l - 2 * c + rr
    return 0.0 if d >= 0 else float(0.5 * (l - rr) / d)
  dx = px + sub(peak, r[py, (px - 1) % w], r[py, (px + 1) % w]); dy = py + sub(peak, r[(py - 1) % h, px], r[(py + 1) % h, px])
  if dx > w / 2: dx -= w
  if dy > h / 2: dy -= h
  return -dx, -dy, float(psr)


def match_rays(narrow_y, wide_y, calib, dst_wh=(1344, 760), patch=96, stride=64, min_psr=6.0, max_shift=None, min_matches=12):
  """Correspondences between the inset and the surround (patch centre -> centre + its measured shift) as narrow and wide
  rays in the current calibration's geometry, or None when the frame has too few usable patches."""
  inset, surround, mw, valid = render_layers(narrow_y, wide_y, calib, dst_wh)
  dw, dh = dst_wh; max_shift = max_shift or patch / 3
  pa, pb = [], []
  for y in range(0, dh - patch + 1, stride):
    for x in range(0, dw - patch + 1, stride):
      if valid[y:y + patch, x:x + patch].mean() < 0.98:
        continue
      a = inset[y:y + patch, x:x + patch]; b = surround[y:y + patch, x:x + patch]
      if a.std() < 4 or b.std() < 4:  # flat: sky, bonnet, night
        continue
      dx, dy, psr = phase_shift(a, b)
      if psr < min_psr or abs(dx) > max_shift or abs(dy) > max_shift:
        continue
      pa.append((x + patch / 2, y + patch / 2)); pb.append((x + patch / 2 + dx, y + patch / 2 + dy))
  if len(pa) < min_matches:
    return None
  pa, pb = np.float32(pa), np.float32(pb)
  Kn = DEVICE_CAMERAS[("mici", "os04c10")].narrow_road.intrinsics
  rays_n = unproject_pinhole(pa, Kn[0, 0], Kn[0, 2], Kn[1, 2])
  xb, yb = np.round(pb[:, 0] - 0.5).astype(int).clip(0, dw - 1), np.round(pb[:, 1] - 0.5).astype(int).clip(0, dh - 1)
  rays_w = unproject_fisheye(mw[yb, xb], calib["wide"])
  return rays_n, rays_w


def kabsch(rays_n, rays_w):
  """Rotation taking narrow rays onto wide rays, trimmed twice at the 80th percentile: (rotvec, n_kept, rms_deg)."""
  for _ in range(2):
    U, _, Vt = np.linalg.svd(rays_w.T @ rays_n); d = np.sign(np.linalg.det(U @ Vt))
    Rm = U @ np.diag([1, 1, d]) @ Vt
    res = np.degrees(np.arccos(np.clip((rays_n @ Rm.T * rays_w).sum(1), -1, 1)))
    keep = res <= np.percentile(res, 80); rays_n, rays_w = rays_n[keep], rays_w[keep]
  return matrix_to_rotvec(Rm), int(len(rays_n)), float(np.sqrt(np.mean(res[keep] ** 2)))


def fit_rotation(narrow_y, wide_y, calib0, dst_wh=(1344, 760), iters=3, coarse=True):
  """One frame pair -> (rotvec, n_matches, rms_deg) or None. With `coarse` the first pass uses big patches (a fleet-median
  seed can be tens of px off); the later passes re-render with the fit and refine with small ones. From a good seed (the
  running mean of earlier frames) coarse=False, iters=2 is enough and about half the work."""
  calib = dict(calib0); R = np.asarray(calib0["R"], float)
  for it in range(iters):
    calib["R"] = tuple(R)
    m = match_rays(narrow_y, wide_y, calib, dst_wh, patch=192, stride=96, max_shift=64) if (it == 0 and coarse) else match_rays(narrow_y, wide_y, calib, dst_wh)
    if m is None:
      return None
    R, n, rms = kabsch(*m)
  return tuple(float(v) for v in R), n, rms


def mean_rotvec(rotvecs):
  """Chordal mean of rotations (fine for the small spreads between frame fits)."""
  M = sum(rotvec_to_matrix(v) for v in rotvecs) / len(rotvecs)
  U, _, Vt = np.linalg.svd(M); d = np.sign(np.linalg.det(U @ Vt))
  return tuple(float(v) for v in matrix_to_rotvec(U @ np.diag([1, 1, d]) @ Vt))


def combine_fits(rotvecs, trim=0.2):
  """Trimmed chordal mean of per-frame fits: drops the `trim` fraction farthest from the mean. Returns (mean, kept indices,
  spread_deg = farthest kept, se_deg = standard error of the kept pitch/yaw, the convergence measure)."""
  m = mean_rotvec(rotvecs)
  dev = np.array([np.linalg.norm(matrix_to_rotvec(rotvec_to_matrix(m).T @ rotvec_to_matrix(v))) for v in rotvecs])
  keep = np.argsort(dev)[:max(1, int(round(len(rotvecs) * (1 - trim))))]
  kept = np.array([rotvecs[i] for i in keep]); mean = mean_rotvec(kept)
  se = float(np.degrees(np.linalg.norm(kept[:, :2].std(0)) / np.sqrt(len(kept)))) if len(kept) > 1 else float("inf")
  return mean, keep, float(np.degrees(dev[keep].max())), se


class RotationRefiner:
  """Continual narrow->wide rotation from the model's own wide_from_device_euler output. With the stage running, the model
  sees the composite, so that output is the residual misalignment of what it sees: averaged over `n_frames` valid frames,
  the rotation steps by `k` of it (the model under-reports by a unit-dependent factor, so it converges over a few steps
  rather than in one). Gate: moving, finite, confident output."""

  def __init__(self, rotvec, n_frames=600, k=0.7, max_step=np.radians(1.0), min_step=np.radians(0.02), min_speed=8.0, max_std=np.radians(0.5)):
    self.rotvec = np.array(rotvec, np.float64); self.n_frames, self.k, self.max_step, self.min_step, self.min_speed, self.max_std = n_frames, k, max_step, min_step, min_speed, max_std
    self.acc = np.zeros(3); self.n = 0; self.steps = 0; self.last_residual = None

  def push(self, euler, stds, v_ego):
    """Feed one model run; returns the new rotvec when a step happened, else None."""
    e = np.asarray(euler, np.float64); s = np.asarray(stds, np.float64)
    if v_ego < self.min_speed or not (np.isfinite(e).all() and np.isfinite(s).all()) or (s > self.max_std).any():
      return None
    self.acc += e; self.n += 1
    if self.n < self.n_frames:
      return None
    res = np.array(rotvec_from_wide_from_device_euler(self.acc / self.n)); self.acc[:] = 0; self.n = 0
    self.last_residual = res
    step = res * self.k; mag = np.linalg.norm(step)
    if mag < self.min_step:  # converged: under half a pixel, not worth a table swap
      return None
    if mag > self.max_step:
      step *= self.max_step / mag
    self.rotvec = matrix_to_rotvec(rotvec_to_matrix(self.rotvec) @ rotvec_to_matrix(step)); self.steps += 1
    return tuple(float(v) for v in self.rotvec)


def matrix_to_rotvec(M):
  a = np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))
  if a < 1e-9:
    return np.zeros(3)
  return a / (2 * np.sin(a)) * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])


ZMIN = np.cos(np.radians(88.0))  # rays further off-axis than this have no 3X wide pixel
FEATHER_PX = 50  # composite seam width in comma 4 narrow px (Amy picked 50 over 24 by eye on the drive clips)
UV_FILL = 128
# encoded-domain luma ratio narrow/wide vs the sensors' exposure ratio (gain*integLines), fitted in the seam ring on a
# day-to-night drive (1607 frame pairs, 4 % residual): the exponent is the ISP tone curve, so a plain constant can't fit
# both dusk and night; the ring (not the whole overlap) because the wide lens falls off towards the seam
EXPOSURE_GAIN_A, EXPOSURE_GAIN_P = 0.868, 0.708


def exposure_gain(narrow_exposure, wide_exposure, lo=0.25, hi=4.0):
  """Gain to apply to the 3X wide surround so it matches the narrow inset; exposures are gain*integLines from the camera states."""
  if not (narrow_exposure > 0 and wide_exposure > 0):
    return 1.0
  return float(np.clip(EXPOSURE_GAIN_A * (narrow_exposure / wide_exposure) ** EXPOSURE_GAIN_P, lo, hi))


def _theta_d(th, L):
  k, tc = L["k"], L["tc"]
  p = lambda t: t * (1 + k[0] * t**2 + k[1] * t**4 + k[2] * t**6)
  dp = lambda t: 1 + 3 * k[0] * t**2 + 5 * k[1] * t**4 + 7 * k[2] * t**6
  return np.where(th <= tc, p(th), p(tc) + dp(tc) * (th - tc))


def _dtheta_d(th, L):
  k = L["k"]; t = np.minimum(th, L["tc"])
  return 1 + 3 * k[0] * t**2 + 5 * k[1] * t**4 + 7 * k[2] * t**6


def unproject_fisheye(px, L):
  dx, dy = px[..., 0] - L["cx"], px[..., 1] - L["cy"]
  r = np.hypot(dx, dy); td = r / L["f"]; th = td.copy()
  for _ in range(10):
    th = th - (_theta_d(th, L) - td) / _dtheta_d(th, L)
  s = np.sin(th); rr = np.where(r > 0, r, 1)
  return np.stack([s * dx / rr, s * dy / rr, np.cos(th)], -1)


def unproject_pinhole(px, f, cx, cy):
  d = np.stack([(px[..., 0] - cx) / f, (px[..., 1] - cy) / f, np.ones(px.shape[:-1])], -1)
  return d / np.linalg.norm(d, axis=-1, keepdims=True)


def project_fisheye(rays, L):
  rho = np.hypot(rays[..., 0], rays[..., 1]); th = np.arctan2(rho, rays[..., 2])
  r = L["f"] * _theta_d(th, L); rr = np.where(rho > 0, rho, 1)
  return np.stack([L["cx"] + r * rays[..., 0] / rr, L["cy"] + r * rays[..., 1] / rr], -1)


def project_pinhole_k1(rays, L):
  x, y = rays[..., 0] / rays[..., 2], rays[..., 1] / rays[..., 2]
  s = 1 + L["k1"] * (x * x + y * y)
  return np.stack([L["cx"] + L["f"] * x * s, L["cy"] + L["f"] * y * s], -1)


def rotvec_to_matrix(v):
  v = np.asarray(v, dtype=np.float64); a = np.linalg.norm(v)
  if a == 0:
    return np.eye(3)
  k = v / a; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
  return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def sample_coords(out_cam, dst_w, dst_h, scale=1.0, calib=None):
  """Float 3X wide / narrow coordinates for every comma 4 `out_cam` pixel (scale 0.5 = the half-res chroma plane)."""
  calib = calib or DEFAULT_CALIB
  xs, ys = np.meshgrid((np.arange(dst_w) + 0.5) / scale, (np.arange(dst_h) + 0.5) / scale)
  px = np.stack([xs, ys], -1)
  if out_cam == "wide":
    rays = unproject_fisheye(px, C4_WIDE)
  else:
    Kn = DEVICE_CAMERAS[("mici", "os04c10")].narrow_road.intrinsics
    rays = unproject_pinhole(px, Kn[0, 0], Kn[0, 2], Kn[1, 2])
  rays_w = rays @ rotvec_to_matrix(calib["R"]).T
  mw = project_fisheye(rays_w, calib["wide"]) * scale
  mw[rays_w[..., 2] < ZMIN] = -1
  mn = project_pinhole_k1(rays, calib["narrow"]) * scale
  mn[rays[..., 2] <= 0] = -1
  return mw, mn


def _nv12_index(xy, src_w, src_h, stride, uv_offset, chroma):
  """Nearest-neighbour byte index into a source NV12 buffer for float sample coords; chroma coords are in the half-res plane.
  Returns (index, valid). For chroma the caller adds +1 for the V byte."""
  xy = np.clip(np.nan_to_num(xy, nan=-1.0), -1e6, 1e6)  # grazing rays project to ±inf
  x = np.round(xy[..., 0] - 0.5).astype(np.int64); y = np.round(xy[..., 1] - 0.5).astype(np.int64)
  w, h = (src_w // 2, src_h // 2) if chroma else (src_w, src_h)
  valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
  x = np.clip(x, 0, w - 1); y = np.clip(y, 0, h - 1)
  idx = (uv_offset + y * stride + 2 * x) if chroma else (y * stride + x)
  return idx, valid


IDX_BITS, ALPHA_SHIFT, INVALID_BIT = 0x3fffff, 22, 1 << 30  # one int32 per output byte: wide index | alpha << 22 | invalid


def build_tables(src_wh, dst_wh, calib=None, feather=FEATHER_PX):
  """Flat gather tables over the whole destination NV12 buffer (comma 4 layout) for the wide output and the narrow
  composite. Byte b of the output = src[idx[b]] (or the fill value where invalid); for the composite,
  alpha[b]*narrow[pn[b]] + (1-alpha[b])*gain*wide[idx[b]]. The kernel is memory-bound, so the wide index, the blend
  weight and the validity share one int32 (`pw`); only the composite needs a second table (`pn`)."""
  sw, sh = src_wh; dw, dh = dst_wh
  s_stride, s_yh, s_uvh, _ = get_nv12_info(sw, sh); s_uv = s_stride * s_yh
  assert s_stride * (s_yh + s_uvh) <= IDX_BITS + 1
  d_stride, d_yh, d_uvh, d_size = get_nv12_info(dw, dh); d_uv = d_stride * d_yh
  n_body = d_stride * (d_yh + d_uvh)
  out = {}
  for cam in ("wide", "narrow"):
    idx_w = np.zeros(n_body, np.int64); idx_n = np.zeros(n_body, np.int64)
    val_w = np.zeros(n_body, bool); val_n = np.zeros(n_body, bool); dist = np.zeros(n_body, np.float32)
    for chroma in (False, True):
      mw, mn = sample_coords(cam, dw // 2 if chroma else dw, dh // 2 if chroma else dh, 0.5 if chroma else 1.0, calib)
      iw, vw = _nv12_index(mw, sw, sh, s_stride, s_uv, chroma)
      inn, vn = _nv12_index(mn, sw, sh, s_stride, s_uv, chroma)
      # distance to the narrow frame edge, in destination px: the composite feathers over `feather` of it
      sc = (0.5 if chroma else 1.0) * ((calib or DEFAULT_CALIB)["narrow"]["f"] / DEVICE_CAMERAS[("mici", "os04c10")].narrow_road.intrinsics[0, 0])
      d = np.minimum(np.minimum(mn[..., 0], sw * (0.5 if chroma else 1) - mn[..., 0]),
                     np.minimum(mn[..., 1], sh * (0.5 if chroma else 1) - mn[..., 1])) / sc
      if chroma:
        rows = np.arange(dh // 2)[:, None]; cols = np.arange(dw // 2)[None, :]
        for plane in (0, 1):
          flat = d_uv + rows * d_stride + 2 * cols + plane
          idx_w[flat] = iw + plane; idx_n[flat] = inn + plane; val_w[flat] = vw; val_n[flat] = vn; dist[flat] = d
      else:
        flat = np.arange(dh)[:, None] * d_stride + np.arange(dw)[None, :]
        idx_w[flat] = iw; idx_n[flat] = inn; val_w[flat] = vw; val_n[flat] = vn; dist[flat] = d
    alpha = np.round(np.clip(dist / feather, 0, 1) * val_n * 255).astype(np.int64)
    pw = idx_w | (alpha << ALPHA_SHIFT) | (~val_w * INVALID_BIT)
    out[cam] = dict(pw=pw.astype(np.int32), size=d_size, body=n_body, uv_offset=d_uv)
    # the inset is sharp and the surround soft (the wide upscaled ~4x), so the blend zone can read the narrow blurred:
    # weight 1 across the feather, fading to 0 over the next feather width, packed above the narrow index
    soft = np.round(np.clip(2 - dist / feather, 0, 1) * val_n * 255).astype(np.int64)
    if cam == "narrow":
      out[cam]["pn"] = (idx_n | (soft << ALPHA_SHIFT)).astype(np.int32)
  return out


TABLE_VERSION = 4  # bump when the lens numbers, the feather or the table layout change


def calib_tag(calib, feather=FEATHER_PX):
  tag = "" if calib is None else "_" + hashlib.sha1(json.dumps(calib, sort_keys=True, default=float).encode()).hexdigest()[:10]
  return tag + ("" if feather == FEATHER_PX else f"_f{feather:g}")


def table_path(src_wh, dst_wh, cache_dir, calib=None, feather=FEATHER_PX):
  return os.path.join(cache_dir, f"reproject_c4_v{TABLE_VERSION}_{src_wh[0]}x{src_wh[1]}_{dst_wh[0]}x{dst_wh[1]}{calib_tag(calib, feather)}.npz")


def load_tables(src_wh, dst_wh, cache_dir=None, calib=None, feather=FEATHER_PX):
  """build_tables takes ~25 s of numpy on the device CPU, so keep a copy on disk (a few MB, compressed)."""
  if cache_dir is None:
    return build_tables(src_wh, dst_wh, calib, feather)
  os.makedirs(cache_dir, exist_ok=True)
  p = table_path(src_wh, dst_wh, cache_dir, calib, feather)
  if os.path.exists(p):
    z = np.load(p); T = {"wide": {}, "narrow": {}}
    for key in z.files:
      cam, k = key.split("_", 1); T[cam][k] = z[key] if z[key].ndim else int(z[key])
    return T
  T = build_tables(src_wh, dst_wh, calib, feather)
  np.savez_compressed(p + ".tmp.npz", **{f"{cam}_{k}": v for cam, tab in T.items() for k, v in tab.items()})
  os.replace(p + ".tmp.npz", p)
  return T


class SeamMeter:
  """Measures the narrow-inset / wide-surround match in the composite's seam ring straight from the two source NV12 buffers
  (host memory, a few thousand nearest-neighbour pixel pairs). The two cameras expose and white-balance independently and
  their constant differs per unit (fleet: +-8 % luma, +-2 U/V steps), so the match is measured live and low-pass filtered;
  the exposure model is the fallback when the ring is too dark/saturated. Beyond one luma gain: the wide lens's shading is
  uneven per unit (25 % side to side on some), measured as a gain gradient across the frame; and the two ISPs' tone curves
  differ (+-10 Y steps between shadows and highlights after the best single gain), measured as a gain per brightness band
  and applied as a lookup on the surround luma."""
  BANDS = ((16, 50), (50, 100), (100, 160), (160, 235))

  def __init__(self, src_wh=(1928, 1208), dst_wh=(1344, 760), calib=None, n_pairs=2048, ring=(30.0, 130.0), alpha=0.3, every=2, feedforward=True):
    """feedforward: filter the luma gains as a correction on top of the exposure model (from the camera states) and apply
    them times the model's live value, so a jump in either camera's exposure is followed the same frame."""
    sw, sh = src_wh; dw, dh = dst_wh
    s_stride, s_yh, _, _ = get_nv12_info(sw, sh); s_uv = s_stride * s_yh
    sc = (calib or DEFAULT_CALIB)["narrow"]["f"] / DEVICE_CAMERAS[("mici", "os04c10")].narrow_road.intrinsics[0, 0]
    mw, mn = sample_coords("narrow", dw, dh, 1.0, calib)
    dist = np.minimum(np.minimum(mn[..., 0], sw - mn[..., 0]), np.minimum(mn[..., 1], sh - mn[..., 1])) / sc
    iw, vw = _nv12_index(mw, sw, sh, s_stride, s_uv, False); inn, vn = _nv12_index(mn, sw, sh, s_stride, s_uv, False)
    in_ring = vw & vn & (dist > ring[0]) & (dist < ring[1])
    sel = np.flatnonzero(in_ring)[:: max(1, int(in_ring.sum()) // n_pairs)][:n_pairs]
    self.y_w, self.y_n = iw.ravel()[sel], inn.ravel()[sel]
    self.pos = np.stack([(sel % dw + 0.5) / dw * 2 - 1, (sel // dw + 0.5) / dh * 2 - 1], 1).astype(np.float32)  # c4 pixel, -1..1
    # the ring in 32x18 cells: a cell whose pairs persistently disagree with the fit (an obstruction on one lens, dirt, a
    # wiper) is excluded from the fit rather than dragged along; passing objects average out before they count
    self.cell = (np.floor((self.pos + 1) / 2 * [32, 18])).astype(int); self.cell = self.cell[:, 0] * 18 + self.cell[:, 1]
    self.cell_bad = np.zeros(32 * 18, np.float32); self.CELL_ALPHA, self.CELL_LIMIT = 0.02, 0.2
    mw2, mn2 = sample_coords("narrow", dw // 2, dh // 2, 0.5, calib)
    iw2, vw2 = _nv12_index(mw2, sw, sh, s_stride, s_uv, True); inn2, vn2 = _nv12_index(mn2, sw, sh, s_stride, s_uv, True)
    in_ring2 = in_ring[::2, ::2] & vw2 & vn2
    sel2 = np.flatnonzero(in_ring2)[:: max(1, int(in_ring2.sum()) // (n_pairs // 2))][: n_pairs // 2]
    self.uv_w, self.uv_n = iw2.ravel()[sel2], inn2.ravel()[sel2]  # U byte; V is the next one
    self.alpha, self.every, self.n_calls, self.moving, self.feedforward = alpha, every, 0, False, feedforward  # measuring every frame is ~0.7 ms of CPU on a PC
    self.state = None  # filtered [gain_y, u_off, v_off, gx, gy, band gains...]

  def measure(self, wide, narrow):
    """One frame's raw state vector, or None when fewer than 256 usable luma pairs (night, glare)."""
    yw = wide[self.y_w].astype(np.float32); yn = narrow[self.y_n].astype(np.float32)
    good = (yw > 16) & (yw < 235) & (yn > 16) & (yn < 235)
    if good.sum() < 256:
      return None
    yw, yn, pos = yw[good], yn[good], self.pos[good]
    ratio = yn / yw; gy = float(np.median(ratio))
    # tone (a gain per brightness band of the wide) and lens shading (a gain gradient over the frame) are confounded in
    # one frame - the sky is always at the top of the ring - so they are fitted jointly: log ratio = band gain + gx*x + gy*y
    band = np.searchsorted([hi for _, hi in self.BANDS[:-1]], yw)
    A = np.zeros((len(yw), len(self.BANDS) + 2), np.float32); A[np.arange(len(yw)), band] = 1; A[:, -2:] = pos
    lr = np.log(np.clip(ratio, 0.25, 4.0))
    cells = self.cell[good]; w = (self.cell_bad[cells] < self.CELL_LIMIT).astype(np.float32)
    if w.sum() < 256:
      w[:] = 1  # too much excluded: fit everything rather than nothing
    # weighted least squares by the normal equations (6x6, ridge 1e-3 so an empty band stays solvable): the general solver
    # costs 1.3 ms per call on the device CPU, this ~0.1 ms
    def solve(wt):
      Aw = A * wt[:, None]
      return np.linalg.solve(A.T @ Aw + 1e-3 * np.eye(A.shape[1], dtype=np.float32), Aw.T @ lr)
    c = solve(w)
    res = lr - A @ c; wr = w * np.clip(1 - (res / 0.3) ** 2, 0, 1) ** 2  # reweight once: a pair's pull falls off with its residual (Tukey-like)
    if wr.sum() >= 256:
      c = solve(wr)
    n = np.bincount(cells, minlength=len(self.cell_bad)); bad = np.bincount(cells, weights=np.abs(lr - A @ c), minlength=len(self.cell_bad))
    seen = n > 0; self.cell_bad[seen] += self.CELL_ALPHA * (bad[seen] / n[seen] - self.cell_bad[seen])
    counts = np.bincount(band, minlength=len(self.BANDS))
    bands = [float(np.exp(c[i])) if counts[i] >= 100 else gy for i in range(len(self.BANDS))]
    gx, gyp = float(np.clip(c[-2], -0.5, 0.5)), float(np.clip(c[-1], -0.5, 0.5))
    uw, un = wide[self.uv_w].astype(np.float32), narrow[self.uv_n].astype(np.float32)
    vw, vn = wide[self.uv_w + 1].astype(np.float32), narrow[self.uv_n + 1].astype(np.float32)
    du, dv = float(np.median(un - UV_FILL - gy * (uw - UV_FILL))), float(np.median(vn - UV_FILL - gy * (vw - UV_FILL)))
    return np.array([gy, du, dv, gx, gyp, *bands], np.float32)

  def update(self, wide, narrow, model_gain=1.0):
    """Filtered match for this frame as Reprojector kwargs; measured when the ring is usable, else decays to the exposure
    model with no colour offset, no gradient and a flat curve."""
    self.n_calls += 1
    if self.state is None or self.moving or self.n_calls % self.every == 0:
      m = self.measure(wide, narrow)
      target = m if m is not None else np.array([model_gain, 0, 0, 0, 0] + [model_gain] * len(self.BANDS), np.float32)
      if self.feedforward:  # the luma entries are kept relative to the exposure model
        target = target.copy(); target[0] /= model_gain; target[5:] /= model_gain
      if self.state is None:
        self.state = target
      else:
        # the two auto-exposures diverge for a second at a time (one brightens while the other darkens): each measurement
        # is a median over thousands of pairs, so when the match moves, follow it almost at once and measure every frame
        jump = abs(float(target[0] - self.state[0])) / max(float(self.state[0]), 0.1)
        self.state += min(self.alpha + 3 * jump, 0.9) * (target - self.state)
        self.moving = jump > 0.03
    s = self.state.copy()
    if self.feedforward:
      s[0] *= model_gain; s[5:] *= model_gain
    return dict(gain_y=float(s[0]), gain_c=float(s[0]), u_off=float(s[1]), v_off=float(s[2]), gx=float(s[3]), gy=float(s[4]),
                bands=[float(v) for v in s[5:]], lut=luma_lut(s[5:], self.BANDS))


def luma_lut(band_gains, bands=SeamMeter.BANDS):
  """Surround luma -> matched luma: the per-band gains interpolated over 0..255 (flat beyond the outer band centres)."""
  v = np.arange(256, dtype=np.float32)
  centres = [0.5 * (lo + hi) for lo, hi in bands]
  return np.clip(v * np.interp(v, centres, band_gains), 0, 255).astype(np.float32)


class Reprojector:
  """Holds the tables on `device` and runs the gathers. Call once per model run with the two 3X NV12 buffers."""

  def __init__(self, src_wh=(1928, 1208), dst_wh=(1344, 760), device=None, cache_dir=None, calib=None, feather=FEATHER_PX, soften=0):
    """soften: 0 = the narrow is one sample per output byte; k > 0 = inside the blend zone it is the mean of 5 samples k
    narrow px apart (a plus), so the inset's sharpness meets the soft surround; 4 extra gathers per byte."""
    T = load_tables(src_wh, dst_wh, cache_dir, calib, feather)
    self.soften = soften
    self.s_stride = get_nv12_info(*src_wh)[0]
    self.src_size = get_nv12_info(*src_wh)[3]
    self.size, self.body, self.uv_offset = T["wide"]["size"], T["wide"]["body"], T["wide"]["uv_offset"]
    self.device = device
    self.t = {cam: {k: Tensor(v, device=device).realize() for k, v in tab.items() if isinstance(v, np.ndarray)} for cam, tab in T.items()}
    # per-frame match parameters [gain_c, u_off, v_off, gx, gy, band gains x4, ...]; on QCOM the jit reads them straight
    # from host memory (a per-call upload costs a scheduled copy, ~4 ms of Python)
    self.params_np = np.zeros(12, np.float32); self.params_np[0] = 1; self.params_np[5:9] = 1
    if str(device or "").startswith("QCOM"):
      self.gains = Tensor.from_blob(self.params_np.ctypes.data, (12,), dtype="float32", device=device); self.host_params = True
    else:
      self.gains = Tensor(self.params_np, device=device).contiguous().realize(); self.host_params = False
    assert self.body == 3 * (self.body - self.uv_offset)  # NV12: the UV plane is the last third of the body
    self.plane = Tensor([0, 0, 1], dtype="uint8", device=device).realize()
    # pixel position for the gain gradient from the byte index (x = i % stride, y = i // stride): a plain sequential read;
    # broadcast row/column vectors cost 4 ms on the Adreno, per-pixel gathers 1.5 ms
    d_stride = get_nv12_info(*dst_wh)[0]; self.stride = d_stride; self.dw, self.dh = dst_wh
    self.idx = Tensor.arange(self.body, dtype="int32").to(device).realize()
    self.dst = None
    self._run = TinyJit(self._both)

  def bind(self, host_wide: np.ndarray, host_narrow: np.ndarray):
    """Write the outputs straight into these host arrays (each `body` bytes, e.g. modeld's packed frame slots) instead of
    returning device tensors: a GPU->host readback costs ~40 ms on the 3X, a direct write ~10 ms."""
    assert host_wide.nbytes == self.body and host_narrow.nbytes == self.body and host_wide.dtype == host_narrow.dtype == np.uint8
    dev = self.gains.device
    self.dst = (Tensor.from_blob(host_wide.ctypes.data, (self.body,), dtype="uint8", device=dev),
                Tensor.from_blob(host_narrow.ctypes.data, (self.body,), dtype="uint8", device=dev))
    self._run = TinyJit(self._both)

  def reload(self, T):
    """Swap in freshly built tables (same sizes) between runs, e.g. after the rotation was refined. The tables are jit
    inputs, so this is one upload and no re-capture (a re-capture is ~1.3 s on the 3X with no model output)."""
    assert T["wide"]["body"] == self.body and T["wide"]["uv_offset"] == self.uv_offset
    self.t = {cam: {k: Tensor(v, device=self.device).realize() for k, v in tab.items() if isinstance(v, np.ndarray)} for cam, tab in T.items()}

  def _chroma(self):
    return self.plane.reshape(3, 1).expand(3, self.body // 3).reshape(self.body).bool()  # a broadcast: no table read

  def _wide(self, wide, pw):
    return (pw < INVALID_BIT).where(wide[pw & IDX_BITS], self._chroma().cast("uint8") * UV_FILL)

  def _narrow(self, wide, narrow, gains, pw, pn):
    chroma = self._chroma()
    # match the wide surround to the narrow: luma times a gain per brightness band (the two ISPs' tone curves; piecewise
    # linear between the band centres, flat beyond) times a gain gradient over the frame (lens shading); chroma gain about
    # the 128 midpoint plus a U/V offset (independent white balance), the offset a parity broadcast: U and V bytes alternate
    off = gains[1:3].reshape(1, 2).expand(self.body // 2, 2).reshape(self.body)
    w = wide[pw & IDX_BITS].float()
    c = [0.5 * (lo + hi) for lo, hi in SeamMeter.BANDS]
    tone = gains[5]
    for i in range(1, len(c)):
      tone = tone + (gains[5 + i] - gains[4 + i]) * ((w - c[i - 1]) * (1 / (c[i] - c[i - 1]))).clip(0, 1)
    x = (self.idx % self.stride).float() * (2.0 / self.dw) - 1; y = (self.idx // self.stride).float() * (2.0 / self.dh) - 1
    w = chroma.where((w - UV_FILL) * gains[0] + UV_FILL + off, w * tone * (1 + gains[3] * x + gains[4] * y)).clip(0, 255)
    w = (pw < INVALID_BIT).where(w, chroma.cast("float32") * UV_FILL)
    a = ((pw >> ALPHA_SHIFT) & 0xff).float() * (1 / 255)
    idx = pn & IDX_BITS; n = narrow[idx].float()
    if self.soften:
      # 5 taps k px apart (2k bytes across for chroma: U and V alternate), edge-clamped to the buffer, faded by the packed weight
      k = self.soften; dx = chroma.where(2 * k, k); dy = k * self.s_stride; last = self.src_size - 1
      blur = (n + narrow[(idx + dx).minimum(last)].float() + narrow[(idx - dx).maximum(0)].float()
              + narrow[(idx + dy).minimum(last)].float() + narrow[(idx - dy).maximum(0)].float()) * 0.2
      sft = ((pn >> ALPHA_SHIFT) & 0xff).float() * (1 / 255)
      n = n + sft * (blur - n)
    return a * n + (1 - a) * w

  def _both(self, wide, narrow, gains, pw_w, pw_n, pn):
    out_w = self._wide(wide, pw_w)
    out_n = self._narrow(wide, narrow, gains, pw_n, pn).round().cast("uint8")
    if self.dst is not None:
      out_w, out_n = self.dst[0].assign(out_w), self.dst[1].assign(out_n)
    return out_w.realize(), out_n.realize()

  def __call__(self, wide, narrow, gain_y=1.0, gain_c=1.0, u_off=0.0, v_off=0.0, gx=0.0, gy=0.0, bands=None, lut=None):
    """wide/narrow: flat uint8 3X NV12 tensors (same buffers every call, e.g. the VisionIPC ring, so the jit can be replayed).
    Match parameters as SeamMeter.update() returns them: `bands` = the surround luma gain per brightness band (None = a flat
    gain_y; `lut` is accepted for callers that hold the 256-entry form). Returns (c4_wide, c4_narrow): flat uint8 NV12
    tensors of `body` bytes (stride * (y_height + uv_height), the part of the comma 4 buffer a consumer reads); with bind()
    they are views of the bound host arrays."""
    if bands is None:
      bands = [lut[int(0.5 * (lo + hi))] / (0.5 * (lo + hi)) for lo, hi in SeamMeter.BANDS] if lut is not None else [gain_y] * len(SeamMeter.BANDS)
    self.params_np[:5] = (gain_c, u_off, v_off, gx, gy); self.params_np[5:5 + len(bands)] = bands
    if not self.host_params:
      self.gains.assign(Tensor(self.params_np, device=self.gains.device)).realize()
    return self._run(wide, narrow, self.gains, self.t["wide"]["pw"], self.t["narrow"]["pw"], self.t["narrow"]["pn"])
