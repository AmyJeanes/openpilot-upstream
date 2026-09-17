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


class RotationRefiner:
  """Continual narrow->wide rotation from the model's own wide_from_device_euler output. With the stage running, the model
  sees the composite, so that output is the residual misalignment of what it sees: averaged over `n_frames` valid frames,
  the rotation steps by `k` of it (the model under-reports by a unit-dependent factor, so it converges over a few steps
  rather than in one). Gate: moving, finite, confident output."""

  def __init__(self, rotvec, n_frames=600, k=0.7, max_step=np.radians(1.0), min_speed=8.0, max_std=np.radians(0.5)):
    self.rotvec = np.array(rotvec, np.float64); self.n_frames, self.k, self.max_step, self.min_speed, self.max_std = n_frames, k, max_step, min_speed, max_std
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
FEATHER_PX = 24  # composite seam width in comma 4 narrow px
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
    if cam == "narrow":
      out[cam]["pn"] = idx_n.astype(np.int32)
  return out


TABLE_VERSION = 2  # bump when the lens numbers or the table layout change


def calib_tag(calib, feather=FEATHER_PX):
  tag = "" if calib is None else "_" + hashlib.sha1(json.dumps(calib, sort_keys=True, default=float).encode()).hexdigest()[:10]
  return tag + ("" if feather == FEATHER_PX else f"_f{feather:g}")


def load_tables(src_wh, dst_wh, cache_dir=None, calib=None, feather=FEATHER_PX):
  """build_tables takes ~25 s of numpy on the device CPU, so keep a copy on disk (a few MB, compressed)."""
  if cache_dir is None:
    return build_tables(src_wh, dst_wh, calib, feather)
  os.makedirs(cache_dir, exist_ok=True)
  p = os.path.join(cache_dir, f"reproject_c4_v{TABLE_VERSION}_{src_wh[0]}x{src_wh[1]}_{dst_wh[0]}x{dst_wh[1]}{calib_tag(calib, feather)}.npz")
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

  def __init__(self, src_wh=(1928, 1208), dst_wh=(1344, 760), calib=None, n_pairs=4096, ring=(30.0, 130.0), alpha=0.3, every=2):
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
    mw2, mn2 = sample_coords("narrow", dw // 2, dh // 2, 0.5, calib)
    iw2, vw2 = _nv12_index(mw2, sw, sh, s_stride, s_uv, True); inn2, vn2 = _nv12_index(mn2, sw, sh, s_stride, s_uv, True)
    in_ring2 = in_ring[::2, ::2] & vw2 & vn2
    sel2 = np.flatnonzero(in_ring2)[:: max(1, int(in_ring2.sum()) // (n_pairs // 2))][: n_pairs // 2]
    self.uv_w, self.uv_n = iw2.ravel()[sel2], inn2.ravel()[sel2]  # U byte; V is the next one
    self.alpha, self.every, self.n_calls = alpha, every, 0  # measuring every frame is ~0.6 ms of CPU on a PC; the filter is slower than that anyway
    self.state = None  # filtered [gain_y, u_off, v_off, gx, gy, band gains...]

  def measure(self, wide, narrow):
    """One frame's raw state vector, or None when fewer than 256 usable luma pairs (night, glare)."""
    yw = wide[self.y_w].astype(np.float32); yn = narrow[self.y_n].astype(np.float32)
    good = (yw > 16) & (yw < 235) & (yn > 16) & (yn < 235)
    if good.sum() < 256:
      return None
    yw, yn, pos = yw[good], yn[good], self.pos[good]
    ratio = yn / yw; gy = float(np.median(ratio))
    # gain gradient across the frame: least squares of the normalised ratio on the pixel position (clipped: robust enough at 4k pairs)
    r = np.clip(ratio / gy, 0.4, 2.5) - 1
    A = np.c_[pos, np.ones(len(pos), np.float32)]
    c = np.linalg.lstsq(A, r, rcond=None)[0]
    keep = np.abs(r - A @ c) < 0.3  # one refit without the outliers (saturated sky, the car's own bonnet, seam-crossing objects)
    if keep.sum() >= 256:
      c = np.linalg.lstsq(A[keep], r[keep], rcond=None)[0]
    gx, gyp = float(np.clip(c[0], -0.5, 0.5)), float(np.clip(c[1], -0.5, 0.5))
    flat = ratio / (1 + gx * pos[:, 0] + gyp * pos[:, 1])
    bands = [float(np.median(flat[m])) if (m := (yw >= lo) & (yw < hi)).sum() >= 100 else gy for lo, hi in self.BANDS]
    uw, un = wide[self.uv_w].astype(np.float32), narrow[self.uv_n].astype(np.float32)
    vw, vn = wide[self.uv_w + 1].astype(np.float32), narrow[self.uv_n + 1].astype(np.float32)
    du, dv = float(np.median(un - UV_FILL - gy * (uw - UV_FILL))), float(np.median(vn - UV_FILL - gy * (vw - UV_FILL)))
    return np.array([gy, du, dv, gx, gyp, *bands], np.float32)

  def update(self, wide, narrow, model_gain=1.0):
    """Filtered match for this frame as Reprojector kwargs; measured when the ring is usable, else decays to the exposure
    model with no colour offset, no gradient and a flat curve."""
    self.n_calls += 1
    if self.state is None or self.n_calls % self.every == 0:
      m = self.measure(wide, narrow)
      target = m if m is not None else np.array([model_gain, 0, 0, 0, 0] + [model_gain] * len(self.BANDS), np.float32)
      self.state = target if self.state is None else self.state + self.alpha * (target - self.state)
    s = self.state
    return dict(gain_y=float(s[0]), gain_c=float(s[0]), u_off=float(s[1]), v_off=float(s[2]), gx=float(s[3]), gy=float(s[4]),
                lut=luma_lut(s[5:], self.BANDS))


def luma_lut(band_gains, bands=SeamMeter.BANDS):
  """Surround luma -> matched luma: the per-band gains interpolated over 0..255 (flat beyond the outer band centres)."""
  v = np.arange(256, dtype=np.float32)
  centres = [0.5 * (lo + hi) for lo, hi in bands]
  return np.clip(v * np.interp(v, centres, band_gains), 0, 255).astype(np.float32)


class Reprojector:
  """Holds the tables on `device` and runs the gathers. Call once per model run with the two 3X NV12 buffers."""

  def __init__(self, src_wh=(1928, 1208), dst_wh=(1344, 760), device=None, cache_dir=None, calib=None, feather=FEATHER_PX):
    T = load_tables(src_wh, dst_wh, cache_dir, calib, feather)
    self.size, self.body, self.uv_offset = T["wide"]["size"], T["wide"]["body"], T["wide"]["uv_offset"]
    self.device = device
    self.t = {cam: {k: Tensor(v, device=device).realize() for k, v in tab.items() if isinstance(v, np.ndarray)} for cam, tab in T.items()}
    # per-frame match parameters: [gain_c, u_off, v_off, gx, gy, 0, 0, 0] and a 256-entry surround-luma lookup; on QCOM the
    # jit reads them straight from host memory (a per-call upload costs a scheduled copy, ~4 ms of Python)
    self.params_np = np.zeros(8, np.float32); self.params_np[0] = 1; self.lut_np = np.arange(256, dtype=np.float32)
    if str(device or "").startswith("QCOM"):
      self.gains = Tensor.from_blob(self.params_np.ctypes.data, (8,), dtype="float32", device=device)
      self.lut = Tensor.from_blob(self.lut_np.ctypes.data, (256,), dtype="float32", device=device)
      self.host_params = True
    else:
      self.gains = Tensor(self.params_np, device=device).contiguous().realize(); self.lut = Tensor(self.lut_np, device=device).contiguous().realize()
      self.host_params = False
    assert self.body == 3 * (self.body - self.uv_offset)  # NV12: the UV plane is the last third of the body
    self.plane = Tensor([0, 0, 1], dtype="uint8", device=device).realize()
    # pixel position for the gain gradient, as two broadcast vectors (no per-pixel table): the body is rows x stride
    d_stride = get_nv12_info(*dst_wh)[0]; self.rows = self.body // d_stride; self.stride = d_stride
    self.xs = Tensor(((np.arange(d_stride) + 0.5) / dst_wh[0] * 2 - 1).astype(np.float32), device=device).reshape(1, d_stride).realize()
    self.ys = Tensor(((np.arange(self.rows) + 0.5) / dst_wh[1] * 2 - 1).astype(np.float32), device=device).reshape(self.rows, 1).realize()
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
    """Swap in freshly built tables (same sizes) between runs, e.g. after the rotation was refined; the jit is re-captured."""
    assert T["wide"]["body"] == self.body and T["wide"]["uv_offset"] == self.uv_offset
    self.t = {cam: {k: Tensor(v, device=self.device).realize() for k, v in tab.items() if isinstance(v, np.ndarray)} for cam, tab in T.items()}
    self._run = TinyJit(self._both)

  def _chroma(self):
    return self.plane.reshape(3, 1).expand(3, self.body // 3).reshape(self.body).bool()  # a broadcast: no table read

  def _wide(self, wide):
    pw = self.t["wide"]["pw"]
    return (pw < INVALID_BIT).where(wide[pw & IDX_BITS], self._chroma().cast("uint8") * UV_FILL)

  def _narrow(self, wide, narrow, gains, lut):
    t = self.t["narrow"]; pw = t["pw"]; chroma = self._chroma()
    # match the wide surround to the narrow: luma through the lookup (tone curve) times a gain gradient over the frame
    # (lens shading), chroma gain about the 128 midpoint plus a U/V offset (independent white balance); the offset is a
    # parity broadcast (U and V bytes alternate in NV12) and the gradient a row/column broadcast, so neither reads a table
    off = gains[1:3].reshape(1, 2).expand(self.body // 2, 2).reshape(self.body)
    grad = (1 + gains[3] * self.xs + gains[4] * self.ys).expand(self.rows, self.stride).reshape(self.body)
    w8 = wide[pw & IDX_BITS]; w = w8.float()
    w = chroma.where((w - UV_FILL) * gains[0] + UV_FILL + off, lut[w8.cast("int32")] * grad).clip(0, 255)
    w = (pw < INVALID_BIT).where(w, chroma.cast("float32") * UV_FILL)
    a = ((pw >> ALPHA_SHIFT) & 0xff).float() * (1 / 255)
    return a * narrow[t["pn"]].float() + (1 - a) * w

  def _both(self, wide, narrow, gains, lut):
    out_w = self._wide(wide)
    out_n = self._narrow(wide, narrow, gains, lut).round().cast("uint8")
    if self.dst is not None:
      out_w, out_n = self.dst[0].assign(out_w), self.dst[1].assign(out_n)
    return out_w.realize(), out_n.realize()

  def __call__(self, wide, narrow, gain_y=1.0, gain_c=1.0, u_off=0.0, v_off=0.0, gx=0.0, gy=0.0, lut=None):
    """wide/narrow: flat uint8 3X NV12 tensors (same buffers every call, e.g. the VisionIPC ring, so the jit can be replayed).
    Match parameters as SeamMeter.update() returns them (lut None = a flat gain_y). Returns (c4_wide, c4_narrow): flat
    uint8 NV12 tensors of `body` bytes (stride * (y_height + uv_height), the part of the comma 4 buffer a consumer reads);
    with bind() they are views of the bound host arrays."""
    self.params_np[:5] = (gain_c, u_off, v_off, gx, gy)
    self.lut_np[:] = np.arange(256, dtype=np.float32) * gain_y if lut is None else lut
    if not self.host_params:
      self.gains.assign(Tensor(self.params_np, device=self.gains.device)).realize(); self.lut.assign(Tensor(self.lut_np, device=self.lut.device)).realize()
    return self._run(wide, narrow, self.gains, self.lut)
