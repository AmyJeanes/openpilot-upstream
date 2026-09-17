"""Side-by-side video of a recorded 3X drive: raw cameras (left) vs the same frames reprojected into comma 4 geometry
(right) by the same kernel that runs on the device, with the per-frame exposure gains the integration used (camera
states from the rlog, times the constant K). Optional overlays: the model's 512x256 warp crops (stock's on the raw 3X
frames with 3X intrinsics, ours on the comma 4 frames with comma 4 intrinsics; both should frame the same scene), the
four reconstructed model inputs, and a telemetry line from the rlog.
cd ~/git/sunnypilot && DEV=CUDA PYTHONPATH=. uv run python ~/tizi-to-mici/harness/render_sbs.py <route dir> <out.mp4>
    [--segs 3,4] [--K 1.0] [--max-frames N] [--overlays crop,inputs,telemetry]"""
import argparse
import glob
import os
import subprocess
import time

import cv2
import numpy as np
from tinygrad import Tensor, Device
from openpilot.tools.lib.logreader import LogReader
from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.common.transformations.model import get_warp_matrix, MEDMODEL_INPUT_SIZE
try:
  from openpilot.selfdrive.modeld import reproject_c4 as RC  # openpilot branch
except ImportError:
  from openpilot.sunnypilot.modeld_v2 import reproject_c4 as RC  # sunnypilot tree
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info

FFMPEG = os.environ.get("FFMPEG", "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg")  # the venv ships a pipe-less ffmpeg
ap = argparse.ArgumentParser()
ap.add_argument("route"); ap.add_argument("out")
ap.add_argument("--segs", default="all"); ap.add_argument("--K", type=float, default=1.0, help="extra multiplier on the fitted exposure gain"); ap.add_argument("--max-frames", type=int, default=10**9)
ap.add_argument("--overlays", default="crop,telemetry,hud", help="comma list of crop,inputs,telemetry,hud,fit (or none); fit = side panel showing the live seam match")
ap.add_argument("--meter", action=argparse.BooleanOptionalAction, default=True, help="live seam match (gain + U/V offsets) as on the device; --no-meter = exposure model only")
ap.add_argument("--start", type=int, default=0, help="skip this many frames of each segment before rendering")
ap.add_argument("--feather", type=float, default=RC.FEATHER_PX, help="composite seam width in comma 4 px")
ap.add_argument("--compare", default=None, choices=["exposure", "soften", "feedforward"], help="A/B on the top row, bottom: raw narrow + reprojected wide. exposure: exposure model only (left) vs live seam meter (right); soften: inset as is (left) vs softened inset (right, --soften); feedforward: meter filtering the gain (left) vs filtering the correction to the exposure model (right)")
ap.add_argument("--stretch", action="store_true", help="three columns: raw 3X | 3X frames simply resized to comma 4 dimensions | our reprojection")
ap.add_argument("--soften", type=int, default=0, help="inset softening inside the blend zone: taps k narrow px apart (0 = off)")
ap.add_argument("--calib", default="auto", help="auto = the unit's own self-cal when the route dir is a fleet device dir (harness/unit_calib.py), else the board calibration; board = always the board")
a = ap.parse_args()
OV = set(a.overlays.split(",")) - {"none", ""}
DEV = os.environ.get("DEV", Device.DEFAULT)
SW, SH, DW, DH = 1928, 1208, 1344, 760
MW, MH = MEDMODEL_INPUT_SIZE
s_stride, s_yh, s_uvh, s_size = get_nv12_info(SW, SH)
COL, LH = 960, 30
rs = (COL, round(COL * SH / SW)); rd = (COL, round(COL * DH / DW)); ROW = rs[1]
IN_S = 0.9; IW, IH = round(MW * IN_S), round(MH * IN_S); IN_ROW = (IH + LH) if "inputs" in OV else 0
PW = 520 if "fit" in OV else 0  # seam-match panel on the right
NCOL = 3 if a.stretch else 2
H, W = 2 * (ROW + LH) + IN_ROW, NCOL * COL + PW
FONT = cv2.FONT_HERSHEY_SIMPLEX
X3 = DEVICE_CAMERAS[("tizi", "ox03c10")]; C4 = DEVICE_CAMERAS[("mici", "os04c10")]
PATH_Z_OFF = 1.22  # UI's _path_offset_z
CORNERS = np.array([[0, 0, 1], [MW, 0, 1], [MW, MH, 1], [0, MH, 1]], np.float64).T


def to_layout(nv12_packed):
  """ffmpeg's tightly packed NV12 -> the device buffer layout (stride 2048, padded plane heights)."""
  buf = np.zeros(s_size, np.uint8)
  y = nv12_packed[:SW * SH].reshape(SH, SW); uv = nv12_packed[SW * SH:].reshape(SH // 2, SW)
  buf[:SH * s_stride].reshape(SH, s_stride)[:, :SW] = y
  buf[s_stride * s_yh:s_stride * s_yh + (SH // 2) * s_stride].reshape(SH // 2, s_stride)[:, :SW] = uv
  return buf


def from_nv12(buf, w, h):
  stride, yh, _, _ = get_nv12_info(w, h)
  y = buf[:h * stride].reshape(h, stride)[:, :w]
  uv = buf[stride * yh:stride * yh + (h // 2) * stride].reshape(h // 2, stride)
  i420 = np.concatenate([y, uv[:, 0:w:2].reshape(h // 4, w), uv[:, 1:w:2].reshape(h // 4, w)])
  return cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)


def label(img, text):
  bar = np.zeros((LH, img.shape[1], 3), np.uint8)
  cv2.putText(bar, text, (8, 21), FONT, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
  return np.vstack([bar, img])


def fit(img, size):
  out = np.zeros((ROW, COL, 3), np.uint8); r = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
  y0 = (ROW - r.shape[0]) // 2; out[y0:y0 + r.shape[0], :r.shape[1]] = r
  return out


def draw_crop(img, M, color):
  """Outline of the 512x256 model frame in camera pixels."""
  p = M @ CORNERS; p = (p[:2] / p[2]).T
  cv2.polylines(img, [np.round(p).astype(np.int32)], True, color, 3, cv2.LINE_AA)


def model_input(img, M):
  return cv2.warpPerspective(img, M, (MW, MH), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)  # the device warp clamps to the edge


def project(pts, view_from_calib, K):
  """calibrated-frame points (N,3) -> image pixels (N,2) or NaN behind the camera; same maths as the UI."""
  v = view_from_calib @ pts.T; c = K @ v
  with np.errstate(divide="ignore", invalid="ignore"):
    uv = (c[:2] / c[2]).T
  uv[c[2] <= 0.5] = np.nan
  return uv


def draw_hud(img, out, view_from_calib, K, path_z=PATH_Z_OFF):
  """Lane lines (white, thickness by probability), road edges (red), planned path (green band, 1.8 m wide, at the UI's
  path height offset). `out` = (laneLines, laneLineProbs, roadEdges, position). Frame conventions come from the
  transformations module in PYTHONPATH, which must be the tree that produced the logs."""
  lanes, probs, edges, pos = out
  h, w = img.shape[:2]
  def poly(p3):
    uv = project(p3, view_from_calib, K); ok = np.isfinite(uv).all(1) & (uv[:, 0] > -w) & (uv[:, 0] < 2 * w) & (uv[:, 1] > -h) & (uv[:, 1] < 2 * h)
    return np.round(uv[ok]).astype(np.int32)
  # path band: left and right edge at y +- 0.9, z + 1.22
  left = pos + np.array([0, -0.9, path_z]); right = pos + np.array([0, 0.9, path_z])
  pl, pr = poly(left), poly(right)
  if len(pl) > 1 and len(pr) > 1:
    band = np.vstack([pl, pr[::-1]]); layer = img.copy(); cv2.fillPoly(layer, [band], (60, 220, 60)); cv2.addWeighted(layer, 0.35, img, 0.65, 0, img)
  for lane, pr_ in zip(lanes, probs):
    if pr_ < 0.1:
      continue  # the UI scales line width by confidence; below this it's invisible
    q = poly(lane)
    if len(q) > 1:
      cv2.polylines(img, [q], False, (255, 255, 255), max(1, int(round(1 + 3 * pr_))), cv2.LINE_AA)
  for e in edges:
    q = poly(e)
    if len(q) > 1:
      cv2.polylines(img, [q], False, (60, 60, 255), 2, cv2.LINE_AA)


calib = None
if a.calib == "auto":
  try:
    import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from unit_calib import calib_for
    calib = calib_for(os.path.basename(os.path.abspath(a.route)))
  except ImportError:
    pass
print("calibration:", "unit's own self-cal" if calib else "board (reference unit)", flush=True)
def fit_panel(comp_bgr, wide_np, narrow_np, match, model_gain, hist):
  """Right-hand panel: the seam ring's measurement points on the composite, coloured by the mismatch left AFTER the
  match (green = matched, red = surround still too bright, blue = too dark), and the meter's fitted values over time."""
  P = np.zeros((H, PW, 3), np.uint8)
  yw = wide_np[meter.y_w].astype(np.float32); yn = narrow_np[meter.y_n].astype(np.float32); pos = meter.pos
  good = (yw > 16) & (yw < 235) & (yn > 16) & (yn < 235)
  matched = match["lut"][yw.astype(int)] * (1 + match["gx"] * pos[:, 0] + match["gy"] * pos[:, 1])
  res = np.where(good, yn / np.maximum(matched, 1) - 1, np.nan)           # after the full match
  raw = np.where(good, yn / np.maximum(yw * model_gain, 1) - 1, np.nan)  # what the exposure model alone would leave
  hist.append(dict(gain=match["gain_y"], model=model_gain, u=match["u_off"], v=match["v_off"], gx=match["gx"], gy=match["gy"],
                   bands=match["lut"][[33, 75, 130, 197]] / np.array([33, 75, 130, 197]), res=float(np.nanmedian(np.abs(res))), raw=float(np.nanmedian(np.abs(raw)))))
  # residual map
  mw_, mh_ = PW - 20, round((PW - 20) * DH / DW)
  small = cv2.resize(comp_bgr, (mw_, mh_), interpolation=cv2.INTER_AREA)
  # single pixel pairs are noisy (the narrow is 4x finer than the wide), so the colour is the mean over small cells of the ring
  cell = (np.floor((pos + 1) / 2 * [32, 18])).astype(int); key = cell[:, 0] * 18 + cell[:, 1]
  sums = np.zeros(32 * 18); cnts = np.zeros(32 * 18); ok = np.isfinite(res)
  np.add.at(sums, key[ok], res[ok]); np.add.at(cnts, key[ok], 1)
  for (px, py), k, r in zip(pos, key, res):
    if np.isfinite(r) and cnts[k] >= 3:
      x, y = int((px + 1) / 2 * mw_), int((py + 1) / 2 * mh_)
      c = max(-1.0, min(1.0, sums[k] / cnts[k] / 0.2))  # +-20 % -> full colour
      col = (0, 200, 60) if abs(c) < 0.25 else ((255, int(120 * (1 - c)), 0) if c > 0 else (0, int(120 * (1 + c)), 255))  # res > 0: narrow brighter = surround too dark = blue
      cv2.circle(small, (x, y), 2, col, -1)
  cv2.putText(P, "seam ring after the match: green matched, red surround too bright, blue too dark", (8, 18), FONT, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
  P[26:26 + mh_, 10:10 + mw_] = small
  y0 = 26 + mh_ + 12
  charts = (("surround luma gain: live vs exposure model", (("live", "gain", (80, 220, 120)), ("model", "model", (120, 120, 120))), 0.3, 1.5),
            ("median |mismatch| in the ring: raw (grey) vs after match", (("after", "res", (80, 220, 120)), ("raw", "raw", (120, 120, 120))), 0.0, 0.4),
            ("U / V offset (steps)", (("U", "u", (255, 160, 60)), ("V", "v", (60, 160, 255))), -8, 8),
            ("gain gradient across the frame: x / y", (("x", "gx", (255, 160, 60)), ("y", "gy", (60, 160, 255))), -0.5, 0.5),
            ("gain per brightness band: dark ... bright", (("16-50", 0, (90, 90, 255)), ("50-100", 1, (90, 200, 255)), ("100-160", 2, (90, 255, 160)), ("160-235", 3, (255, 255, 120))), 0.3, 1.5))
  ch = (H - y0 - 10) // len(charts)
  for name, series, lo, hi in charts:
    cv2.putText(P, name, (10, y0 + 14), FONT, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    top, bot = y0 + 20, y0 + ch - 8
    cv2.rectangle(P, (10, top), (PW - 10, bot), (70, 70, 70), 1)
    for k, (lab, key, col) in enumerate(series):
      vals = [(h["bands"][key] if isinstance(key, int) else h[key]) for h in hist]
      pts = [(int(10 + (PW - 20) * i / max(a.max_frames, 1)), int(bot - (bot - top) * (min(max(v, lo), hi) - lo) / (hi - lo))) for i, v in enumerate(vals)]
      if len(pts) > 1:
        cv2.polylines(P, [np.array(pts, np.int32)], False, col, 1, cv2.LINE_AA)
      cv2.putText(P, f"{lab} {vals[-1]:+.2f}" if lo < 0 else f"{lab} {vals[-1]:.2f}", (PW - 10 - 95 * (len(series) - k), top + 14), FONT, 0.4, col, 1, cv2.LINE_AA)
    cv2.putText(P, f"{hi:g}", (14, top + 12), FONT, 0.35, (120, 120, 120), 1, cv2.LINE_AA); cv2.putText(P, f"{lo:g}", (14, bot - 3), FONT, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
    y0 += ch
  return P


rp = RC.Reprojector(device=DEV, feather=a.feather, calib=calib, soften=a.soften if a.compare != "soften" else 0)
rp_b = RC.Reprojector(device=DEV, feather=a.feather, calib=calib, soften=a.soften) if a.compare == "soften" else None  # the tweaked kernel for the A/B
meter = RC.SeamMeter(calib=calib)
meter_a = RC.SeamMeter(calib=calib, feedforward=False) if a.compare == "feedforward" else None  # the previous behaviour for the A/B
enc = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "20", "-i", "-",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
segs = sorted(glob.glob(os.path.join(a.route, "*--*--*")), key=lambda p: int(p.rsplit("--", 1)[1]))
if a.segs != "all":
  want = {int(x) for x in a.segs.split(",")}; segs = [p for p in segs if int(p.rsplit("--", 1)[1]) in want]
total = 0; t_start = time.time(); rpy = np.zeros(3, np.float32); path_z = PATH_Z_OFF; fit_hist = []
for p in segs:
  seg = int(p.rsplit("--", 1)[1])
  cs = {"road": {}, "wide": {}}; idx = {"road": {}, "wide": {}}; model = {}; hud = {}; v_ego = []; engaged = []; cal = []
  for m in LogReader(os.path.join(p, "rlog.zst")):
    w = m.which()
    if w in ("narrowRoadCameraState", "wideRoadCameraState"):
      c = getattr(m, w); cs["road" if w == "narrowRoadCameraState" else "wide"][c.frameId] = (c.gain, c.integLines, c.timestampSof)
    elif w in ("narrowRoadEncodeIdx", "wideRoadEncodeIdx"):
      e = getattr(m, w); idx["road" if w == "narrowRoadEncodeIdx" else "wide"][e.segmentId] = e.frameId
    elif w == "modelV2":
      mv = m.modelV2; model[mv.frameId] = (mv.modelExecutionTime, mv.frameDropPerc, mv.big)
      if "hud" in OV:
        hud[mv.frameId] = ([np.array([l.x, l.y, l.z], np.float32).T for l in mv.laneLines], list(mv.laneLineProbs),
                           [np.array([e.x, e.y, e.z], np.float32).T for e in mv.roadEdges], np.array([mv.position.x, mv.position.y, mv.position.z], np.float32).T)
    elif w == "carState":
      v_ego.append((m.logMonoTime, m.carState.vEgo))
    elif w == "selfdriveState":
      engaged.append((m.logMonoTime, m.selfdriveState.enabled))
    elif w == "extrinsicsCalibration":
      ec = m.extrinsicsCalibration; cal.append((np.array(ec.rpyCalib, np.float32), ec.height[0] if len(ec.height) else PATH_Z_OFF))
  if cal:
    rpy, path_z = cal[-1]
  # warp matrices for this segment's calibration: stock's crops on the 3X frames, ours on the comma 4 frames
  M_x3 = {"n": get_warp_matrix(rpy, X3.narrow_road.intrinsics, False), "w": get_warp_matrix(rpy, X3.wide_road.intrinsics, True)}
  M_c4 = {"n": get_warp_matrix(rpy, C4.narrow_road.intrinsics, False), "w": get_warp_matrix(rpy, C4.wide_road.intrinsics, True)}
  view_from_calib = view_frame_from_device_frame @ rot_from_euler(rpy)
  v_t = np.array([t for t, _ in v_ego]); v_v = np.array([v for _, v in v_ego]); e_t = np.array([t for t, _ in engaged]); e_v = np.array([v for _, v in engaged])
  dec = [subprocess.Popen([FFMPEG, "-v", "error", "-i", os.path.join(p, f), "-pix_fmt", "nv12", "-f", "rawvideo", "-"], stdout=subprocess.PIPE, bufsize=10**7)
         for f in ("ecamera.hevc", "fcamera.hevc")]
  n = 0
  while n < a.start + a.max_frames:
    raws = [d.stdout.read(SW * SH * 3 // 2) for d in dec]
    if any(len(r) < SW * SH * 3 // 2 for r in raws):
      break
    if n < a.start:
      n += 1; continue
    wide_np, narrow_np = (to_layout(np.frombuffer(r, np.uint8)) for r in raws)
    fid_n, fid_w = idx["road"].get(n), idx["wide"].get(n)
    sn, sw_ = cs["road"].get(fid_n), cs["wide"].get(fid_w)
    g = float(np.clip(a.K * RC.exposure_gain(sn[0] * sn[1], sw_[0] * sw_[1]), 0.25, 4.0)) if sn and sw_ else 1.0
    g_model = g
    if a.meter:  # what the device does: measure the match in the seam ring of these frames, the exposure model as fallback
      match = meter.update(wide_np, narrow_np, g); g, du, dv = match["gain_y"], match["u_off"], match["v_off"]
    else:
      match = dict(gain_y=g, gain_c=g, u_off=0.0, v_off=0.0, gx=0.0, gy=0.0, lut=np.arange(256, dtype=np.float32) * g); du = dv = 0.0
    ow, on = rp(Tensor(wide_np, device=DEV).realize(), Tensor(narrow_np, device=DEV).realize(), **match)
    ow_bgr, on_bgr = from_nv12(ow.numpy(), DW, DH), from_nv12(on.numpy(), DW, DH)
    comp_clean = on_bgr.copy() if "fit" in OV else None
    if a.compare == "exposure":
      on_model = from_nv12(rp(Tensor(wide_np, device=DEV).realize(), Tensor(narrow_np, device=DEV).realize(), gain_y=g_model, gain_c=g_model)[1].numpy(), DW, DH)
    elif a.compare == "feedforward":
      match_a = meter_a.update(wide_np, narrow_np, g_model)
      on_model = from_nv12(rp(Tensor(wide_np, device=DEV).realize(), Tensor(narrow_np, device=DEV).realize(), **match_a)[1].numpy(), DW, DH)
    elif a.compare == "soften":
      on_model = from_nv12(rp_b(Tensor(wide_np, device=DEV).realize(), Tensor(narrow_np, device=DEV).realize(), **match)[1].numpy(), DW, DH)
    raw_w, raw_n = from_nv12(wide_np, SW, SH), from_nv12(narrow_np, SW, SH)
    if a.stretch:
      str_w, str_n = cv2.resize(raw_w, (DW, DH), interpolation=cv2.INTER_AREA), cv2.resize(raw_n, (DW, DH), interpolation=cv2.INTER_AREA)
    inputs = [model_input(raw_n, M_x3["n"]), model_input(raw_w, M_x3["w"]), model_input(on_bgr, M_c4["n"]), model_input(ow_bgr, M_c4["w"])] if "inputs" in OV else None
    if a.stretch and inputs is not None:
      inputs[2:2] = [model_input(str_n, M_c4["n"]), model_input(str_w, M_c4["w"])]
    if "hud" in OV and cal and fid_n in hud:
      o = hud[fid_n]
      draw_hud(raw_n, o, view_from_calib, X3.narrow_road.intrinsics, path_z); draw_hud(raw_w, o, view_from_calib, X3.wide_road.intrinsics, path_z)
      draw_hud(on_bgr, o, view_from_calib, C4.narrow_road.intrinsics, path_z); draw_hud(ow_bgr, o, view_from_calib, C4.wide_road.intrinsics, path_z)
      if a.compare:
        draw_hud(on_model, o, view_from_calib, C4.narrow_road.intrinsics, path_z)
      if a.stretch:
        draw_hud(str_n, o, view_from_calib, C4.narrow_road.intrinsics, path_z); draw_hud(str_w, o, view_from_calib, C4.wide_road.intrinsics, path_z)
    if "crop" in OV:
      draw_crop(raw_n, M_x3["n"], (0, 200, 255)); draw_crop(raw_w, M_x3["w"], (0, 200, 255)); draw_crop(on_bgr, M_c4["n"], (80, 255, 80)); draw_crop(ow_bgr, M_c4["w"], (80, 255, 80))
      if a.compare:
        draw_crop(on_model, M_c4["n"], (80, 255, 80))
      if a.stretch:
        draw_crop(str_n, M_c4["n"], (255, 80, 255)); draw_crop(str_w, M_c4["w"], (255, 80, 255))
    t = n / 20
    tele = ""
    if "telemetry" in OV and fid_n is not None:
      tele = f"frame {fid_n}"
      if sn and len(v_t):
        k = np.searchsorted(v_t, sn[2]) - 1; ke = np.searchsorted(e_t, sn[2]) - 1
        tele += f"  |  {v_v[max(k, 0)] * 3.6:.0f} km/h" + (", engaged" if len(e_t) and e_v[max(ke, 0)] else "")
      mt = model.get(fid_n)
      if mt:
        tele += f"  |  model {mt[0] * 1e3:.1f} ms, {mt[1]:.0f} % drops" + ("" if mt[2] else ", small model")
    expo = f"  |  exposure n {sn[0]:.1f}x{sn[1]} w {sw_[0]:.1f}x{sw_[1]}" if ("telemetry" in OV and sn and sw_) else ""
    live_lab = f"feather {a.feather:g} px, wide gain {g:.2f}" + (f" U {du:+.1f} V {dv:+.1f} grad {match['gx']:+.2f},{match['gy']:+.2f} bands {'/'.join(f'{v:.2f}' for v in match['lut'][[33, 75, 130, 197]] / [33, 75, 130, 197])} (seam meter)" if a.meter else " (exposure model)")
    if a.compare == "exposure":
      top = np.hstack([label(fit(on_model, rd), f"comma 4 narrow, EXPOSURE MODEL only: wide gain {g_model:.2f}  |  segment {seg}  t = {t:5.1f} s"),
                       label(fit(on_bgr, rd), "comma 4 narrow, LIVE SEAM METER: " + live_lab)])
    elif a.compare == "feedforward":
      top = np.hstack([label(fit(on_model, rd), f"BEFORE: meter filters the gain itself: wide gain {match_a['gain_y']:.2f}  |  segment {seg}  t = {t:5.1f} s"),
                       label(fit(on_bgr, rd), f"AFTER: meter filters the correction, applied on the exposure model's live value ({g_model:.2f}): wide gain {g:.2f}")])
    elif a.compare == "soften":
      top = np.hstack([label(fit(on_bgr, rd), f"BEFORE: inset one sample per pixel  |  segment {seg}  t = {t:5.1f} s  |  " + live_lab),
                       label(fit(on_model, rd), f"AFTER: inset softened in the blend zone (5 taps {a.soften} px apart, fading over one feather width)")])
    if a.compare:
      bot = np.hstack([label(fit(raw_n, rs), "3X narrow, raw" + (f"  |  {tele}" if tele else "") + expo),
                       label(fit(ow_bgr, rd), "comma 4 wide, reprojected from the 3X wide" + ("  |  green: our warp crop" if "crop" in OV else ""))])
    elif a.stretch:
      top = np.hstack([label(fit(raw_w, rs), f"3X wide, raw  |  segment {seg}  t = {t:5.1f} s" + ("  |  orange: stock's 512x256 warp crop" if "crop" in OV else "")),
                       label(fit(str_w, rd), "3X wide simply resized to 1344x760, no geometry" + ("  |  magenta: comma 4 warp crop, HUD with comma 4 intrinsics" if "crop" in OV else "")),
                       label(fit(ow_bgr, rd), "comma 4 wide, reprojected from the 3X wide" + ("  |  green: our warp crop" if "crop" in OV else ""))])
      bot = np.hstack([label(fit(raw_n, rs), "3X narrow, raw" + (f"  |  {tele}" if tele else "")),
                       label(fit(str_n, rd), "3X narrow simply resized to 1344x760, no geometry"),
                       label(fit(on_bgr, rd), "comma 4 narrow: narrow inset + wide surround, " + live_lab + expo)])
    else:
      top = np.hstack([label(fit(raw_w, rs), f"3X wide, raw  |  segment {seg}  t = {t:5.1f} s" + ("  |  orange: stock's 512x256 warp crop" if "crop" in OV else "")),
                       label(fit(ow_bgr, rd), "comma 4 wide, reprojected from the 3X wide" + ("  |  green: our warp crop" if "crop" in OV else ""))])
      bot = np.hstack([label(fit(raw_n, rs), "3X narrow, raw" + (f"  |  {tele}" if tele else "")),
                       label(fit(on_bgr, rd), "comma 4 narrow: narrow inset + wide surround, " + live_lab + expo)])
    frame = [top, bot]
    if inputs is not None:
      row = np.zeros((IH + LH, NCOL * COL, 3), np.uint8)
      names = ["stock model input: narrow (3X warp)", "stock: wide (3X warp)"] + (["resized: narrow (comma 4 warp)", "resized: wide (comma 4 warp)"] if a.stretch else []) + ["ours: narrow (comma 4 warp)", "ours: wide (comma 4 warp)"]
      for k, (im, nm) in enumerate(zip(inputs, names)):
        x0 = (k % 2) * (IW + 10) + (k // 2) * COL
        cv2.putText(row, nm, (x0 + 4, 20), FONT, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        row[LH:LH + IH, x0:x0 + IW] = cv2.resize(im, (IW, IH), interpolation=cv2.INTER_AREA)
      frame.append(row)
    out = np.vstack(frame)
    if "fit" in OV:
      out = np.hstack([out, fit_panel(comp_clean, wide_np, narrow_np, match, g_model, fit_hist)])
    enc.stdin.write(out.tobytes())
    n += 1; total += 1
  for d in dec:
    d.kill()
  print(f"seg {seg}: {n} frames, {total / (time.time() - t_start):.1f} fps overall, calibration rpy {np.round(rpy, 4).tolist()}", flush=True)
enc.stdin.close(); enc.wait()
print(f"wrote {a.out}: {total} frames, {os.path.getsize(a.out) / 1e6:.0f} MB")
