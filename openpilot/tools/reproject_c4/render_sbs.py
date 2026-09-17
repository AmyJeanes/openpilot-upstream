"""Side-by-side video of a recorded 3X drive: raw cameras (left) vs the same frames reprojected into comma 4 geometry
(right) by the same kernel that runs on the device, with the per-frame exposure gains the integration used (camera
states from the rlog, times the constant K). Optional overlays: the model's 512x256 warp crops (stock's on the raw 3X
frames with 3X intrinsics, ours on the comma 4 frames with comma 4 intrinsics; both should frame the same scene), the
four reconstructed model inputs, and a telemetry line from the rlog.
DEV=CUDA python openpilot/tools/reproject_c4/render_sbs.py <route dir> <out.mp4>
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
ap.add_argument("--overlays", default="crop,telemetry,hud", help="comma list of crop,inputs,telemetry,hud (or none)")
a = ap.parse_args()
OV = set(a.overlays.split(",")) - {"none", ""}
DEV = os.environ.get("DEV", Device.DEFAULT)
SW, SH, DW, DH = 1928, 1208, 1344, 760
MW, MH = MEDMODEL_INPUT_SIZE
s_stride, s_yh, s_uvh, s_size = get_nv12_info(SW, SH)
COL, LH = 960, 30
rs = (COL, round(COL * SH / SW)); rd = (COL, round(COL * DH / DW)); ROW = rs[1]
IN_S = 0.9; IW, IH = round(MW * IN_S), round(MH * IN_S); IN_ROW = (IH + LH) if "inputs" in OV else 0
H, W = 2 * (ROW + LH) + IN_ROW, 2 * COL
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


rp = RC.Reprojector(device=DEV)
enc = subprocess.Popen([FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "20", "-i", "-",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
segs = sorted(glob.glob(os.path.join(a.route, "*--*--*")), key=lambda p: int(p.rsplit("--", 1)[1]))
if a.segs != "all":
  want = {int(x) for x in a.segs.split(",")}; segs = [p for p in segs if int(p.rsplit("--", 1)[1]) in want]
total = 0; t_start = time.time(); rpy = np.zeros(3, np.float32); path_z = PATH_Z_OFF
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
  while n < a.max_frames:
    raws = [d.stdout.read(SW * SH * 3 // 2) for d in dec]
    if any(len(r) < SW * SH * 3 // 2 for r in raws):
      break
    wide_np, narrow_np = (to_layout(np.frombuffer(r, np.uint8)) for r in raws)
    fid_n, fid_w = idx["road"].get(n), idx["wide"].get(n)
    sn, sw_ = cs["road"].get(fid_n), cs["wide"].get(fid_w)
    g = float(np.clip(a.K * RC.exposure_gain(sn[0] * sn[1], sw_[0] * sw_[1]), 0.25, 4.0)) if sn and sw_ else 1.0
    ow, on = rp(Tensor(wide_np, device=DEV).realize(), Tensor(narrow_np, device=DEV).realize(), g, g)
    ow_bgr, on_bgr = from_nv12(ow.numpy(), DW, DH), from_nv12(on.numpy(), DW, DH)
    raw_w, raw_n = from_nv12(wide_np, SW, SH), from_nv12(narrow_np, SW, SH)
    inputs = [model_input(raw_n, M_x3["n"]), model_input(raw_w, M_x3["w"]), model_input(on_bgr, M_c4["n"]), model_input(ow_bgr, M_c4["w"])] if "inputs" in OV else None
    if "hud" in OV and fid_n in hud:
      o = hud[fid_n]
      draw_hud(raw_n, o, view_from_calib, X3.narrow_road.intrinsics, path_z); draw_hud(raw_w, o, view_from_calib, X3.wide_road.intrinsics, path_z)
      draw_hud(on_bgr, o, view_from_calib, C4.narrow_road.intrinsics, path_z); draw_hud(ow_bgr, o, view_from_calib, C4.wide_road.intrinsics, path_z)
    if "crop" in OV:
      draw_crop(raw_n, M_x3["n"], (0, 200, 255)); draw_crop(raw_w, M_x3["w"], (0, 200, 255)); draw_crop(on_bgr, M_c4["n"], (80, 255, 80)); draw_crop(ow_bgr, M_c4["w"], (80, 255, 80))
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
    top = np.hstack([label(fit(raw_w, rs), f"3X wide, raw  |  segment {seg}  t = {t:5.1f} s" + ("  |  orange: stock's 512x256 warp crop" if "crop" in OV else "")),
                     label(fit(ow_bgr, rd), "comma 4 wide, reprojected from the 3X wide" + ("  |  green: our warp crop" if "crop" in OV else ""))])
    bot = np.hstack([label(fit(raw_n, rs), "3X narrow, raw" + (f"  |  {tele}" if tele else "")),
                     label(fit(on_bgr, rd), f"comma 4 narrow: narrow inset + wide surround, wide gain {g:.2f}" + expo)])
    frame = [top, bot]
    if inputs is not None:
      row = np.zeros((IH + LH, W, 3), np.uint8)
      names = ["stock model input: narrow (3X warp)", "stock: wide (3X warp)", "ours: narrow (comma 4 warp)", "ours: wide (comma 4 warp)"]
      for k, (im, nm) in enumerate(zip(inputs, names)):
        x0 = (k % 2) * (IW + 10) + (k // 2) * COL
        cv2.putText(row, nm, (x0 + 4, 20), FONT, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        row[LH:LH + IH, x0:x0 + IW] = cv2.resize(im, (IW, IH), interpolation=cv2.INTER_AREA)
      frame.append(row)
    enc.stdin.write(np.vstack(frame).tobytes())
    n += 1; total += 1
  for d in dec:
    d.kill()
  print(f"seg {seg}: {n} frames, {total / (time.time() - t_start):.1f} fps overall, calibration rpy {np.round(rpy, 4).tolist()}", flush=True)
enc.stdin.close(); enc.wait()
print(f"wrote {a.out}: {total} frames, {os.path.getsize(a.out) / 1e6:.0f} MB")
