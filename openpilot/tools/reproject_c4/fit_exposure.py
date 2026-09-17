"""Fit the constant part of the composite's exposure match from a real drive.

For every 20th frame of each segment the wide (ecamera) and narrow (fcamera) videos are decoded at quarter resolution,
paired through the reprojection's own sample coordinates (the comma 4 narrow output pixels that see both the 3X narrow
inset and the 3X wide surround), and the per-frame luma ratio narrow/wide in the overlap is compared with the exposure
ratio (gain*integLines) from the two camera states in the rlog. ratio = K * exposure_ratio ** gamma; K is the constant
the kernel needs (reproject_c4 applies gain_y to the wide surround in the encoded domain).
OUT=... python openpilot/tools/reproject_c4/fit_exposure.py <route dir> [step]"""
import glob
import os
import subprocess
import sys

import numpy as np
FFMPEG = os.environ.get("FFMPEG", "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg")  # the venv ships a pipe-less ffmpeg
from openpilot.tools.lib.logreader import LogReader
try:
  from openpilot.selfdrive.modeld import reproject_c4 as RC  # openpilot branch
except ImportError:
  from openpilot.sunnypilot.modeld_v2 import reproject_c4 as RC  # sunnypilot tree

root = sys.argv[1]; STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SW, SH = 1928, 1208; Q = 4; qw, qh = SW // Q, SH // Q
OUT = os.environ.get("OUT", "/tmp/exposure_const"); os.makedirs(OUT, exist_ok=True)

# overlap pairs: comma 4 narrow output pixels (full res) -> 3X wide coords, 3X narrow coords; keep the well-inside-inset ones
mw, mn = RC.sample_coords("narrow", 1344, 760, 1.0)
edge = np.minimum(np.minimum(mn[..., 0], SW - mn[..., 0]), np.minimum(mn[..., 1], SH - mn[..., 1]))
inside = (mw[..., 0] > 0) & (mw[..., 1] > 0) & (mw[..., 0] < SW) & (mw[..., 1] < SH)
REGIONS = {"all": (edge > 80) & inside, "ring": (edge > 30) & (edge < 130) & inside}  # ring = where the seam blends
def _idx(mask):
  iw = (mw[mask] / Q).astype(int); inn = (mn[mask] / Q).astype(int)
  iw[:, 0] = iw[:, 0].clip(0, qw - 1); iw[:, 1] = iw[:, 1].clip(0, qh - 1); inn[:, 0] = inn[:, 0].clip(0, qw - 1); inn[:, 1] = inn[:, 1].clip(0, qh - 1)
  return iw, inn
IDX = {k: _idx(v) for k, v in REGIONS.items()}
print("overlap pixels used:", {k: int(v.sum()) for k, v in REGIONS.items()}, flush=True)


def decode(path):
  raw = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-vf", f"select='not(mod(n\\,{STEP}))',scale={qw}:{qh}", "-vsync", "0",
                        "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
  return np.frombuffer(raw, np.uint8).reshape(-1, qh, qw)


rows = []
segs = sorted(glob.glob(os.path.join(root, "*--*--*")), key=lambda p: int(p.rsplit("--", 1)[1]))
for p in segs:
  if not all(os.path.exists(os.path.join(p, f)) for f in ("rlog.zst", "fcamera.hevc", "ecamera.hevc")):
    continue
  seg = int(p.rsplit("--", 1)[1])
  exp = {"road": {}, "wide": {}}; idx = {"road": {}, "wide": {}}
  for m in LogReader(os.path.join(p, "rlog.zst")):
    w = m.which()
    if w in ("narrowRoadCameraState", "wideRoadCameraState"):
      c = getattr(m, w); exp["road" if w == "narrowRoadCameraState" else "wide"][c.frameId] = (c.gain, c.integLines, c.exposureValPercent, c.timestampSof)
    elif w in ("narrowRoadEncodeIdx", "wideRoadEncodeIdx"):
      e = getattr(m, w); idx["road" if w == "narrowRoadEncodeIdx" else "wide"][e.segmentId] = e.frameId
  try:
    fn, fw = decode(os.path.join(p, "fcamera.hevc")), decode(os.path.join(p, "ecamera.hevc"))
  except subprocess.CalledProcessError as e:
    print(f"seg {seg}: decode failed {e}", flush=True); continue
  n = min(len(fn), len(fw))
  for k in range(n):
    i = k * STEP
    fid_n, fid_w = idx["road"].get(i), idx["wide"].get(i)
    if fid_n is None or fid_w is None or fid_n not in exp["road"] or fid_w not in exp["wide"]:
      continue
    rr = {}
    for name, (iw, inn) in IDX.items():
      yn = fn[k][inn[:, 1], inn[:, 0]].astype(np.float32); yw = fw[k][iw[:, 1], iw[:, 0]].astype(np.float32)
      good = (yn > 16) & (yn < 235) & (yw > 16) & (yw < 235)
      if good.sum() < 1000:
        break
      rr[name] = (np.median(yn[good] / yw[good]), float(np.median(yn[good])), float(np.median(yw[good])))
    if len(rr) < 2:
      continue
    r = rr["all"][0]
    gn, tn, evn, tsn = exp["road"][fid_n]; gw, tw, evw, tsw = exp["wide"][fid_w]
    rows.append((seg, i, fid_n, r, gn, tn, gw, tw, evn, evw, rr["all"][1], rr["all"][2], (tsn - tsw) * 1e-6, rr["ring"][0]))
  print(f"seg {seg}: {n} pairs, {sum(1 for x in rows if x[0] == seg)} usable; last ratio {rows[-1][3]:.3f} exposure n/w {rows[-1][4] * rows[-1][5] / (rows[-1][6] * rows[-1][7]):.3f} luma n {rows[-1][10]:.0f} w {rows[-1][11]:.0f}" if rows and rows[-1][0] == seg else f"seg {seg}: {n} pairs, none usable", flush=True)

R = np.array(rows, dtype=np.float64)
np.save(os.path.join(OUT, "exposure_rows.npy"), R)
e = (R[:, 4] * R[:, 5]) / (R[:, 6] * R[:, 7]); r = R[:, 3]
K = np.exp(np.median(np.log(r) - np.log(e)))
A = np.vstack([np.ones_like(e), np.log(e)]).T; coef, *_ = np.linalg.lstsq(A, np.log(r), rcond=None)
print(f"\n{len(R)} frame pairs. exposure ratio n/w: median {np.median(e):.3f} range {e.min():.3f}..{e.max():.3f} | measured luma ratio n/w: median {np.median(r):.3f} range {r.min():.3f}..{r.max():.3f}")
print(f"K = ratio / exposure_ratio: median {K:.3f}, IQR {np.percentile(r / e, 25):.3f}..{np.percentile(r / e, 75):.3f}")
print(f"power-law fit ratio = {np.exp(coef[0]):.3f} * exposure_ratio ** {coef[1]:.3f}; residual sd {np.std(np.log(r) - A @ coef):.3f} (log)")
rr = R[:, 13]; coef_r, *_ = np.linalg.lstsq(A, np.log(rr), rcond=None)
print(f"SEAM RING: luma ratio n/w median {np.median(rr):.3f}; power-law ratio = {np.exp(coef_r[0]):.3f} * exposure_ratio ** {coef_r[1]:.3f}; residual sd {np.std(np.log(rr) - A @ coef_r):.3f}; ring/all ratio median {np.median(rr / r):.3f}")
for lo, hi, name in ((0, 60, "bright (wide luma > 60)"), (30, 60, "mid"), (0, 30, "dark (wide luma < 30)")):
  sel = (R[:, 11] >= lo) & (R[:, 11] < hi) if name != "bright (wide luma > 60)" else R[:, 11] > 60
  if sel.sum() > 10:
    print(f"  {name}: n={sel.sum()} K median {np.median(r[sel] / e[sel]):.3f}, exposure ratio median {np.median(e[sel]):.3f}")
