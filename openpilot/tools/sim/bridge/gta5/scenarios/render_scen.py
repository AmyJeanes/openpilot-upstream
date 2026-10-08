"""Scenario figures from verify_<name>.json: per trip a context panel (whole route) and a zoom on the key maneuver with
the map's lane markings; E trips add the height profile. Index sheet of all trips. argv: verify json(s), out dir."""
import json
import math
import os
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Polygon

HOME = os.path.expanduser("~")
SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, GREEN, RED, VIOLET = "#2a78d6", "#eb6834", "#008300", "#e34948", "#4a3aa7"
HALO = [pe.withStroke(linewidth=3, foreground=SURF)]
R_COL = ["#8c8c8c", "#969696", "#a3a3a3", "#b0b0b0", "#bcbcbc", "#c9c9c9", "#c9c9c9", "#dcdcdc", "#e2e2e2"]
R_LW = [3.2, 3.0, 2.6, 2.3, 2.0, 1.6, 1.6, 0.9, 0.8]
DASH = (0, (4, 4))
DOT = (0, (1, 2))
LANE_STYLE = {"edge": ("#555", 0.9, "-"), "dashed": ("#888", 0.7, DASH), "solid": ("#666", 0.9, "-"),
              "centre": ("#c9a400", 1.1, "-"), "centre_dashed": ("#c9a400", 0.9, DASH), "stop": ("#111", 2.0, "-"),
              "give_way": ("#111", 1.2, (0, (1, 1))), "crossing": ("#bbb", 0.6, "-"), "guide_left": ("#999", 0.5, DOT),
              "guide_through": ("#999", 0.5, DOT), "guide_right": ("#999", 0.5, DOT), "parking": ("#bbb", 0.5, "-")}
plt.rcParams.update({"font.size": 10, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK})

roads = json.load(open(f"{HOME}/gta5map_lanes/roads.json"))
W_CLS = np.array([w[0] for w in roads["ways"]])
W_PTS = [np.array(w[3:], float).reshape(-1, 2) for w in roads["ways"]]
W_BOX = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in W_PTS])
SIG = np.array(roads["signals"], float).reshape(-1, 2) if roads["signals"] else np.zeros((0, 2))
lanes = json.load(open(f"{HOME}/gta5map_lanes/lanes.json"))
LK = lanes["kinds"]
L_PTS = [(LK[ln[0]], np.array(ln[1:], float).reshape(-1, 2)) for ln in lanes["lines"]]
L_BOX = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for _, p in L_PTS])


def inbox(B, box):
  x0, y0, x1, y1 = box
  return np.flatnonzero((B[:, 2] >= x0) & (B[:, 0] <= x1) & (B[:, 3] >= y0) & (B[:, 1] <= y1))


def setbox(ax, box):
  ax.set_xlim(box[0], box[2])
  ax.set_ylim(box[1], box[3])
  ax.set_aspect("equal")
  ax.set_xticks([])
  ax.set_yticks([])


def draw_roads(ax, box, scale=1.0):
  sel = inbox(W_BOX, box)
  for c in range(len(R_COL) - 1, -1, -1):
    segs = [W_PTS[i] for i in sel if W_CLS[i] == c]
    if segs:
      ax.add_collection(LineCollection(segs, colors=R_COL[c], linewidths=R_LW[c] * scale, capstyle="round", zorder=1 + (8 - c) * 0.01))
  setbox(ax, box)


def draw_lanes(ax, box):
  sel = inbox(L_BOX, box)
  by = {}
  for i in sel:
    k, p = L_PTS[i]
    by.setdefault(k, []).append(p)
  for k, segs in by.items():
    c, lw, ls = LANE_STYLE.get(k, ("#999", 0.5, "-"))
    ax.add_collection(LineCollection(segs, colors=c, linewidths=lw, linestyles=[ls], zorder=3))
  s = SIG[(SIG[:, 0] > box[0]) & (SIG[:, 0] < box[2]) & (SIG[:, 1] > box[1]) & (SIG[:, 1] < box[3])]
  if len(s):
    ax.plot(s[:, 0], s[:, 1], "o", ms=5, mfc="#2ecc40", mec=INK, mew=0.6, zorder=6)
  setbox(ax, box)


def square(pts, least=400.0, margin=60.0):
  pts = np.asarray(pts, float)
  lo, hi = pts.min(0) - margin, pts.max(0) + margin
  c = (lo + hi) / 2
  h = max(least / 2, *(hi - lo) / 2)
  return (c[0] - h, c[1] - h, c[0] + h, c[1] + h)


def scalebar(ax, box):
  x0, y0, x1, y1 = box
  w = x1 - x0
  m = 20 if w < 150 else 50 if w < 320 else 100 if w < 900 else 500 if w < 4000 else 1000
  bx, by = x0 + 0.04 * w, y0 + 0.04 * w
  ax.plot([bx, bx + m], [by, by], color=INK, lw=2.5, solid_capstyle="butt", zorder=20)
  ax.text(bx + m / 2, by + 0.012 * w, f"{m} m", ha="center", va="bottom", fontsize=8, zorder=20, path_effects=HALO)


def start_mark(ax, x, y, heading, size):
  b = math.radians((-heading) % 360)
  d, n = np.array([math.sin(b), math.cos(b)]), np.array([math.cos(b), -math.sin(b)])
  p = np.array([x, y])
  ax.add_patch(Polygon([p + d * size * 0.7, p - d * size * 0.3 + n * size * 0.26, p - d * size * 0.3 - n * size * 0.26],
                       closed=True, fc=GREEN, ec=SURF, lw=1.2, zorder=12))


def route_line(ax, pts, lw=3.0, colour=BLUE, alpha=0.85):
  ax.plot(pts[:, 0], pts[:, 1], color=colour, lw=lw, alpha=alpha, solid_capstyle="round", zorder=10,
          path_effects=[pe.Stroke(linewidth=lw + 2, foreground=SURF), pe.Normal()])
  along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
  for s in np.linspace(along[-1] * 0.1, along[-1] * 0.9, 6):
    k = min(int(np.searchsorted(along, s)), len(pts) - 1)
    if k == 0:
      continue
    ax.annotate("", pts[k], pts[k - 1], arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 0, "mutation_scale": 12}, zorder=11)


def maneuvers(ax, rec, box, numbered=False):
  real = [m for m in rec["route"]["maneuvers"] if m["real"]]
  for i, m in enumerate(real):
    x, y = m["xy"]
    if box[0] < x < box[2] and box[1] < y < box[3]:
      ax.plot(x, y, "D", ms=6, mfc=ORANGE, mec=SURF, mew=1, zorder=13)
      lab = f"{i + 1}" if numbered else m["kind"]
      ax.annotate(lab, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8 if numbered else 9, color=INK, path_effects=HALO, zorder=14)


def panel_text(rec):
  c = rec["comment"]
  if rec.get("cat"):
    c = c.replace(f"[{rec['cat']}] ", "")
  parts = [p.strip() for p in c.split(";")]
  desc = [p for p in parts if not p.startswith("expect:") and not p.startswith("success:")]
  succ = [p for p in parts if p.startswith("success:")]
  lines = textwrap.fill("; ".join(desc), 110).split("\n")
  for s in succ:
    lines += textwrap.fill(s, 110).split("\n")
  r = rec.get("route") or {}
  meta = f"route {r.get('length', '?')} m"
  if rec.get("key_s") is not None:
    meta += f", key maneuver {rec['key_s']:.0f} m from the start"
  if rec.get("dtrain"):
    meta += f"; training routes >= {min(rec['dtrain'])} m, training junctions >= {min(rec['dtrain_junction'])} m"
  if rec.get("opts"):
    meta += "; options " + " ".join(rec["opts"])
  meta += "; router check " + ("ok" if rec.get("ok") else "FAIL: " + "; ".join(rec.get("why", [])))
  return "\n".join(lines + textwrap.fill(meta, 110).split("\n"))


def trip_figure(rec, out):
  r = rec["route"]
  pts = np.array(r["points"])
  sx, sy, sz, sh = [float(v) for v in rec["spec"].split(">")[0].split(",")[:4]]
  dx, dy = [float(v) for v in rec["spec"].split(">")[1].split(",")]
  exps = rec["expect"]
  key = np.array(exps[0][1:]) if exps else pts[len(pts) // 2]
  long_ = r["length"] > 1500 or rec["cat"] == "G1"
  prof = rec["cat"] in ("E1", "E2")
  ncol = 3 if prof else 2
  fig = plt.figure(figsize=(7.2 * ncol, 9.2))
  ax0 = fig.add_axes((0.01, 0.22, 0.98 / ncol - 0.01, 0.70))
  ax1 = fig.add_axes((0.98 / ncol + 0.01, 0.22, 0.98 / ncol - 0.01, 0.70))
  box0 = square(np.vstack([pts, [[sx, sy], [dx, dy]]]), least=500)
  draw_roads(ax0, box0, 0.6 if box0[2] - box0[0] > 2000 else 1.0)
  route_line(ax0, pts, 2.6)
  maneuvers(ax0, rec, box0, numbered=long_)
  for _k, x, y in exps:
    ax0.add_patch(Circle((x, y), 0.035 * (box0[2] - box0[0]), fc="none", ec=RED, lw=2, ls=(0, (3, 2)), zorder=12))
  start_mark(ax0, sx, sy, sh, 0.05 * (box0[2] - box0[0]))
  ax0.plot(dx, dy, "s", ms=10, mfc=INK, mec=SURF, mew=1.5, zorder=12)
  scalebar(ax0, box0)
  ax0.set_title("whole route" + (" (real maneuvers numbered)" if long_ else ""), loc="left", fontsize=11, color=INK2)
  half = 110 if rec["cat"] not in ("C1", "C2", "E1", "E2", "G1") else 220
  zpts = [key] + [np.array(e[1:]) for e in exps[1:2] if math.hypot(e[1] - key[0], e[2] - key[1]) < 180]
  c = np.mean(zpts, axis=0)
  box1 = (c[0] - half, c[1] - half, c[0] + half, c[1] + half)
  draw_roads(ax1, box1, 2.2)
  draw_lanes(ax1, box1)
  route_line(ax1, pts, 3.2, alpha=0.55)
  maneuvers(ax1, rec, box1)
  for k, x, y in exps:
    if box1[0] < x < box1[2] and box1[1] < y < box1[3]:
      ax1.add_patch(Circle((x, y), 14, fc="none", ec=RED, lw=2.2, ls=(0, (3, 2)), zorder=12))
      ax1.annotate(k.replace("_", " "), (x, y), xytext=(-10, -22), textcoords="offset points", fontsize=10, color=RED,
                   fontweight="bold", path_effects=HALO, zorder=14)
  if box1[0] < sx < box1[2] and box1[1] < sy < box1[3]:
    start_mark(ax1, sx, sy, sh, 14)
  scalebar(ax1, box1)
  ax1.set_title("key maneuver, lane markings from the live map (green dots: signals)", loc="left", fontsize=11, color=INK2)
  if prof:
    ax2 = fig.add_axes((2 * 0.98 / ncol + 0.04, 0.28, 0.98 / ncol - 0.06, 0.60))
    z = rec.get("z")
    if z:
      z = np.array(z)
      ax2.plot(z[:, 0], z[:, 1], color=VIOLET, lw=2)
      ax2.set_xlabel("m along the route")
      ax2.set_ylabel("height (m)")
      ax2.grid(lw=0.3)
      g = rec.get("grade_max")
      ax2.set_title(f"height profile (steepest 150 m: {g:.0f}%)" if g is not None else "height profile", loc="left", fontsize=11, color=INK2)
  fig.text(0.01, 0.975, f"{rec['name']}  [{rec['cat']}]", fontsize=17, fontweight="bold", va="top")
  fig.text(0.01, 0.20, panel_text(rec), fontsize=10, va="top", color=INK2)
  fig.savefig(out, dpi=95)
  plt.close(fig)


def index_sheet(recs, out, title):
  n = len(recs)
  cols = 6
  rows = math.ceil(n / cols)
  fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 4.2), squeeze=False)
  for ax, rec in zip(np.ravel(axs), recs, strict=False):
    r = rec.get("route")
    if not r:
      ax.axis("off")
      continue
    pts = np.array(r["points"])
    box = square(pts, least=400)
    draw_roads(ax, box, 0.5)
    route_line(ax, pts, 2.0)
    for _k, x, y in rec["expect"]:
      ax.add_patch(Circle((x, y), 0.05 * (box[2] - box[0]), fc="none", ec=RED, lw=1.6, zorder=12))
    sx, sy, sz, sh = [float(v) for v in rec["spec"].split(">")[0].split(",")[:4]]
    start_mark(ax, sx, sy, sh, 0.07 * (box[2] - box[0]))
    dx, dy = [float(v) for v in rec["spec"].split(">")[1].split(",")]
    ax.plot(dx, dy, "s", ms=6, mfc=INK, mec=SURF, zorder=12)
    ok = "" if rec.get("ok") else "  (check FAIL)"
    ax.set_title(f"{rec['name']} [{rec['cat']}]{ok}", fontsize=10, loc="left", fontweight="bold", color=INK if rec.get("ok") else RED)
    first = rec["comment"].replace(f"[{rec['cat']}] ", "").split(";")[0]
    opts = (" | " + " ".join(rec["opts"])) if rec.get("opts") else ""
    ax.text(0, -0.02, textwrap.fill(first, 48) + f"\n{r['length']} m{opts}", transform=ax.transAxes, va="top", fontsize=7.5, color=INK2)
  for ax in np.ravel(axs)[n:]:
    ax.axis("off")
  fig.suptitle(title, fontsize=14, x=0.01, ha="left")
  fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=4)
  fig.savefig(out, dpi=85)
  plt.close(fig)


if __name__ == "__main__":
  *srcs, outdir = sys.argv[1:]
  os.makedirs(outdir, exist_ok=True)
  for s in srcs:
    recs = json.load(open(s))
    for rec in recs:
      if rec.get("route"):
        trip_figure(rec, os.path.join(outdir, f"{rec['name']}.png"))
    base = os.path.splitext(os.path.basename(s))[0].replace("verify_", "")
    index_sheet(recs, os.path.join(outdir, f"index_{base}.png"),
                f"{base} (2026-10-08): routes on the live router; red circles = key maneuvers; green = start, black square = destination")
    print(len(recs), "figures from", s, "->", outdir)
