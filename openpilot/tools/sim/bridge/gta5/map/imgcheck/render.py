"""Pictures of a checked tile: our map drawn as the overlay colours it, alone and over the game, and the issues."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from openpilot.tools.sim.bridge.gta5.map.imgcheck.compare import TileCheck

# as the overlay draws them (README's layer table)
COLOURS = {"e": (40, 230, 70), "d": (255, 255, 255), "w": (255, 255, 255), "p": (190, 90, 255), "c": (255, 200, 0),
           "y": (255, 200, 0), "l": (255, 40, 40), "s": (255, 140, 0), "k": (255, 140, 0), "x": (230, 230, 230),
           "L": (60, 140, 255), "T": (255, 255, 255), "R": (255, 140, 0), "t": (0, 230, 230), "j": (70, 110, 255)}
WIDTHS = {"e": 0.35, "l": 0.6, "s": 0.6, "k": 0.6, "x": 0.5, "t": 1.2, "j": 0.15}
ISSUE_COLOURS = {"missing": (255, 0, 255), "stray": (0, 255, 255), "colour": (255, 128, 0), "kerb": (255, 40, 40),
                 "stop": (255, 255, 0), "junction": (120, 160, 255)}


def font(size: int):
  try:
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
  except OSError:
    return ImageFont.load_default()


def draw_map(tc: TileCheck, base: Image.Image, scale: float = 1.0, dashed: bool = True) -> Image.Image:
  """Our map's marks drawn on base (the analysed image's size times scale), at their widths, dashes 3 m on 6 m off."""
  im = base.convert("RGB")
  dr = ImageDraw.Draw(im)
  cam = tc.cam
  kinds = "jexpdwcylskLTRt"
  items = tc.map.lines(kinds, level_only=False)
  items.sort(key=lambda it: kinds.index(it[0]))
  for kind, pts in items:
    if kind == "j":
      uv = cam.project(pts) * scale
      dr.line([tuple(p) for p in uv], fill=COLOURS["j"], width=max(1, int(2 * scale)))
      continue
    wpx = max(1, int(round(WIDTHS.get(kind, 0.2) / cam.m_per_px * scale)))
    if dashed and kind in "dyk":
      from openpilot.tools.sim.bridge.gta5.map.imgcheck.compare import resample
      q, _ = resample(pts, 0.5)
      on = (np.arange(len(q)) * 0.5) % 9.0 < 3.0
      uv = cam.project(q) * scale
      for i in range(len(q) - 1):
        if on[i]:
          dr.line([tuple(uv[i]), tuple(uv[i + 1])], fill=COLOURS[kind], width=wpx)
      continue
    uv = cam.project(pts) * scale
    dr.line([tuple(p) for p in uv], fill=COLOURS.get(kind, (255, 255, 255)), width=wpx, joint="curve")
  return im


def masks_view(tc: TileCheck) -> Image.Image:
  """The game dimmed, its paint as found (white, yellow), and what of it the map doesn't account for (magenta)."""
  v = tc.img * 0.45
  p = tc.paint
  v[p.white] = (235, 235, 235)
  v[p.yellow] = (255, 210, 0)
  if hasattr(tc, "unexplained"):
    v[tc.unexplained] = (255, 0, 255)
  v[tc.hidden] = v[tc.hidden] * 0.4 + np.array([0, 0, 120])
  return Image.fromarray(v.clip(0, 255).astype(np.uint8))


def draw_issues(im: Image.Image, tc: TileCheck, scale: float = 1.0, numbers: bool = True) -> Image.Image:
  dr = ImageDraw.Draw(im)
  f = font(max(12, int(14 * scale)))
  for n, i in enumerate(tc.issues):
    c = ISSUE_COLOURS[i.kind]
    r = max(8, i.length / tc.mpp * scale / 2) if i.kind != "stop" else 14 * scale
    r = min(r, 200 * scale)
    u, v = i.u * scale, i.v * scale
    dr.ellipse([u - r, v - r, u + r, v + r], outline=c, width=max(2, int(2 * scale)))
    if numbers:
      dr.text((u + r + 2, v - 8), f"{n} {i.kind}", fill=c, font=f)
  return im


def sheet(tc: TileCheck, title: str = "") -> Image.Image:
  """Game | our map over the game | paint found with the issues, stacked."""
  game = Image.fromarray(tc.img.clip(0, 255).astype(np.uint8))
  over = draw_map(tc, game)
  m = draw_issues(masks_view(tc), tc)
  w, h = game.size
  out = Image.new("RGB", (w, 3 * h + 30), (20, 20, 20))
  out.paste(game, (0, 30))
  out.paste(over, (0, 30 + h))
  out.paste(m, (0, 30 + 2 * h))
  ImageDraw.Draw(out).text((8, 6), title, fill=(255, 255, 255), font=font(16))
  return out
