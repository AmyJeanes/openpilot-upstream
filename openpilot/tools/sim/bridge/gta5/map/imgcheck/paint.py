"""Road paint, road surface and kerbs in a top-down tile, by colour and local contrast (CPU, numpy only).

Paint is what stands out brighter than its surroundings in a narrow band: the brightness less its morphological
opening (a min then max filter over a square OPEN_M wide) is a top-hat that keeps lines, arrows' strokes and stop bars
and drops pavements, kerbs with pavement beyond and anything else wider; it is relative, so a line in a shadow still
shows. White is unsaturated and bright, yellow has red and green well over blue. Thresholds were tuned on the map
audit's survey shots (2560x1440 at 45 m, analysed at half size, about 6 cm a pixel)."""
import numpy as np

OPEN_M = 0.7  # m: the opening's square, wider than any line (stop bars up to 0.6 m)
TOPHAT_MIN = 40.0  # brightness levels above the opening
TOPHAT_REL = 0.35  # and this share of the opening's brightness, so bright concrete's texture isn't paint
BRIGHT_MIN = 120.0  # a white line's brightness at least (shadows included)
YELLOW_LIFT = 15.0
YELLOW_WARM = 0.13
WHITE_WARM = 0.07
SPECK_M = 0.5  # m: paint fills 8% of a square this wide about it at least, more than grit does
LOCAL_M = 2.0  # m: the square paint must stand out of for paint the map is said to miss
STRONG = 2.2  # standard deviations


def window(a: np.ndarray, k: int, fn, axis: int) -> np.ndarray:
  pad = [(0, 0)] * a.ndim
  pad[axis] = (k // 2, k // 2)
  return fn(np.lib.stride_tricks.sliding_window_view(np.pad(a, pad, mode="edge"), k, axis=axis), axis=-1)


def min_filter(a: np.ndarray, k: int) -> np.ndarray:
  return window(window(a, k, np.min, 0), k, np.min, 1)


def max_filter(a: np.ndarray, k: int) -> np.ndarray:
  return window(window(a, k, np.max, 0), k, np.max, 1)


def dilate(mask: np.ndarray, r_px: int) -> np.ndarray:
  """A square dilation by r_px each way."""
  if r_px <= 0:
    return mask
  return max_filter(mask.view(np.uint8), 2 * r_px + 1).astype(bool)


def box_sum(a: np.ndarray, k: int) -> np.ndarray:
  """The sum (count, for a mask) over the k x k square about each pixel."""
  a = a.astype(np.int32) if a.dtype == bool else a.astype(np.float64)
  c = np.pad(a, ((k // 2 + 1, k // 2), (k // 2 + 1, k // 2)), mode="edge" if a.dtype != np.int32 else "constant").cumsum(0).cumsum(1)
  return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def downscale(img: np.ndarray, factor: int) -> np.ndarray:
  if factor <= 1:
    return img
  h, w = img.shape[0] // factor * factor, img.shape[1] // factor * factor
  return img[:h, :w].reshape(h // factor, factor, w // factor, factor, -1).mean(axis=(1, 3)).astype(np.float32)


class Paint:
  """Masks of a tile image [h, w, 3] (float RGB) at m_per_px."""

  def __init__(self, img: np.ndarray, m_per_px: float):
    rgb = img.astype(np.float32)
    self.rgb = rgb
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    v = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    self.v = v
    k = max(3, int(round(OPEN_M / m_per_px)) | 1)
    self.opened = max_filter(min_filter(v, k), k)
    tophat = v - self.opened
    self.tophat = tophat
    lifted = (tophat > TOPHAT_MIN) & (tophat > TOPHAT_REL * self.opened)
    # yellow is told by its colour, so it needs less lift: on pale concrete it is hardly brighter than the road
    lifted_y = (tophat > YELLOW_LIFT) & (tophat > 0.1 * self.opened)
    # warmth (red over blue) as a share of red, less the road's own (the grade's tint, warm concrete): on the survey
    # shots white paint is under 0.07 in 90% of pixels, yellow over 0.16 in 75% (worn yellow lower)
    sat = (v - mn) / np.maximum(v, 1.0)
    grey = (sat < 0.2) & (v > 22) & (v < 215)
    bias = float(np.median((r - b)[grey])) if grey.any() else 0.0
    warm = (r - b - bias) / np.maximum(r, 1.0)
    self.yellow = lifted_y & (warm > YELLOW_WARM) & (r - b - bias > 15) & (g > 0.6 * r) & (r > 90)
    self.white = lifted & ~self.yellow & (warm < WHITE_WARM) & (v - mn < 0.22 * v + 12) & (v > BRIGHT_MIN)
    # paint of neither colour for sure (worn yellow, tinted white): it counts as paint, not as either colour
    self.unsure = lifted_y & ~self.yellow & ~self.white & (warm >= WHITE_WARM) & (g > 0.6 * r) & (v > 90)
    k2 = max(3, int(round(SPECK_M / m_per_px)) | 1)
    need = max(4, int(0.08 * k2 * k2))
    self.white &= box_sum(self.white, k2) >= need
    self.yellow &= box_sum(self.yellow, k2) >= need
    self.unsure &= box_sum(self.unsure | self.white | self.yellow, k2) >= need
    self.paint = self.white | self.yellow | self.unsure
    # paint that stands out of the road's own texture around it (worn concrete has bright streaks along the wheel
    # tracks): this many standard deviations over the mean of a LOCAL_M square; a line fills little of the square
    k3 = max(5, int(round(LOCAL_M / m_per_px)) | 1)
    n = float(k3 * k3)
    mean = box_sum(v, k3) / n
    std = np.sqrt(np.maximum(box_sum(v.astype(np.float64) ** 2, k3) / n - mean ** 2, 1.0))
    self.contrast = ((v - mean) / std).astype(np.float32)
    self.strong = self.paint & (self.contrast > STRONG)
    vegetation = (g > r + 6) & (g > b + 6) & (sat > 0.12)
    # road surface as seen: unsaturated grey of asphalt or concrete, in sun or shade, or paint
    self.asphalt = ((sat < 0.2) & (v > 22) & (v < 215) & ~vegetation) | self.paint
    self.vegetation = vegetation
    self.blank = float(np.std(v)) < 6.0

  def sharpness(self) -> float:
    """Mean absolute Laplacian of the brightness: low for an unstreamed (LOD) or blank frame."""
    v = self.v
    lap = np.abs(4 * v[1:-1, 1:-1] - v[:-2, 1:-1] - v[2:, 1:-1] - v[1:-1, :-2] - v[1:-1, 2:])
    return float(lap.mean())
