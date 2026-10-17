"""The route input's layouts: which floats of the encoder's vector (route input v2, ROUTE_LEN = 223 floats) each version a
route-conditioned driving model reads, in order, and where its parts are. Whoever knows the route writes the whole vector
(the GTA V bridge's encoder, gta5/navd/route_input.py, which says what each float holds), modeld reads its model's
layout of it (route_input.py), and gta5-train trains and labels with the same ones:
- v1 (173): the start of it, [0:173];
- v2 (223): all of it;
- v3 (218): v2 without NEXT's lane fields 166-170, which LANES supersedes.
Each version's width is its own, so a width alone names its layout."""
from typing import NamedTuple

import numpy as np

HEADING = slice(151, 161)
NEXT = slice(161, 171)
STOP = slice(171, 173)
V1_LEN = 173
LANE_SLOTS_LEN = 50  # LANES: the lane slots (the encoder's lane_slots.py), since v2
LANES = slice(V1_LEN, V1_LEN + LANE_SLOTS_LEN)
ROUTE_LEN = LANES.stop
NEXT_LANES = slice(NEXT.start + 5, NEXT.stop)  # NEXT's lane fields (v1, v2), which v3 leaves out


class Layout(NamedTuple):
  """One version of the route input: which of the encoder's floats it holds, in order, and where its parts are."""
  version: int
  take: np.ndarray  # [width] indices into the encoder's vector
  heading: slice
  next: slice
  stop: slice
  lanes: slice  # empty without lane slots (v1)

  @property
  def width(self) -> int:
    return len(self.take)


def _layout(version: int, keep: np.ndarray) -> Layout:
  pos = {int(k): i for i, k in enumerate(keep)}

  def part(sl: slice) -> slice:
    idx = [pos[k] for k in range(sl.start, sl.stop) if k in pos]
    return slice(idx[0], idx[-1] + 1) if idx else slice(len(keep), len(keep))
  return Layout(version, keep, part(HEADING), part(NEXT), part(STOP), part(LANES))


LAYOUTS = {1: _layout(1, np.arange(V1_LEN)), 2: _layout(2, np.arange(ROUTE_LEN)),
           3: _layout(3, np.delete(np.arange(ROUTE_LEN), np.arange(NEXT_LANES.start, NEXT_LANES.stop)))}
LATEST = 3


def layout(version: int | str | None = None, width: int | None = None) -> Layout:
  """A layout by its version, else by its width; a version that disagrees with the width is an error."""
  if version not in (None, "", 0):
    lay = LAYOUTS.get(int(version))
    if lay is None:
      raise ValueError(f"no route input layout v{version}")
    if width is not None and width != lay.width:
      raise ValueError(f"route input v{version} is {lay.width} floats, not {width}")
    return lay
  lay = next((v for v in LAYOUTS.values() if v.width == width), None)
  if lay is None:
    raise ValueError(f"no route input layout is {width} floats wide")
  return lay


def convert(vec: np.ndarray, version: int) -> np.ndarray:
  """A route input (any layout, by its width; batched on the last axis) in layout `version`. v1 has no lane slots, so
  they are zero (no lanes) from it; v3 has no NEXT lane fields, so it can't become v1 or v2."""
  vec = np.asarray(vec, np.float32)
  src, dst = layout(width=vec.shape[-1]), LAYOUTS[version]
  if src.version == dst.version:
    return vec.copy()
  if src.version == 3 and dst.version in (1, 2):
    raise ValueError(f"route input v3 has no NEXT lane fields for a v{dst.version} model")
  full = np.zeros(vec.shape[:-1] + (ROUTE_LEN,), np.float32)
  full[..., src.take] = vec
  return full[..., dst.take]


assert NEXT.stop == STOP.start and STOP.stop == V1_LEN == LANES.start
assert [v.width for v in LAYOUTS.values()] == [V1_LEN, ROUTE_LEN, ROUTE_LEN - 5] and LAYOUTS[3].lanes.stop == LAYOUTS[3].width
