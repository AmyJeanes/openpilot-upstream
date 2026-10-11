"""The route input of a route-conditioned driving model, from whoever knows the route (the GTA V bridge) to modeld, in
a shared-memory file: uint32 seq (odd while being written), uint32 n, float64 CLOCK_MONOTONIC time, float32[n].
Plain mmap, as multiprocessing.shared_memory registers a reader's attach with the resource tracker, which unlinks it.
The writer writes the encoder's whole vector (selfdrive/navd/route_input.py, route input v2); a reader takes the floats
of its model's layout from it (`take`), so v1 reads the start, v3 leaves NEXT's lane fields out.
modeld records the layout its model reads beside the file (`write_model_layout`) for the route viewer and e2e."""
import json
import mmap
import os
import struct
import time

import numpy as np

from openpilot.common.hardware.hw import Paths
from openpilot.selfdrive.navd.route_input import Layout, layout

HEADER = struct.Struct('<IId')
MAX_AGE = 1.0  # s: an older input means no route


def route_input_path() -> str:
  return os.path.join(Paths.shm_path(), "route_input" + os.environ.get("OPENPILOT_PREFIX", ""))


class RouteInputWriter:
  def __init__(self, n: int, path: str | None = None):
    self.n = n
    size = HEADER.size + 4 * n
    fd = os.open(path or route_input_path(), os.O_RDWR | os.O_CREAT, 0o644)
    try:
      if os.fstat(fd).st_size < size:  # never shrink it: a reader's mapping past the end would fault
        os.ftruncate(fd, size)
      self.mm = mmap.mmap(fd, size)
    finally:
      os.close(fd)
    self.seq = HEADER.unpack_from(self.mm)[0] & ~1  # carry on from a previous writer, even if it died mid-write

  def write(self, vec: np.ndarray):
    vec = np.asarray(vec, dtype=np.float32)
    assert vec.shape == (self.n,)
    struct.pack_into('<I', self.mm, 0, self.seq + 1)
    struct.pack_into('<Id', self.mm, 4, self.n, time.monotonic())
    self.mm[HEADER.size:HEADER.size + vec.nbytes] = vec.tobytes()
    self.seq = (self.seq + 2) & 0xFFFFFFFF
    struct.pack_into('<I', self.mm, 0, self.seq)


def model_layout(width: int, version: str | int | None = None) -> Layout:
  """The route input layout of a model whose route input is `width` floats, by its route_version metadata if it has
  one (gta5-train's export writes it), else by the width."""
  return layout(version, width)


def model_layout_path() -> str:
  return route_input_path() + ".model"


def write_model_layout(lay: Layout | None, path: str | None = None):
  """What modeld's model reads, for the viewer and e2e: {"version", "width"}, or {} for a model without a route input."""
  path = path or model_layout_path()
  with open(path + ".tmp", "w") as f:
    json.dump({"version": lay.version, "width": lay.width} if lay is not None else {}, f)
  os.replace(path + ".tmp", path)


def read_model_layout(path: str | None = None) -> dict | None:
  """write_model_layout's record, None without one."""
  try:
    with open(path or model_layout_path()) as f:
      return json.load(f)
  except (OSError, ValueError):
    return None


class RouteInputReader:
  """read() is the newest input (its first n floats, or with `take` those floats of it in that order), or zeros (no
  route) when there is none, it's shorter than it needs or it's stale. A torn read gives the last good input."""
  def __init__(self, n: int, path: str | None = None, take: np.ndarray | None = None):
    self.n = n
    self.take = None if take is None else np.asarray(take, np.int64)
    assert self.take is None or len(self.take) == n
    self.src_n = n if self.take is None else int(self.take.max()) + 1  # floats of the written vector it reads
    self.path = path or route_input_path()
    self.mm: mmap.mmap | None = None
    self.ino = 0
    self.zeros = np.zeros(n, dtype=np.float32)
    self.last, self.last_t = self.zeros, 0.0

  @classmethod
  def for_layout(cls, lay: Layout, path: str | None = None) -> "RouteInputReader":
    return cls(lay.width, path, lay.take)

  def _open(self) -> bool:
    try:
      fd = os.open(self.path, os.O_RDONLY)
    except OSError:
      return False
    try:
      st = os.fstat(fd)
      if st.st_size < HEADER.size + 4 * self.src_n:
        return False
      self.mm, self.ino = mmap.mmap(fd, st.st_size, access=mmap.ACCESS_READ), st.st_ino
      return True
    finally:
      os.close(fd)

  def _replaced(self) -> bool:
    try:
      return os.stat(self.path).st_ino != self.ino
    except OSError:
      return True

  def read(self) -> np.ndarray:
    if self.mm is None and not self._open():
      return self.zeros
    now = time.monotonic()
    seq, n, t = HEADER.unpack_from(self.mm)
    if n < self.src_n:
      return self.zeros
    vec = np.frombuffer(self.mm[HEADER.size:HEADER.size + 4 * self.src_n], dtype=np.float32)
    if self.take is not None:
      vec = vec[self.take]
    if seq & 1 or struct.unpack_from('<I', self.mm)[0] != seq:
      return self.last if now - self.last_t <= MAX_AGE else self.zeros
    if now - t > MAX_AGE:
      if self._replaced():  # a writer that started afresh on a new file
        self.mm = None
      return self.zeros
    self.last, self.last_t = vec, t
    return vec
