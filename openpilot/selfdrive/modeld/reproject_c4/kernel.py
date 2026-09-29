"""The GPU stage: one gather per output byte through the tables, the composite blended in the seam band."""

from tinygrad import Tensor, TinyJit
import numpy as np

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from .geometry import UV_FILL
from .tables import ALPHA_SHIFT, IDX_BITS, INVALID_BIT


class Reprojector:
  """Holds the tables `T` (load_tables) on `device` and runs the gathers. Call once per model run with the two 3X NV12 buffers."""

  def __init__(self, T, dst_wh, device):
    self.stride, yh, uvh, _ = get_nv12_info(*dst_wh)
    self.dw, self.dh = dst_wh
    self.uv_offset, self.body = self.stride * yh, self.stride * (yh + uvh)
    assert self.body == 3 * (self.body - self.uv_offset)  # NV12: the UV plane is the last third of the body
    self.device = device
    self.reload(T)
    self.plane = Tensor([0, 0, 1], dtype="uint8", device=device).realize()
    self.dst = None
    self._run = TinyJit(self._both)
  def bind(self, host_wide: np.ndarray, host_narrow: np.ndarray):
    """Write the outputs straight into these host arrays (each `body` bytes, e.g. modeld's packed frame slots) instead of
    returning device tensors: a GPU->host readback costs ~40 ms on the 3X, a direct write ~10 ms."""
    assert host_wide.nbytes == self.body and host_narrow.nbytes == self.body and host_wide.dtype == host_narrow.dtype == np.uint8
    self.dst = (Tensor.from_blob(host_wide.ctypes.data, (self.body,), dtype="uint8", device=self.device),
                Tensor.from_blob(host_narrow.ctypes.data, (self.body,), dtype="uint8", device=self.device))
    self._run = TinyJit(self._both)

  def reload(self, T):
    """Swap in the tables of another rotation between runs: they are jit inputs, so one upload and no re-capture (~1.3 s on the 3X)."""
    assert len(T["wide"]["pw"]) == self.body
    self.t = {cam: {k: Tensor(v, device=self.device).realize() for k, v in T[cam].items()} for cam in ("wide", "narrow")}

  def _chroma(self):
    return self.plane.reshape(3, 1).expand(3, self.body // 3).reshape(self.body).bool()  # a broadcast: no table read

  def _wide(self, wide, pw):
    return (pw < INVALID_BIT).where(wide[pw & IDX_BITS], self._chroma().cast("uint8") * UV_FILL)

  def _narrow(self, wide, narrow, pw, pn):
    w = (pw < INVALID_BIT).where(wide[pw & IDX_BITS].float(), self._chroma().cast("float32") * UV_FILL)
    a = ((pw >> ALPHA_SHIFT) & 0xff).float() * (1 / 255)
    # the mask looks redundant but bounds the index; without it every gather is guarded (0.8 ms on the 3X)
    return a * narrow[pn & IDX_BITS].float() + (1 - a) * w

  def _both(self, wide, narrow, pw_w, pw_n, pn):
    out_w = self._wide(wide, pw_w)
    out_n = self._narrow(wide, narrow, pw_n, pn).round().cast("uint8")
    if self.dst is not None:
      out_w, out_n = self.dst[0].assign(out_w), self.dst[1].assign(out_n)
    return out_w.realize(), out_n.realize()

  def __call__(self, wide, narrow):
    """wide/narrow: flat uint8 3X NV12 tensors (the same buffers every call, e.g. the VisionIPC ring, so the jit replays).
    Returns (c4_wide, c4_narrow): flat uint8 NV12 tensors of `body` bytes; with bind() they are views of the bound host arrays."""
    return self._run(wide, narrow, self.t["wide"]["pw"], self.t["narrow"]["pw"], self.t["narrow"]["pn"])
