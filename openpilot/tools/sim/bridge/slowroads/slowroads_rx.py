"""Game connection for the Slow Roads bridge, run in its own process.

Receiving and decoding two video streams in the bridge process competes for the GIL with the 100 Hz car and sensor
threads, which made frame delivery jittery enough for modeld outputs to arrive late. This process owns the WebSocket,
decodes into shared memory and hands the bridge only small messages: (seq, slot, views, state)."""
import asyncio
import json
import struct
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory

import av
import numpy as np
from websockets.asyncio.server import broadcast, serve

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.sim.lib.camerad import W, H

VIEWS = ("road", "wide")
SLOTS = 3  # the bridge may still be copying the previous frame while the next one is written
NV12_SIZE = get_nv12_info(W, H)[3]


def nv12_planes(buf) -> tuple[np.ndarray, np.ndarray]:
  """Y and interleaved UV views into a padded NV12 buffer for a W x H frame."""
  stride, y_height, uv_height, size = get_nv12_info(W, H)
  planes = np.frombuffer(buf, dtype=np.uint8, count=size)[:stride * (y_height + uv_height)].reshape(-1, stride)
  return planes[:H, :W], planes[y_height:y_height + H // 2, :W].reshape(H // 2, W // 2, 2)


def write_decoded(frame: "av.VideoFrame", buf) -> None:
  """Decoded WebCodecs frame -> padded NV12; WebGL readback rows are bottom-up, so flip both planes."""
  arr = frame.to_ndarray(format="yuv420p")
  y, uv = nv12_planes(buf)
  quarter = (H // 2) * (W // 2)
  u = arr[H:].reshape(-1)[:quarter].reshape(H // 2, W // 2)
  v = arr[H:].reshape(-1)[quarter:].reshape(H // 2, W // 2)
  y[:] = arr[:H][::-1]
  uv[..., 0] = u[::-1]
  uv[..., 1] = v[::-1]


class Receiver:
  def __init__(self, frames: Connection, controls: Connection, shm_names: dict[str, str]):
    self.frames = frames
    self.controls = controls
    self.shm = {name: SharedMemory(name=shm_names[name]) for name in VIEWS}
    self.decoders: dict[str, av.CodecContext] = {}
    self.clients: set = set()
    self.seq = 0

  def slot_buf(self, view: str, slot: int):
    buf = self.shm[view].buf
    assert buf is not None
    return buf[slot * NV12_SIZE:(slot + 1) * NV12_SIZE]

  def decode(self, view: str, data: bytes) -> "av.VideoFrame | None":
    dec = self.decoders.get(view)
    if dec is None:
      dec = self.decoders[view] = av.CodecContext.create("h264", "r")
      dec.flags |= av.codec.context.Flags.low_delay
    out = None
    try:
      for packet in dec.parse(data):
        for frame in dec.decode(packet):
          out = frame
    except av.error.InvalidDataError:
      # lost the stream state (no keyframe since connecting); ask the page for one
      self.request_keyframe()
      return None
    return out

  def request_keyframe(self) -> None:
    broadcast(self.clients, json.dumps({"type": "keyframe"}))

  def on_frame(self, msg: bytes) -> None:
    head_len = struct.unpack_from("<I", msg, 0)[0]
    head = json.loads(msg[4:4 + head_len])
    w, h = head["width"], head["height"]
    slot = self.seq % SLOTS
    written = []
    if (w, h) != (W, H):
      raise ValueError(f"unsupported frame size {w}x{h}")
    off = 4 + head_len
    for view in head["views"]:
      frame = self.decode(view["name"], msg[off:off + view["len"]])
      off += view["len"]
      if frame is not None and view["name"] in self.shm:
        write_decoded(frame, self.slot_buf(view["name"], slot))
        written.append(view["name"])
    if written:
      self.seq += 1
      self.frames.send((slot, written, head["state"]))

  async def handler(self, ws):
    self.clients.add(ws)
    print("slowroads: game connected", flush=True)
    try:
      async for msg in ws:
        if isinstance(msg, bytes):
          self.on_frame(msg)
    finally:
      self.clients.discard(ws)
      print("slowroads: game disconnected", flush=True)

  def on_control(self) -> None:
    while self.controls.poll():
      broadcast(self.clients, json.dumps(self.controls.recv()))

  async def serve(self, port: int, ready: Connection) -> None:
    asyncio.get_running_loop().add_reader(self.controls.fileno(), self.on_control)
    async with serve(self.handler, "0.0.0.0", port, max_size=None, compression=None):
      ready.send(None)
      await asyncio.Future()


def rx_main(port: int, frames: Connection, controls: Connection, ready: Connection, shm_names: dict[str, str]) -> None:
  try:
    asyncio.run(Receiver(frames, controls, shm_names).serve(port, ready))
  except Exception as e:
    ready.send(f"{type(e).__name__}: {e}")
    raise
