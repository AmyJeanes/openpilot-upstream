import os
import struct
import tempfile
import time

import numpy as np

from openpilot.selfdrive.modeld.route_input import (HEADER, RouteInputReader, RouteInputWriter, model_layout, read_model_layout,
                                                    write_model_layout)
from openpilot.tools.sim.bridge.gta5.navd import route_input as ri

N = 173


def vec(seed: int) -> np.ndarray:
  return np.random.default_rng(seed).random(N).astype(np.float32)


def age(writer: RouteInputWriter, seconds: float):
  struct.pack_into('<d', writer.mm, 8, time.monotonic() - seconds)


class TearingBuffer(bytearray):
  """A mapping the writer changes while the reader copies the vector."""
  def __getitem__(self, k):
    if isinstance(k, slice):
      struct.pack_into('<I', self, 0, struct.unpack_from('<I', self)[0] + 2)
    return super().__getitem__(k)


def test_missing_file():
  with tempfile.TemporaryDirectory() as d:
    reader = RouteInputReader(N, os.path.join(d, "route_input"))
    assert not reader.read().any()
    writer = RouteInputWriter(N, reader.path)  # the bridge starting after modeld
    writer.write(vec(0))
    np.testing.assert_array_equal(reader.read(), vec(0))


def test_round_trip():
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "route_input")
    writer, reader = RouteInputWriter(N, path), RouteInputReader(N, path)
    assert not reader.read().any()  # created but never written
    for i in range(3):
      writer.write(vec(i))
      np.testing.assert_array_equal(reader.read(), vec(i))
    assert writer.seq == 6 and HEADER.unpack_from(writer.mm)[0] == 6


def test_wrong_size():
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "route_input")
    RouteInputWriter(N, path).write(vec(0))
    assert not RouteInputReader(N + 1, path).read().any()
    RouteInputWriter(N - 1, path).write(vec(1)[:-1])  # a smaller writer leaves the file at its size
    assert os.path.getsize(path) == HEADER.size + 4 * N
    assert not RouteInputReader(N, path).read().any()


def test_prefix():
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "route_input")
    longer = np.random.default_rng(3).random(N + 50).astype(np.float32)
    RouteInputWriter(N + 50, path).write(longer)  # a newer layout, appended to
    np.testing.assert_array_equal(RouteInputReader(N, path).read(), longer[:N])
    np.testing.assert_array_equal(RouteInputReader(N + 50, path).read(), longer)


def test_torn():
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "route_input")
    writer, reader = RouteInputWriter(N, path), RouteInputReader(N, path)
    writer.write(vec(0))
    struct.pack_into('<I', writer.mm, 0, writer.seq + 1)  # mid-write
    assert not reader.read().any()  # no good input yet
    struct.pack_into('<I', writer.mm, 0, writer.seq)
    np.testing.assert_array_equal(reader.read(), vec(0))
    writer.write(vec(1))
    struct.pack_into('<I', writer.mm, 0, writer.seq + 1)
    np.testing.assert_array_equal(reader.read(), vec(0))  # the last good one
    struct.pack_into('<I', writer.mm, 0, writer.seq)
    reader.mm = TearingBuffer(writer.mm[:])
    np.testing.assert_array_equal(reader.read(), vec(0))
    reader.last_t -= 2.0
    assert not reader.read().any()  # the last good one is stale too


def test_stale():
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "route_input")
    writer, reader = RouteInputWriter(N, path), RouteInputReader(N, path)
    writer.write(vec(0))
    age(writer, 0.5)
    np.testing.assert_array_equal(reader.read(), vec(0))
    age(writer, 1.5)
    assert not reader.read().any()
    writer.write(vec(1))
    np.testing.assert_array_equal(reader.read(), vec(1))


def test_writer_restart():
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "route_input")
    writer, reader = RouteInputWriter(N, path), RouteInputReader(N, path)
    writer.write(vec(0))
    struct.pack_into('<I', writer.mm, 0, writer.seq + 1)  # died mid-write
    writer = RouteInputWriter(N, path)
    assert writer.seq % 2 == 0
    writer.write(vec(1))
    np.testing.assert_array_equal(reader.read(), vec(1))
    # a new file in its place: the reader follows once the old one goes stale
    age(writer, 2.0)
    os.unlink(path)
    RouteInputWriter(N, path).write(vec(2))
    assert not reader.read().any()
    np.testing.assert_array_equal(reader.read(), vec(2))


def test_layouts():
  """The bridge writes the encoder's whole vector (v2); each model's reader takes its layout's floats of it."""
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "route_input")
    full = np.random.default_rng(4).random(ri.ROUTE_LEN).astype(np.float32)
    RouteInputWriter(ri.ROUTE_LEN, path).write(full)
    for version in (1, 2, 3):
      lay = model_layout(ri.LAYOUTS[version].width)
      assert lay.version == version
      np.testing.assert_array_equal(RouteInputReader.for_layout(lay, path).read(), ri.convert(full, version))
    v3 = RouteInputReader.for_layout(model_layout(218, "3"), path).read()
    assert len(v3) == 218 and v3[165] == full[165] and v3[166] == full[171]
    RouteInputWriter(N, path).write(vec(5))  # a writer of v1 only: a v3 model gets no route rather than v1's floats
    assert not RouteInputReader.for_layout(ri.LAYOUTS[3], path).read().any()


def test_model_layout():
  assert model_layout(218).version == 3 and model_layout(223, "2").version == 2
  for width, version in ((218, "2"), (200, None)):
    try:
      model_layout(width, version)
      raise AssertionError((width, version))
    except ValueError:
      pass
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "route_input.model")
    assert read_model_layout(path) is None
    write_model_layout(ri.LAYOUTS[3], path)
    assert read_model_layout(path) == {"version": 3, "width": 218}
    write_model_layout(None, path)
    assert read_model_layout(path) == {}
