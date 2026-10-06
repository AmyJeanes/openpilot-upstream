"""Reads an OpenStreetMap PBF file (as osmium, ynd_to_osm.py and Geofabrik write them) with only the standard library
and numpy, for the bridge and gta5-train, whose environments have no pyosmium. It reads raw and zlib blobs, dense and
plain nodes, ways and relations; metadata (versions, users) is skipped.

The format: https://wiki.openstreetmap.org/wiki/PBF_Format (fileformat.proto, osmformat.proto).
"""
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

VARINT, LENGTH = 0, 2


def varint(buf, i: int) -> tuple[int, int]:
  out = shift = 0
  while True:
    b = buf[i]
    i += 1
    out |= (b & 0x7f) << shift
    if b < 0x80:
      return out, i
    shift += 7


def fields(buf):
  """(field number, value) for each field of a message: an int for varints, a memoryview for length-delimited ones."""
  i, n = 0, len(buf)
  while i < n:
    key, i = varint(buf, i)
    wire = key & 7
    if wire == VARINT:
      v, i = varint(buf, i)
    elif wire == LENGTH:
      size, i = varint(buf, i)
      v, i = buf[i:i + size], i + size
    elif wire == 1:
      v, i = buf[i:i + 8], i + 8
    elif wire == 5:
      v, i = buf[i:i + 4], i + 4
    else:
      raise ValueError(f"wire type {wire}")
    yield key >> 3, v


def packed(buf) -> np.ndarray:
  """A packed repeated varint field as uint64."""
  b = np.frombuffer(buf, np.uint8)
  if not len(b):
    return np.zeros(0, np.uint64)
  last = b < 0x80
  starts = np.flatnonzero(np.concatenate(([True], last[:-1])))
  which = np.cumsum(np.concatenate(([0], last[:-1].astype(np.int64))))
  shift = ((np.arange(len(b)) - starts[which]) * 7).astype(np.uint64)
  return np.add.reduceat((b & 0x7f).astype(np.uint64) << shift, starts)


def ints(buf) -> list[int]:
  """A short packed repeated varint field, without numpy's overhead per call."""
  out, v, shift = [], 0, 0
  for b in bytes(buf):
    v |= (b & 0x7f) << shift
    if b < 0x80:
      out.append(v)
      v = shift = 0
    else:
      shift += 7
  return out


def deltas(buf) -> list[int]:
  """A short packed sint64 field of differences, summed."""
  out, x = [], 0
  for u in ints(buf):
    x += (u >> 1) ^ -(u & 1)
    out.append(x)
  return out


def zigzag(u: np.ndarray) -> np.ndarray:
  return (u >> np.uint64(1)).astype(np.int64) ^ -(u & np.uint64(1)).astype(np.int64)


def signed(v: int) -> int:
  return v - (1 << 64) if v >= 1 << 63 else v


@dataclass
class OsmData:
  node_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))  # sorted
  lat: np.ndarray = field(default_factory=lambda: np.zeros(0))
  lon: np.ndarray = field(default_factory=lambda: np.zeros(0))
  node_tags: dict[int, dict[str, str]] = field(default_factory=dict)  # only the nodes with tags
  ways: dict[int, tuple[dict[str, str], list[int]]] = field(default_factory=dict)  # id -> (tags, node ids)
  relations: dict[int, tuple[dict[str, str], list[tuple[str, int, str]]]] = field(default_factory=dict)  # (type n/w/r, ref, role)

  def index(self, ids) -> np.ndarray:
    """Where each node id is in node_ids (-1 for none)."""
    ids = np.asarray(ids, np.int64)
    k = np.clip(np.searchsorted(self.node_ids, ids), 0, max(len(self.node_ids) - 1, 0))
    return np.where(len(self.node_ids) and (self.node_ids[k] == ids), k, -1)


def blobs(path: str):
  """(type, data) for each blob of the file."""
  with open(path, 'rb') as f:
    while head := f.read(4):
      size, = struct.unpack('>I', head)
      kind, length = '', 0
      for k, v in fields(memoryview(f.read(size))):
        if k == 1:
          kind = bytes(v).decode()
        elif k == 3:
          length = v
      data = b''
      for k, v in fields(memoryview(f.read(length))):
        if k == 1:
          data = bytes(v)
        elif k == 3:
          data = zlib.decompress(v)
        elif k in (4, 6, 7):
          raise ValueError("only raw and zlib PBF blobs are read")
      yield kind, data


def read(path: str, relations: tuple[str, ...] | None = None) -> OsmData:
  """The whole file; with `relations`, only the relations of those types."""
  ids, lats, lons = [], [], []
  out = OsmData()
  for kind, data in blobs(path):
    if kind != 'OSMData':
      continue
    strings, groups, gran, lat0, lon0 = [], [], 100, 0, 0
    for k, v in fields(memoryview(data)):
      if k == 1:
        strings = [bytes(s).decode() for kk, s in fields(v) if kk == 1]
      elif k == 2:
        groups.append(v)
      elif k == 17:
        gran = v
      elif k == 19:
        lat0 = signed(v)
      elif k == 20:
        lon0 = signed(v)
    for g in groups:
      for k, v in fields(g):
        if k == 2:
          _dense(v, strings, gran, lat0, lon0, ids, lats, lons, out.node_tags)
        elif k == 1:
          _node(v, strings, gran, lat0, lon0, ids, lats, lons, out.node_tags)
        elif k == 3:
          _way(v, strings, out.ways)
        elif k == 4:
          _relation(v, strings, out.relations, relations)
  if ids:
    i = np.concatenate(ids)
    order = np.argsort(i, kind='stable')
    out.node_ids, out.lat, out.lon = i[order], np.concatenate(lats)[order], np.concatenate(lons)[order]
  return out


def _tags(keys, vals, strings) -> dict[str, str]:
  return {strings[k]: strings[v] for k, v in zip(keys, vals, strict=True)}


def _dense(buf, strings, gran, lat0, lon0, ids, lats, lons, node_tags):
  i = lat = lon = kv = None
  for k, v in fields(buf):
    if k == 1:
      i = np.cumsum(zigzag(packed(v)))
    elif k == 8:
      lat = np.cumsum(zigzag(packed(v)))
    elif k == 9:
      lon = np.cumsum(zigzag(packed(v)))
    elif k == 10:
      kv = packed(v).astype(np.int64)
  if i is None:
    return
  ids.append(i)
  lats.append((lat0 + gran * lat) * 1e-9)
  lons.append((lon0 + gran * lon) * 1e-9)
  if kv is not None and len(kv):
    ends = np.flatnonzero(kv == 0)  # each node's keys and values, then 0
    start = 0
    for n, end in enumerate(ends):
      if end > start:
        node_tags[int(i[n])] = {strings[kv[j]]: strings[kv[j + 1]] for j in range(start, end, 2)}
      start = end + 1


def _node(buf, strings, gran, lat0, lon0, ids, lats, lons, node_tags):
  nid, keys, vals, lat, lon = 0, [], [], 0, 0
  for k, v in fields(buf):
    if k == 1:
      nid = (v >> 1) ^ -(v & 1)
    elif k == 2:
      keys = ints(v)
    elif k == 3:
      vals = ints(v)
    elif k == 8:
      lat = (v >> 1) ^ -(v & 1)
    elif k == 9:
      lon = (v >> 1) ^ -(v & 1)
  ids.append(np.array([nid], np.int64))
  lats.append(np.array([(lat0 + gran * lat) * 1e-9]))
  lons.append(np.array([(lon0 + gran * lon) * 1e-9]))
  if keys:
    node_tags[nid] = _tags(keys, vals, strings)


def _way(buf, strings, ways):
  wid, keys, vals, refs = 0, [], [], []
  for k, v in fields(buf):
    if k == 1:
      wid = signed(v)
    elif k == 2:
      keys = ints(v)
    elif k == 3:
      vals = ints(v)
    elif k == 8:
      refs = deltas(v)
  ways[wid] = (_tags(keys, vals, strings), refs)


def _relation(buf, strings, relations, wanted):
  rid, keys, vals, roles, mems, types = 0, [], [], b'', b'', b''
  for k, v in fields(buf):
    if k == 1:
      rid = signed(v)
    elif k == 2:
      keys = ints(v)
    elif k == 3:
      vals = ints(v)
    elif k == 8:
      roles = v
    elif k == 9:
      mems = v
    elif k == 10:
      types = v
  tags = _tags(keys, vals, strings)
  if wanted is not None and tags.get('type') not in wanted:
    return
  members = [('nwr'[t], m, strings[r]) for r, m, t in zip(ints(roles), deltas(mems), ints(types), strict=True)]
  relations[rid] = (tags, members)
