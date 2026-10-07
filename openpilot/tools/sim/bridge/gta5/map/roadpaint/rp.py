"""Readers for roadpaint's outputs: height-layered paint tiles and exact paint segments.

Tile pixels (5 cm, north up, row 0 at the tile's north edge) hold up to 3 surfaces at different heights; each surface's
code is class | paint << 3 (class 0 none, 1 road, 2 kerb, 3 pavement, 4 gutter; paint 0 none, 1 white, 2 yellow).
"""
import os, pathlib, struct, zlib, functools
import numpy as np

TILE, RES, N = 64.0, 0.05, 1280
MAP = pathlib.Path(os.environ.get('GTA5MAP', '~/gta5map')).expanduser()  # ynddump's paths.jsonl
PATHS = MAP / 'paths.jsonl'
NONE, ROAD, KERB, WALK, GUTTER = 0, 1, 2, 3, 4
SEG = np.dtype([('x1', '<f4'), ('y1', '<f4'), ('z1', '<f4'), ('x2', '<f4'), ('y2', '<f4'), ('z2', '<f4'), ('w', '<f4'),
                ('colour', 'u1'), ('decal', 'u1'), ('tex', '<u2'), ('geom', '<i4')])


def read_tile(path):
  b = pathlib.Path(path).read_bytes()
  magic, tx, ty, n, res, nl = struct.unpack_from('<iiiifi', b, 0)
  o = 24
  ln, = struct.unpack_from('<i', b, o); o += 4
  code = np.frombuffer(zlib.decompress(b[o:o + ln]), np.uint8).reshape(nl, n, n); o += ln
  ln, = struct.unpack_from('<i', b, o); o += 4
  z = np.frombuffer(zlib.decompress(b[o:o + ln]), np.float32).reshape(nl, n, n)
  return code, z


class Tiles:
  """Lazy, cached access to a tile directory; sample(x, y, zref) picks, per point, the surface nearest zref."""

  def __init__(self, d, cache=64):
    self.d = pathlib.Path(d)
    self._get = functools.lru_cache(maxsize=cache)(self._load)

  def _load(self, tx, ty):
    p = self.d / f'{tx}_{ty}.bin'
    return read_tile(p) if p.exists() else None

  def sample(self, x, y, zref, zwin=3.0):
    """code (uint8) and height at world points (arrays), choosing the layer nearest zref within zwin (else 0, nan)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    zref = np.broadcast_to(np.asarray(zref, float), x.shape)
    code = np.zeros(x.shape, np.uint8)
    zz = np.full(x.shape, np.nan, np.float32)
    tx, ty = np.floor(x / TILE).astype(int), np.floor(y / TILE).astype(int)
    for key in set(zip(tx.ravel().tolist(), ty.ravel().tolist())):
      t = self._get(*key)
      if t is None:
        continue
      c, z = t
      m = (tx == key[0]) & (ty == key[1])
      col = np.clip(((x[m] - key[0] * TILE) / RES).astype(int), 0, N - 1)
      row = np.clip(((key[1] * TILE + TILE - y[m]) / RES).astype(int), 0, N - 1)
      cl = c[:, row, col]                      # layers x points
      zl = np.where(cl > 0, z[:, row, col], np.nan)
      dz = np.abs(zl - zref[m][None])
      dz = np.where(np.isnan(dz), np.inf, dz)
      best = dz.argmin(0)
      ok = dz[best, np.arange(len(best))] <= zwin
      idx = np.arange(len(best))
      code[m] = np.where(ok, cl[best, idx], 0)
      zz[m] = np.where(ok, zl[best, idx], np.nan)
    return code, zz


def read_segments(path):
  return np.fromfile(path, SEG)


def read_textures(path):
  out = {}
  for i, line in enumerate(open(path)):
    if i == 0:
      continue
    f = line.rstrip('\n').split('\t')
    out[int(f[0])] = {'name': f[1], 'decal': f[2] == '1', 'cls': int(f[3]), 'bands': f[6] if len(f) > 6 else ''}
  return out
