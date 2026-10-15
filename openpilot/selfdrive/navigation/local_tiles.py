"""A PMTiles archive in a local file, as a byte-range source for PMTiles in place of RemoteTiles (a simulator's map, or
an offline region)."""
import os


class LocalTiles:
  def __init__(self, path: str):
    self.file = open(path, 'rb')  # closed by close()
    self.size = os.fstat(self.file.fileno()).st_size

  def read(self, offset: int, length: int) -> bytes:
    self.file.seek(offset)
    return self.file.read(length)

  def prefetch(self, ranges):
    pass  # every read is local

  def close(self):
    self.file.close()
