"""The game-file sources the chain reads (paint lines, markings, GTA's path nodes and links), parsed once per process and
kept parsed in ~/.cache/paintnet (JSON lines parse in seconds, a pickle loads in a fraction of that)."""
import functools, json, os, pathlib, pickle

PAINT = "/mnt/e/gta5_scratch/gamefiles_v2/rp_all/polylines.jsonl"
FEATS = "/mnt/e/gta5_scratch/gamefiles_v2/rp_all/features.jsonl"
PATHS = os.path.expanduser("~/gta5map/paths.jsonl")
CACHE = pathlib.Path("~/.cache/paintnet").expanduser()


@functools.cache
def records(path):
  """every line of a JSON lines file, parsed (shared: don't change them)."""
  st = os.stat(path)
  c = CACHE / f"{pathlib.Path(path).name}.{st.st_mtime_ns}.{st.st_size}.pkl"
  if c.exists():
    return pickle.loads(c.read_bytes())
  out = [json.loads(l) for l in open(path)]
  CACHE.mkdir(parents=True, exist_ok=True)
  for old in CACHE.glob(f"{pathlib.Path(path).name}.*.pkl"):
    old.unlink()
  tmp = c.with_suffix(".tmp")
  tmp.write_bytes(pickle.dumps(out, pickle.HIGHEST_PROTOCOL))
  tmp.replace(c)
  return out


def paint():
  return records(PAINT)


def features():
  return records(FEATS)


def paths():
  return records(PATHS)


def fast_tiles(rp):
  """rp.read_tile reading each surface raster tile from an uncompressed copy in the cache, mapped rather than read
  (decompressing a tile takes ~30 ms and most steps only read scattered pixels of it)."""
  read_tile = rp.read_tile

  def read(path):
    st = os.stat(path)
    c = CACHE / "tiles" / f"{pathlib.Path(path).stem}.{st.st_mtime_ns}.{st.st_size}"
    if not c.with_suffix(".z.npy").exists():
      import numpy as np
      code, z = read_tile(path)
      c.parent.mkdir(parents=True, exist_ok=True)
      for old in c.parent.glob(f"{pathlib.Path(path).stem}.*.npy"):
        old.unlink()
      for suf, a in ((".code", code), (".z", z)):  # z last: it marks a complete copy
        tmp = c.with_suffix(suf + ".tmp.npy")
        np.save(tmp, a)
        tmp.replace(c.with_suffix(suf + ".npy"))
    import numpy as np
    return tuple(np.asarray(np.load(c.with_suffix(s + ".npy"), mmap_mode="r")) for s in (".code", ".z"))

  rp.read_tile = read
