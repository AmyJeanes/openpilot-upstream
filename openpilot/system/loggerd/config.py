import os
import shutil
from openpilot.common.hardware.hw import Paths


CAMERA_FPS = 20
SEGMENT_LENGTH = 60

def _free_and_total_bytes(path: str) -> tuple[int, int]:
  if statvfs := getattr(os, "statvfs", None):
    st = statvfs(path)
    return st.f_bavail * st.f_frsize, st.f_blocks * st.f_frsize
  usage = shutil.disk_usage(path)  # Windows has no statvfs
  return usage.free, usage.total


def get_available_percent(default: float) -> float:
  try:
    free, total = _free_and_total_bytes(Paths.log_root())
    available_percent = 100.0 * free / total
  except OSError:
    available_percent = default

  return available_percent


def get_available_bytes(default: int) -> int:
  try:
    available_bytes, _ = _free_and_total_bytes(Paths.log_root())
  except OSError:
    available_bytes = default

  return available_bytes
