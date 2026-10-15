#!/usr/bin/env python3
"""Render a route's 3X->comma 4 reprojection debug views to an mp4, off the device.
  openpilot/tools/reproject_c4/debug_clip.py <dongle/route> -s start -e end -o out.mp4 [-d data_dir] [-f MB]
Top row: the comma 4 narrow and wide frames the big model saw, rebuilt on this PC's GPU from the route's fcamera and ecamera
at a narrow->wide rotation (rotations.py: the one the route logged, else one fitted here), and the seam check. Bottom row:
the 3X frames, the model's two 512x256 inputs and the logged timings. Any 3X route with its full-resolution cameras
(fcamera, ecamera) uploaded.

route_data.py reads the route, stage.py runs the reprojection, clip.py puts each frame through them and hands it to the
drawing processes (draw.py), which lay out the picture (view.py) with the stats (stats.py)."""
import logging
import os
import subprocess
import time
from argparse import ArgumentParser

import tqdm

from openpilot.tools.clip.run import FRAMERATE
from openpilot.tools.lib.route import Route
from openpilot.tools.reproject_c4 import view as V
from openpilot.tools.reproject_c4.clip import Clip

logger = logging.getLogger(__name__)


def render(clip: Clip, output: str, target_mb: float) -> None:
  ffmpeg = ['ffmpeg', '-v', 'warning', '-nostats', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{V.W}x{V.H}', '-r', str(FRAMERATE),
            '-i', 'pipe:0', '-vf', 'format=yuv420p', '-c:v', 'libx264', '-preset', 'veryfast']
  if target_mb > 0:
    rate = f'{int(target_mb * 8 * 1024 / ((clip.last - clip.first) / FRAMERATE))}k'
    ffmpeg += ['-b:v', rate, '-maxrate', rate, '-bufsize', rate]
  else:
    ffmpeg += ['-crf', '20']
  enc = subprocess.Popen(ffmpeg + ['-y', '-f', 'mp4', output], stdin=subprocess.PIPE)
  assert enc.stdin is not None
  with tqdm.tqdm(total=clip.last - clip.first, desc='Rendering', unit='frame') as bar:
    def write(keep: int) -> None:
      while clip.pending and (len(clip.pending) > keep or clip.ready()):
        enc.stdin.write(clip.take()[1])
        bar.update(1)
    try:
      while True:
        write(clip.depth - 1)
        if not clip.submit():
          break
      write(0)
    finally:
      enc.stdin.close()
      enc.wait()


def main():
  logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
  ap = ArgumentParser(description="Render a route's reprojection debug views to an mp4")
  ap.add_argument('route', help='Route ID (dongle/route or dongle/route/start/end)')
  ap.add_argument('-s', '--start', type=int, help='Start time in seconds')
  ap.add_argument('-e', '--end', type=int, help='End time in seconds')
  ap.add_argument('-o', '--output', required=True, help='The mp4 to render')
  ap.add_argument('-d', '--data-dir', help='Local directory with route data')
  ap.add_argument('-f', '--file-size', type=float, default=0, help="A target file size in MB (default: ffmpeg's quality)")
  a = ap.parse_args()
  if a.route.count('/') == 3:
    parts = a.route.split('/')
    a.route, a.start, a.end = '/'.join(parts[:2]), a.start or int(parts[2]), a.end or int(parts[3])
  if a.start is None or a.end is None:
    ap.error('--start and --end are required')
  if a.end <= a.start:
    ap.error(f'end ({a.end}) must be greater than start ({a.start})')

  t0 = time.monotonic()
  try:
    clip = Clip(Route(a.route, data_dir=a.data_dir), a.start, a.end)
    try:
      logger.info(f'rendering {a.end - a.start} s to {a.output}')
      render(clip, a.output, a.file_size)
      logger.info(f'{os.path.abspath(a.output)} in {time.monotonic() - t0:.0f} s')
    finally:
      clip.close()
  except RuntimeError as e:
    raise SystemExit(f'error: {e}') from None
  except KeyboardInterrupt:
    raise SystemExit(130) from None


if __name__ == '__main__':
  main()
