#!/usr/bin/env python3
"""Serves the map view: a web page of the roads with the car and routes, for a browser on this machine or the network.

MapView.update() sets what /state reports: {"t": seconds, "car": {"x", "y", "z", "bearing"}, "routes": {name: [[x, y], ...]},
"waypoint": [x, y], "text": "..."}, in the same metres as roads.json, bearing in degrees clockwise from north, z the
height of the road under the car (optional: the view dims the roads on other levels from it).
A destination picked on the page (POST /destination {"x", "y"}, or {} to clear) waits in take_destination().
lanes.json beside roads.json (osm_to_roads.py --lanes), where there is one, gives the lines painted on the roads.
Every CHECK_EVERY s the view asks itself for /state: when that goes unanswered it says so, with the threads' stacks in
`stacks` where given, and it serves again if its server thread has ended.
"""
import argparse
import faulthandler
import gzip
import http.client
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TextIO

PAGE = os.path.join(os.path.dirname(__file__), 'view.html')
REQUEST_TIMEOUT = 10.0  # s for a connection's request and reply, rather than a silent client holding a thread for good
CHECK_EVERY = 30.0  # s
CHECK_TIMEOUT = 10.0  # s


class MapView:
  def __init__(self, roads: str, port: int, host: str = '0.0.0.0', stacks: TextIO | None = None,
               check_every: float = CHECK_EVERY, request_timeout: float = REQUEST_TIMEOUT):
    self.files = {'/roads.json': roads, '/lanes.json': os.path.join(os.path.dirname(roads), 'lanes.json')}
    self.cache: dict[str, tuple[float, bytes]] = {}
    self.cache_lock = threading.Lock()  # the page asks for both maps at once: each is compressed once
    self.stacks = stacks
    self.stuck: str | None = None  # why the last check went unanswered
    self.closing = threading.Event()
    self.state = b'{}'
    self.destination: tuple | None = None  # ([x, y] or None,) until taken
    view = self

    class Handler(BaseHTTPRequestHandler):
      timeout = request_timeout

      def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/':
          with open(PAGE, 'rb') as f:
            self._reply(f.read(), 'text/html; charset=utf-8')
        elif path in view.files:
          body = view.read(path)
          if body is None:
            self.send_error(404)
          else:
            self._reply(body, 'application/json', gzipped=True)
        elif path == '/state':
          self._reply(view.state, 'application/json')
        else:
          self.send_error(404)

      def do_POST(self):
        if self.path.split('?')[0] != '/destination':
          self.send_error(404)
          return
        try:
          d = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
          view.destination = ([float(d['x']), float(d['y'])] if 'x' in d else None,)
        except (ValueError, KeyError, TypeError):
          self.send_error(400)
          return
        self._reply(b'{}', 'application/json')

      def _reply(self, body: bytes, kind: str, gzipped: bool = False):
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Cache-Control', 'no-store')
        if gzipped:
          self.send_header('Content-Encoding', 'gzip')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

      def log_message(self, *args):
        pass

    self.server = ThreadingHTTPServer((host, port), Handler)
    self.server.daemon_threads = True
    self.server.block_on_close = False
    self.port = self.server.server_address[1]
    self.host = '127.0.0.1' if host in ('', '0.0.0.0') else host
    self.thread = self._start()
    if check_every:
      threading.Thread(target=self._watch, args=(check_every,), daemon=True, name='map view watch').start()

  def _start(self) -> threading.Thread:
    t = threading.Thread(target=self._serve, daemon=True, name='map view')
    t.start()
    return t

  def _serve(self):
    while not self.closing.is_set():
      try:
        self.server.serve_forever()
      except Exception:
        print(f"map view: its server failed, serving again: {traceback.format_exc()}", flush=True)
        self.closing.wait(1.0)

  def answers(self, timeout: float = CHECK_TIMEOUT) -> str | None:
    """None if the view answers /state within `timeout` s, else why not."""
    conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
    try:
      conn.request('GET', '/state')
      reply = conn.getresponse()
      reply.read()
      return None if reply.status == 200 else f"status {reply.status}"
    except OSError as e:
      return f"{type(e).__name__}: {e}"
    finally:
      conn.close()

  def _watch(self, every: float):
    while not self.closing.wait(every):
      self.check()

  def check(self, timeout: float = CHECK_TIMEOUT):
    if not self.thread.is_alive():
      print("map view: its server thread had ended; serving again", flush=True)
      self.thread = self._start()
    why = self.answers(timeout)
    if why is not None and self.stuck is None:
      print(f"map view: no answer on port {self.port} ({why})" +
            (f"; the threads' stacks are in {self.stacks.name}" if self.stacks is not None else ""), flush=True)
      self.dump_stacks(f"map view: no answer on port {self.port} ({why})")
    elif why is None and self.stuck is not None:
      print(f"map view: answering again on port {self.port}", flush=True)
    self.stuck = why

  def dump_stacks(self, why: str):
    if self.stacks is None:
      return
    try:
      self.stacks.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}: {why}\n")
      self.stacks.flush()
      faulthandler.dump_traceback(self.stacks, all_threads=True)
    except (OSError, ValueError):
      pass

  def read(self, name: str) -> bytes | None:
    """A map file, gzipped; None where there isn't one."""
    path = self.files[name]
    if not os.path.exists(path):
      return None
    mtime = os.path.getmtime(path)  # picks up a rebuilt map
    with self.cache_lock:
      if name not in self.cache or self.cache[name][0] != mtime:
        with open(path, 'rb') as f:
          self.cache[name] = (mtime, gzip.compress(f.read(), 6))
      return self.cache[name][1]

  def take_destination(self) -> tuple | None:
    d, self.destination = self.destination, None
    return d

  def update(self, state: dict):
    self.state = json.dumps(state, separators=(',', ':')).encode()

  def close(self):
    self.closing.set()
    self.server.shutdown()
    self.server.server_close()


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('roads', help="osm_to_roads.py's roads.json")
  p.add_argument('--port', type=int, default=8793)
  p.add_argument('--state', help='a JSON file to show as the state, reread as it changes')
  args = p.parse_args()
  view = MapView(args.roads, args.port)
  print(f"map view on http://localhost:{args.port}/")
  mtime = 0.0
  while True:
    if args.state and os.path.exists(args.state) and os.path.getmtime(args.state) != mtime:
      mtime = os.path.getmtime(args.state)
      with open(args.state) as f:
        view.update(json.load(f))
    time.sleep(0.5)


if __name__ == '__main__':
  main()
