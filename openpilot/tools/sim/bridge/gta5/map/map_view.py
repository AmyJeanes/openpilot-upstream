#!/usr/bin/env python3
"""Serves the map view: a web page of the roads with the car and routes, for a browser on this machine or the network.

MapView.update() sets what /state reports: {"t": seconds, "car": {"x", "y", "bearing"}, "routes": {name: [[x, y], ...]},
"waypoint": [x, y], "text": "..."}, in the same metres as roads.json, bearing in degrees clockwise from north.
A destination picked on the page (POST /destination {"x", "y"}, or {} to clear) waits in take_destination().
lanes.json beside roads.json (osm_to_roads.py --lanes), where there is one, gives the lines painted on the roads.
"""
import argparse
import gzip
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PAGE = os.path.join(os.path.dirname(__file__), 'view.html')


class MapView:
  def __init__(self, roads: str, port: int, host: str = '0.0.0.0'):
    self.files = {'/roads.json': roads, '/lanes.json': os.path.join(os.path.dirname(roads), 'lanes.json')}
    self.cache: dict[str, tuple[float, bytes]] = {}
    self.state = b'{}'
    self.destination: tuple | None = None  # ([x, y] or None,) until taken
    view = self

    class Handler(BaseHTTPRequestHandler):
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
    threading.Thread(target=self.server.serve_forever, daemon=True).start()

  def read(self, name: str) -> bytes | None:
    """A map file, gzipped; None where there isn't one."""
    path = self.files[name]
    if not os.path.exists(path):
      return None
    mtime = os.path.getmtime(path)  # picks up a rebuilt map
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
    self.server.shutdown()


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
