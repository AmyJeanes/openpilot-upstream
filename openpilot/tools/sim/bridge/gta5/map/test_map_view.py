"""The map view's server (map_view.py): a silent client doesn't hold it, it serves again after its server fails, and
its check says when it stops answering, with the threads' stacks."""
import json
import socket
import threading
import time
import urllib.request

from openpilot.tools.sim.bridge.gta5.map.map_view import MapView

NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def make_view(tmp_path, **kw) -> MapView:
  roads = tmp_path / "roads.json"
  roads.write_text("{}")
  return MapView(str(roads), 0, host="127.0.0.1", check_every=0, **kw)


def test_state_and_destination(tmp_path):
  v = make_view(tmp_path)
  try:
    v.update({"t": 1.0})
    with NO_PROXY.open(f"http://127.0.0.1:{v.port}/state", timeout=5) as r:
      assert json.loads(r.read()) == {"t": 1.0}
    req = urllib.request.Request(f"http://127.0.0.1:{v.port}/destination", json.dumps({"x": 1, "y": 2}).encode())
    with NO_PROXY.open(req, timeout=5) as r:
      r.read()
    assert v.take_destination() == ([1.0, 2.0],) and v.take_destination() is None
  finally:
    v.close()


def test_a_silent_client_is_dropped(tmp_path):
  v = make_view(tmp_path, request_timeout=0.5)
  try:
    silent = socket.create_connection(("127.0.0.1", v.port), timeout=5)
    assert v.answers(timeout=2) is None
    assert silent.recv(1) == b""  # closed on it, its thread free
    silent.close()
  finally:
    v.close()


def test_serves_again_after_its_server_fails(tmp_path, capsys):
  v = make_view(tmp_path)
  failed = threading.Event()
  actions = v.server.service_actions

  def fail_once():
    if not failed.is_set():
      failed.set()
      raise RuntimeError("a bug")
    actions()

  v.server.service_actions = fail_once
  try:
    assert failed.wait(5)
    assert v.answers(timeout=5) is None
    assert "its server failed" in capsys.readouterr().out
  finally:
    v.close()


def test_check_says_when_the_view_stops_answering(tmp_path, capsys):
  stacks = open(tmp_path / "stacks.txt", "a", buffering=1)
  v = make_view(tmp_path, stacks=stacks)
  handle = v.server._handle_request_noblock
  release = threading.Event()

  def stuck():
    release.wait(10)
    handle()

  v.server._handle_request_noblock = stuck
  try:
    v.check(timeout=0.5)
    out = capsys.readouterr().out
    assert "no answer on port" in out and str(tmp_path / "stacks.txt") in out
    dump = (tmp_path / "stacks.txt").read_text()
    assert "map view: no answer" in dump and "in stuck" in dump  # the server thread, where it waits
    v.check(timeout=0.5)
    assert capsys.readouterr().out == ""  # said once
    v.server._handle_request_noblock = handle
    release.set()
    deadline = time.monotonic() + 5
    while v.stuck is not None and time.monotonic() < deadline:
      v.check(timeout=1)
    assert v.stuck is None and "answering again" in capsys.readouterr().out
  finally:
    release.set()
    v.close()
    stacks.close()


def test_close_frees_the_port(tmp_path):
  v = make_view(tmp_path)
  v.close()
  try:
    socket.create_connection(("127.0.0.1", v.port), timeout=2).close()
    raise AssertionError("still listening")
  except ConnectionRefusedError:
    pass
