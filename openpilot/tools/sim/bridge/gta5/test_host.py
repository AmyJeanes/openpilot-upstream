"""gta5_host.py --no-link: the game, the relay and the host alone, against a stand-in relay on localhost."""
import json
import os
import socket
import subprocess
import sys
import threading
import time

GTA5 = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(GTA5, *[".."] * 5))


def free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def test_host_without_link(tmp_path):
  """With no link the host steps the brain on the relay's states and sends the HUD, disengaged; the engage key with no
  destination engages nothing (no openpilot to fall back on) and the car's controls stay as they were. Its recorder
  waits for the relay's frames, with no model tap to ask for."""
  srv = socket.create_server(("127.0.0.1", free_port()))
  srv.settimeout(60.0)
  env = {k: v for k, v in os.environ.items() if not k.startswith("GTA5_")}
  env.update(PYTHONPATH=os.pathsep.join(p for p in (ROOT, os.environ.get("PYTHONPATH")) if p), PYTHONUNBUFFERED="1",
             GTA5_OVERLAY_PROCESS="0", GTA5_MARKS=str(tmp_path / "marks.jsonl"), GTA5_RECORD=str(tmp_path / "rec"))
  out = open(tmp_path / "host.log", "w+")
  host = subprocess.Popen([sys.executable, "-m", "openpilot.tools.sim.bridge.gta5.gta5_host", "--no-link", "--brain-port",
                           str(srv.getsockname()[1]), "--frames-port", str(free_port())], env=env, stdout=out,
                          stderr=subprocess.STDOUT, cwd=tmp_path)
  try:
    conn, _ = srv.accept()
    presses = [0]
    done = threading.Event()

    def states():
      t = 0.0
      while not done.is_set():
        state = {"t": t, "inVehicle": True, "vEgo": 0.0, "heading": 0.0, "yawRate": 0.0, "pos": [0.0, 0.0, 30.0],
                 "aMeas": 0.0, "engagePresses": presses[0]}
        try:
          conn.sendall((json.dumps({"views": [], "state": state}) + "\n").encode())
        except OSError:
          return
        t += 0.05
        time.sleep(0.05)
    threading.Thread(target=states, daemon=True).start()

    got: list[dict] = []
    reader = conn.makefile("r")

    def until(cond, timeout=30.0):
      end = time.monotonic() + timeout
      while not cond():
        assert time.monotonic() < end, f"timed out; host said:\n{open(tmp_path / 'host.log').read()[-3000:]}"
        got.append(json.loads(reader.readline()))

    until(lambda: any(m.get("type") == "hud" for m in got))
    presses[0] = 1
    until(lambda: "no destination for the map driver" in open(tmp_path / "host.log").read() and
          sum(m.get("type") == "hud" for m in got) >= 5)
    log = open(tmp_path / "host.log").read()
    assert "no link: no openpilot" in log and "the engage key drives with the map driver (no openpilot)" in log
    until(lambda: "waiting for the relay's frames" in open(tmp_path / "host.log").read())  # Windows refuses in about 1 s
    assert "recording to" in log and "Traceback" not in open(tmp_path / "host.log").read()
    assert all(not m.get("engaged") for m in got if m.get("type") == "hud")
    assert not any(m.get("type") in ("control", "owner") and (m.get("active") or m.get("who") == "brain") for m in got)
    done.set()
  finally:
    host.kill()
    host.wait(10)
    srv.close()
