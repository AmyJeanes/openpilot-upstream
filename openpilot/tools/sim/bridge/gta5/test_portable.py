"""The host side imports without openpilot's runtime (cereal, msgq, capnp, zmq, opendbc), as it must on Windows' own
Python, so a stray import can't creep back."""
import os
import subprocess
import sys

BLOCKED = ("cereal", "openpilot.cereal", "msgq", "capnp", "zmq", "opendbc")
GTA5 = "openpilot.tools.sim.bridge.gta5"
MODULES = [
  *(f"{GTA5}.map.{m}" for m in ("osm_pbf", "osm_lanes", "paths", "gta5_map", "router", "dest_snap", "stop_lines", "junctions",
                                "lane_match", "map_view")),
  *(f"{GTA5}.{m}" for m in ("gta5_mapdrive", "gta5_wrongway", "gta5_blindspot", "gta5_scene", "gta5_expert", "gta5_overlay",
                            "gta5_driver", "gta5_relay", "gta5_cmd", "gta5_stacks", "gta5_record", "gta5_rx")),
  *(f"openpilot.selfdrive.navd.{m}" for m in ("planner", "maneuvers", "lane_slots", "route_input", "inputs", "map_match")),
]
IMPORT_BLOCKED = """
import importlib, sys
blocked, modules = sys.argv[1].split(","), sys.argv[2].split(",")

class Block:
  def find_spec(self, name, path=None, target=None):
    if any(name == b or name.startswith(b + ".") for b in blocked):
      raise ImportError(f"{name} is openpilot's runtime")

sys.meta_path.insert(0, Block())
for m in modules:
  importlib.import_module(m)
"""


def test_host_modules_import_without_openpilot():
  root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 5))
  env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in (root, os.environ.get("PYTHONPATH")) if p)}
  r = subprocess.run([sys.executable, "-c", IMPORT_BLOCKED, ",".join(BLOCKED), ",".join(MODULES)], env=env,
                     capture_output=True, text=True, timeout=120)
  assert r.returncode == 0, r.stderr[-2000:]
