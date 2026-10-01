#!/usr/bin/env python3
"""Installs the bridge's page script in a running Slow Roads over the Chrome DevTools protocol.

Under WSL it falls back to game/inject.ps1 through Windows PowerShell: the Windows game's DevTools port only listens on
Windows' localhost, which WSL can't reach in its default (NAT) networking. Run directly to reinstall the page script in
a running bridge's game, e.g. after editing it."""
import argparse
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import ClientConnection, connect

GAME_DIR = Path(__file__).parent / "game"
GAME_HOST = os.getenv("SLOWROADS_GAME_HOST", "127.0.0.1")  # the machine running the game, if not this one
# evaluated in one function scope; sr-page.js, last, puts the others together
PAGE_SCRIPTS = ("lens.js", "scene.js", "controls.js", "sr-page.js")
# The game's state is module-scoped; a never-pausing conditional breakpoint just inside a method that runs every frame
# exposes it: (global, marker at the method's start, pattern naming the value to expose, else `this`)
GLOBALS = (
  ("__srGame", "renderLive(){", None),
  # the driver's raw controls; the vehicle ignores them while the game's autodrive drives
  ("__srInput", "handleInput(", r"handleInput\(\w+\)\{[^}]{0,1200}?([\w$]+)\.signal\.Forward"),
)


def host_address(game_host: str = "127.0.0.1") -> str:
  """Our address for the game to connect back to: on the interface that reaches it, else the outbound one (the game may
  be behind a forwarded port, and under WSL its localhost isn't ours)."""
  toward = game_host if not ipaddress.ip_address(socket.gethostbyname(game_host)).is_loopback else "192.0.2.1"
  with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
    s.connect((toward, 9))  # no packet is sent; this just selects the interface
    return s.getsockname()[0]


def is_game_page(target: dict) -> bool:
  # the Steam (Electron) build loads from app://, the web version from slowroads.io
  return target.get("type") == "page" and (target["url"].startswith("app://") or "slowroads.io" in target["url"])


class DevTools:
  def __init__(self, ws: ClientConnection):
    self.ws = ws
    self.next_id = 0
    self.events: list[dict] = []

  def call(self, method: str, **params) -> dict:
    self.next_id += 1
    msg_id = self.next_id
    self.ws.send(json.dumps({"id": msg_id, "method": method, "params": params}))
    while True:
      msg = json.loads(self.ws.recv())
      if msg.get("id") == msg_id:
        if "error" in msg:
          raise RuntimeError(f"{method} failed: {msg['error'].get('message')}")
        return msg["result"]
      if "method" in msg:
        self.events.append(msg)

  def eval(self, expression: str):
    r = self.call("Runtime.evaluate", expression=expression, awaitPromise=True, returnByValue=True)
    if "exceptionDetails" in r:
      raise RuntimeError(f"page eval failed: {r['exceptionDetails'].get('exception', {}).get('description')}")
    return r["result"].get("value")

  def expose_global(self, name: str, marker: str, pattern: str | None) -> None:
    if self.eval(f"!!window.{name}"):
      return
    self.events.clear()
    self.call("Debugger.enable")  # replays a scriptParsed event for every loaded script
    try:
      location, value = None, "this"
      for e in self.events:
        if e["method"] != "Debugger.scriptParsed" or not re.search(r"/_app/immutable/.*\.js$", e["params"]["url"]):
          continue
        src = self.call("Debugger.getScriptSource", scriptId=e["params"]["scriptId"])["scriptSource"]
        i = src.find(marker)
        while i >= 0 and pattern:
          m = re.match(pattern, src[i:i + 2000])
          if m:
            value = m.group(1)
            break
          i = src.find(marker, i + 1)
        if i < 0:
          continue
        column = i - (src.rfind("\n", 0, i) + 1) + len(marker)
        location = {"scriptId": e["params"]["scriptId"], "lineNumber": src.count("\n", 0, i), "columnNumber": column}
        break
      if location is None:
        raise RuntimeError(f"could not find '{marker}' in the game scripts; the game may have updated")
      bp = self.call("Debugger.setBreakpoint", location=location, condition=f"(window.{name}={value},false)")
      deadline = time.monotonic() + 10
      while not self.eval(f"!!window.{name}") and time.monotonic() < deadline:
        time.sleep(0.1)
      self.call("Debugger.removeBreakpoint", breakpointId=bp["breakpointId"])
    finally:
      self.call("Debugger.disable")
    if not self.eval(f"!!window.{name}"):
      raise RuntimeError(f"the game loop did not run ({marker}); is the game paused in a menu or minimized?")


def inject(url: str, host: str = "127.0.0.1", port: int = 9339) -> str:
  """Installs the page script, connecting back to the bridge at `url`. Raises ConnectionError if the port isn't reachable."""
  try:
    with urllib.request.urlopen(f"http://{host}:{port}/json/list", timeout=3) as r:
      targets = json.load(r)
  except OSError as e:
    raise ConnectionError(f"Slow Roads DevTools port {host}:{port} is not reachable; start the game with --remote-debugging-port={port}") from e
  page = next((t for t in targets if is_game_page(t)), None)
  if page is None:
    raise RuntimeError(f"no Slow Roads page found on {host}:{port}")
  with connect(f"ws://{host}:{port}/devtools/page/{page['id']}", max_size=None) as ws:
    devtools = DevTools(ws)
    for name, marker, pattern in GLOBALS:
      devtools.expose_global(name, marker, pattern)
    devtools.eval(f"window.__srbConfig = {json.dumps({'url': url})}; true")
    src = "\n".join((GAME_DIR / f).read_text(encoding="utf-8") for f in PAGE_SCRIPTS)
    return str(devtools.eval(f"(() => {{\n{src}\n}})()"))


def inject_from_windows(url: str, port: int = 9339) -> str:
  """inject() through Windows PowerShell, for WSL."""
  def winpath(p: Path) -> str:
    return subprocess.check_output(["wslpath", "-w", str(p)], text=True).strip()

  cmd = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", winpath(GAME_DIR / "inject.ps1"),
         "-ScriptDir", winpath(GAME_DIR), "-Port", str(port), "-Url", url]
  try:
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
  except OSError as e:
    if e.errno == 8:  # ENOEXEC: WSL's Windows-interop binfmt handler is not registered in this distro instance
      fix = "sudo sh -c 'echo :WSLInterop:M::MZ::/init:PF > /proc/sys/fs/binfmt_misc/register'"
      raise RuntimeError(f"WSL cannot run Windows programs right now; re-register interop with: {fix}") from e
    raise
  if out.returncode != 0:
    raise RuntimeError(f"game injection failed: {out.stderr.strip() or out.stdout.strip()}")
  return out.stdout.strip()


def install(bridge_port: int = 8790, host: str = GAME_HOST, port: int = 9339, url: str | None = None) -> str:
  """Installs the page script, connecting back to the bridge on `bridge_port` (or at `url`)."""
  url = url or f"ws://{host_address(host)}:{bridge_port}"
  try:
    return inject(url, host, port)
  except ConnectionError:
    # WSL can't reach a Windows game's DevTools port, which only listens on Windows' localhost; go through Windows
    if shutil.which("powershell.exe") is None:
      raise
    return inject_from_windows(url, port)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--host", default=GAME_HOST, help="the game's machine (default: SLOWROADS_GAME_HOST, else this one)")
  parser.add_argument("--port", type=int, default=9339, help="the game's --remote-debugging-port")
  parser.add_argument("--bridge-port", type=int, default=8790, help="the bridge's WebSocket port on this machine")
  parser.add_argument("--url", help="the bridge's WebSocket, if not on this machine")
  args = parser.parse_args()
  print(install(args.bridge_port, args.host, args.port, args.url))
