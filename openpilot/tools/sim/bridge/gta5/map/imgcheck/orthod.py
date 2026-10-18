"""A minimal stand-in for the bridge while capturing: the plugin's link (port 8791) with openpilot's frames off, and the
bridge's debug port (8792) so gta5_cmd.py and the capture tools work as with the bridge: commands go to the plugin,
{"type": "state", "path": p} writes the plugin's next state to p.

  python orthod.py
"""
import json
import os
import socket
import struct
import threading
import time

GAME_PORT, DEBUG_PORT = 8791, 8792


class Link:
  def __init__(self):
    self.conn = None
    self.lock = threading.Lock()
    self.state = None
    self.state_t = 0.0
    self.waiters = []  # paths for the next state
    self.cv = threading.Condition()

  def send(self, msg: dict):
    with self.lock:
      if self.conn is None:
        return False
      try:
        self.conn.sendall((json.dumps(msg) + "\n").encode())
        return True
      except OSError:
        return False

  def serve_game(self):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", GAME_PORT))
    srv.listen(4)
    while True:
      conn, addr = srv.accept()
      threading.Thread(target=self.game_client, args=(conn, addr), daemon=True).start()

  def game_client(self, conn, addr):
    """The newest connection is the game's: the plugin reconnects after a core reload or a dropped link."""
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    print(f"orthod: game connected from {addr[0]}", flush=True)
    with self.lock:
      old, self.conn = self.conn, conn
    if old is not None:
      try:
        old.shutdown(socket.SHUT_RDWR)
        old.close()
      except OSError:
        pass
    self.send({"type": "frames", "on": 0})
    try:
      self.read_game(conn)
    except OSError as e:
      print(f"orthod: game link: {e}", flush=True)
    with self.lock:
      if self.conn is conn:
        self.conn = None
    print("orthod: game disconnected", flush=True)

  def read_game(self, conn):
    buf = b""
    while True:
      chunk = conn.recv(1 << 20)
      if not chunk:
        return
      buf += chunk
      while len(buf) >= 8:
        total, head_len = struct.unpack_from("<II", buf, 0)
        if len(buf) < 4 + total:
          break
        head = buf[8:8 + head_len]
        buf = buf[4 + total:]
        try:
          msg = json.loads(head)
        except ValueError:
          continue
        if "state" in msg:
          with self.cv:
            self.state = msg["state"]
            self.state_t = time.time()
            waiters, self.waiters = self.waiters, []
            self.cv.notify_all()
          for path in waiters:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
              json.dump(self.state, f)
            os.replace(tmp, path)

  def serve_debug(self):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", DEBUG_PORT))
    srv.listen(8)
    while True:
      conn, _ = srv.accept()
      threading.Thread(target=self.debug_client, args=(conn,), daemon=True).start()

  def debug_client(self, conn):
    with conn, conn.makefile("r") as f:
      for line in f:
        try:
          msg = json.loads(line)
        except ValueError:
          continue
        if msg.get("type") == "state" and msg.get("path"):
          with self.cv:
            self.waiters.append(msg["path"])
        else:
          self.send(msg)


def main():
  link = Link()
  threading.Thread(target=link.serve_game, daemon=True).start()
  threading.Thread(target=link.serve_debug, daemon=True).start()
  print("orthod: listening", flush=True)
  while True:
    time.sleep(3600)


if __name__ == "__main__":
  main()
