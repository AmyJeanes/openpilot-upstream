"""Tiny static server with HTTP Range support (python -m http.server has none, which PMTiles needs)."""
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class H(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def end_headers(self):
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_GET(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or not os.path.isfile(path):
            return super().do_GET()
        m = re.match(r"bytes=(\d*)-(\d*)$", rng)
        size = os.path.getsize(path)
        if not m:
            return super().do_GET()
        a, b = m.groups()
        if a == "":
            start, end = max(0, size - int(b)), size - 1
        else:
            start, end = int(a), int(b) if b else size - 1
        end = min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left:
                d = f.read(min(1 << 20, left))
                if not d:
                    break
                try:
                    self.wfile.write(d)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(d)


if __name__ == "__main__":
    root, port = sys.argv[1], int(sys.argv[2])
    os.chdir(root)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
