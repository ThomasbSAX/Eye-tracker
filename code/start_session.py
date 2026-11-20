import argparse
import functools
import http.server
import mimetypes
import re
import threading
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from start_calibration import open_chrome

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
SESSIONS_DIR = ROOT / "data" / "sessions"
NAME = re.compile(r"[A-Za-z0-9_.-]+\.(mp4|webm|json)")


class SessionHandler(http.server.SimpleHTTPRequestHandler):
    received = threading.Event()
    stimuli: dict[str, Path] = {}

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/stimulus/") and path[10:] in self.stimuli:
            body = self.stimuli[path[10:]].read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:
        url = urlparse(self.path)
        name = Path(parse_qs(url.query).get("name", [""])[0]).name
        if url.path != "/upload" or not NAME.fullmatch(name):
            self.send_error(400, "upload refused")
            return
        target = SESSIONS_DIR / name
        remaining = int(self.headers.get("Content-Length", "0"))
        with target.open("wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        print(f"[SESSION] received: {target} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
        if target.suffix == ".json":
            SessionHandler.received.set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--open")
    ap.add_argument("--keep-serving", action="store_true")
    a = ap.parse_args()

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    for p in a.image:
        SessionHandler.stimuli[Path(p).name] = Path(p).resolve()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), functools.partial(SessionHandler, directory=str(WEB_DIR)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://localhost:{a.port}"
    print(f"[SESSION] add this line to the page under test: <script src=\"{base}/gaze-recorder.js\" data-upload=\"{base}/upload\"></script>", flush=True)
    print(f"[SESSION] recordings will be saved to {SESSIONS_DIR}", flush=True)
    if a.image:
        url = f"{base}/image_test.html?seconds={a.seconds:g}&" + "&".join(f"img=/stimulus/{quote(Path(p).name)}" for p in a.image)
        open_chrome(url)
        print(f"[SESSION] page: {url}", flush=True)
    elif a.open:
        open_chrome(a.open)
    try:
        if a.keep_serving or not (a.image or a.open):
            threading.Event().wait()
        else:
            SessionHandler.received.wait()
    except KeyboardInterrupt:
        pass
    server.shutdown()


if __name__ == "__main__":
    main()
