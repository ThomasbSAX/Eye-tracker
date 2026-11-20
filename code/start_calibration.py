import argparse
import functools
import http.server
import json
import re
import subprocess
import tempfile
import threading
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
PAGE_DIR = ROOT / "calibration"
RECORDINGS_DIR = ROOT / "data" / "calibration"
CHROME = Path("/Applications/Google Chrome.app")


class CalibrationHandler(http.server.SimpleHTTPRequestHandler):
    received_log = threading.Event()

    def do_POST(self) -> None:
        url = urlparse(self.path)
        name = Path(parse_qs(url.query).get("name", [""])[0]).name
        if url.path != "/upload" or not re.fullmatch(r"calib_[0-9_]+\.(mp4|webm|json)", name):
            self.send_error(400, "upload refused")
            return
        remaining = int(self.headers.get("Content-Length", "0"))
        target = RECORDINGS_DIR / name
        with target.open("wb") as f:
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        body = json.dumps({"path": str(target.relative_to(ROOT))}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        print(f"[CALIB] received: {target} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
        if target.suffix == ".json":
            CalibrationHandler.received_log.set()


def open_chrome(url: str) -> None:
    if not CHROME.exists():
        webbrowser.open(url)
        return
    profile = Path(tempfile.gettempdir()) / "gaze_calibration_chrome"
    subprocess.run(["pkill", "-f", f"user-data-dir={profile}"], check=False)
    subprocess.run(["open", "-na", str(CHROME), "--args", f"--user-data-dir={profile}", "--no-first-run",
                    "--start-fullscreen", "--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required",
                    f"--app={url}"], check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Opens the gaze calibration page")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds (about 1 min each)")
    parser.add_argument("--manual", action="store_true", help="No automatic start (buttons and space bar)")
    parser.add_argument("--keep-serving", action="store_true", help="Keep serving after the files are received")
    args = parser.parse_args()

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    handler = functools.partial(CalibrationHandler, directory=str(PAGE_DIR))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://localhost:{args.port}/calibration.html"
    if not args.manual:
        url += f"?auto=1&rounds={args.rounds}"
    open_chrome(url)
    print(f"[CALIB] page: {url}", flush=True)
    print(f"[CALIB] files will be saved to: {RECORDINGS_DIR}", flush=True)
    try:
        if args.keep_serving:
            threading.Event().wait()
        else:
            CalibrationHandler.received_log.wait()
            print("[CALIB] video and log received, stopping the server", flush=True)
    except KeyboardInterrupt:
        pass
    server.shutdown()


if __name__ == "__main__":
    main()
