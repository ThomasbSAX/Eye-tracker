import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRESETS = json.loads((ROOT / "calibration" / "screens.json").read_text())


@dataclass
class Screen:
    name: str
    width_px: float
    height_px: float
    width_mm: float
    height_mm: float
    camera_x_mm: float
    camera_y_mm: float

    @property
    def mm_per_px(self) -> float:
        return self.width_mm / self.width_px

    @property
    def camera_mm(self) -> np.ndarray:
        return np.array([self.camera_x_mm, self.camera_y_mm])

    def mm_from_hit(self, hit: np.ndarray) -> np.ndarray:
        return self.camera_mm - hit[..., :2]

    def px_from_mm(self, mm: np.ndarray) -> np.ndarray:
        return mm / self.mm_per_px

    def inside(self, px: np.ndarray, margin_px: float = 0.0) -> np.ndarray:
        return ((px[..., 0] >= -margin_px) & (px[..., 0] <= self.width_px + margin_px)
                & (px[..., 1] >= -margin_px) & (px[..., 1] <= self.height_px + margin_px))

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "Screen":
        return cls(**json.loads(Path(path).read_text()))


def from_preset(preset: dict, width_px: float, height_px: float) -> Screen:
    return Screen(preset["name"], width_px, height_px, preset["width_mm"], preset["height_mm"],
                  preset["camera_x"] * preset["width_mm"], preset["camera_y_mm"])


def find_preset(width_px: float, height_px: float) -> dict | None:
    return next((p for p in PRESETS if [int(width_px), int(height_px)] in p["resolutions"]), None)


def from_diagonal(name: str, width_px: float, height_px: float, diagonal_in: float, camera_y_mm: float) -> Screen:
    k = diagonal_in * 25.4 / math.hypot(width_px, height_px)
    return Screen(name, width_px, height_px, width_px * k, height_px * k, width_px * k / 2.0, camera_y_mm)


def viewport_offset_px(viewport: dict) -> np.ndarray:
    if viewport.get("fullscreen") or viewport.get("browser_fullscreen"):
        return np.zeros(2)
    return np.array([viewport["screen_x"] + (viewport["outer_width"] - viewport["width"]) / 2.0,
                     viewport["screen_y"] + viewport["outer_height"] - viewport["height"]])


def from_log(log: dict) -> Screen:
    if "screen_model" in log:
        return Screen(**log["screen_model"])
    w, h = log["screen"]["width"], log["screen"]["height"]
    preset = find_preset(w, h)
    if preset:
        return from_preset(preset, w, h)
    k = float(log["mm_per_css_px"])
    cam = np.array([log["camera_guess_css_px"]["x"], log["camera_guess_css_px"]["y"]]) + viewport_offset_px(log["viewport"])
    return Screen("screen from the log", w, h, w * k, h * k, cam[0] * k, cam[1] * k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--preset")
    ap.add_argument("--resolution", nargs=2, type=float, metavar=("W", "H"))
    ap.add_argument("--size-mm", nargs=2, type=float, metavar=("W", "H"))
    ap.add_argument("--diagonal", type=float)
    ap.add_argument("--camera-mm", nargs=2, type=float, metavar=("X", "Y"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for p in PRESETS:
            print(f"{p['name']}: {p['width_mm']} x {p['height_mm']} mm, " + ", ".join(f"{w}x{h}" for w, h in p["resolutions"]))
        return
    w, h = a.resolution
    if a.preset:
        s = from_preset(next(p for p in PRESETS if a.preset.lower() in p["name"].lower()), w, h)
    elif a.diagonal:
        s = from_diagonal(f"{a.diagonal:g}-inch screen", w, h, a.diagonal, -12.0)
    else:
        s = Screen("screen", w, h, a.size_mm[0], a.size_mm[1], a.size_mm[0] / 2.0, -12.0)
    if a.camera_mm:
        s.camera_x_mm, s.camera_y_mm = a.camera_mm
    s.save(Path(a.out))
    print(json.dumps(asdict(s), ensure_ascii=False))


if __name__ == "__main__":
    main()
