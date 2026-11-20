import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

FIX_SETTLE_MS = 450.0
FIX_END_GUARD_MS = 50.0
PURSUIT_SETTLE_MS = 800.0
IRIS_RADIUS_MM = 5.9
IRIS = {"left": 468, "right": 473}


@dataclass
class Setup:
    focal_px: float = 1000.0
    camera_x_mm: float = 0.0
    camera_y_mm: float = 0.0
    mirrored: bool = False


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def fick(v: np.ndarray) -> np.ndarray:
    v = unit(v)
    return np.degrees(np.stack([np.arctan2(v[..., 0], v[..., 2]), np.arcsin(np.clip(v[..., 1], -1, 1))], -1))


def from_fick(a: np.ndarray) -> np.ndarray:
    y, p = np.radians(a[..., 0]), np.radians(a[..., 1])
    return np.stack([np.cos(p) * np.sin(y), np.sin(p), np.cos(p) * np.cos(y)], -1)


def focal_from_iris(lm: np.ndarray, pose: np.ndarray, ok: np.ndarray) -> float:
    radius_px = np.max(np.abs(lm[ok][:, 469:473, :2] - lm[ok][:, 468:469, :2]), axis=(1, 2))
    return float(np.median(radius_px * np.abs(pose[ok][:, 2, 3])) * 10.0 / IRIS_RADIUS_MM)


def synchronize(t_ms: np.ndarray, bright: np.ndarray, flashes: list[dict]) -> float:
    z = bright - median_filter(bright, size=31, mode="nearest")
    z = (z - np.median(z)) / (1.4826 * np.median(np.abs(z - np.median(z))) + 1e-9)
    offsets = np.arange(-3000.0, 3000.0, 1.0)
    score = np.full(len(offsets), -np.inf)
    for j, off in enumerate(offsets):
        s = []
        for fl in flashes:
            m = (t_ms >= fl["t_start"] + off) & (t_ms < fl["t_end"] + off)
            if m.sum() < 2:
                break
            s.append(z[m].mean())
        else:
            score[j] = min(s)
    return float(np.median(offsets[score >= score.max() - 1e-9]))


def eye_positions_mm(lm: np.ndarray, pose: np.ndarray, width: int, height: int, setup: Setup) -> dict:
    depth = uniform_filter1d(np.abs(pose[:, 2, 3]) * 10.0, size=15, mode="nearest")
    out = {}
    for eye, i in IRIS.items():
        u = lm[:, i, 0] - width / 2.0
        v = lm[:, i, 1] - height / 2.0
        out[eye] = np.stack([u * depth / setup.focal_px, -v * depth / setup.focal_px, -depth], -1)
    return out


def screen_points_mm(target_mm: np.ndarray, setup: Setup) -> np.ndarray:
    sx = target_mm[:, 0] - setup.camera_x_mm
    sy = target_mm[:, 1] - setup.camera_y_mm
    return np.stack([sx if setup.mirrored else -sx, -sy, np.zeros_like(sx)], -1)


def screen_hit_mm(origin: np.ndarray, gaze: np.ndarray) -> np.ndarray:
    s = -origin[:, 2] / np.where(np.abs(gaze[:, 2]) > 1e-6, gaze[:, 2], np.nan)
    return origin + s[:, None] * gaze


def build(features: Path, log_path: Path, setup: Setup) -> dict:
    f = dict(np.load(features))
    width, height = (int(v) for v in f["size"])
    log = json.loads(Path(log_path).read_text())
    offset = synchronize(f["t_ms"], f["bright"], log["flashes"])
    t_log = f["t_ms"] - offset

    s = log["samples"]
    st = np.array([r[0] for r in s])
    sround = np.array([r[1] for r in s])
    sidx = np.array([r[2] for r in s])
    sphase = np.array([r[3] for r in s])
    sxy = np.array([[r[4], r[5]] for r in s], float)
    k = np.clip(np.searchsorted(st, t_log, side="right") - 1, 0, len(st) - 2)
    inside = (t_log >= st[0]) & (t_log <= st[-1])
    same = (sround[k] == sround[k + 1]) & (sidx[k] == sidx[k + 1])
    w = np.clip((t_log - st[k]) / np.maximum(st[k + 1] - st[k], 1e-6), 0, 1)[:, None]
    xy = np.where(same[:, None], sxy[k] * (1 - w) + sxy[k + 1] * w, sxy[k])
    near = inside & ((np.abs(t_log - st[k]) < 60.0) | same)
    phase = np.where(near, sphase[k], "none")

    age = np.full(len(t_log), np.nan)
    remaining = np.full(len(t_log), np.nan)
    for seg in log["segments"]:
        if seg.get("phase") in ("grid", "random", "pursuit") and seg.get("t_end") is not None:
            m = (t_log >= seg["t_start"]) & (t_log < seg["t_end"])
            age[m] = t_log[m] - seg["t_start"]
            remaining[m] = seg["t_end"] - t_log[m]

    target_mm = xy * float(log["mm_per_css_px"])
    eyes = eye_positions_mm(f["lm"], f["pose"], width, height, setup)
    fixed = np.isin(phase, ["grid", "random"])
    return {
        **f,
        "offset_ms": offset,
        "t_log": t_log,
        "phase": phase,
        "round": np.where(near, sround[k], -1).astype(int),
        "point": np.where(near, sidx[k], -1).astype(int),
        "target_mm": target_mm,
        "eye_pos": eyes,
        "cyclops": (eyes["left"] + eyes["right"]) / 2.0,
        "fixation": f["ok"] & fixed & (age >= FIX_SETTLE_MS) & (remaining >= FIX_END_GUARD_MS),
        "pursuit": f["ok"] & (phase == "pursuit") & (age >= PURSUIT_SETTLE_MS),
        "screen_mm": np.array([log["viewport"]["width"], log["viewport"]["height"]]) * float(log["mm_per_css_px"]),
        "width": width,
        "height": height,
    }
