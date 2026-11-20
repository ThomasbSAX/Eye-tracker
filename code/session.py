import json
from pathlib import Path

import numpy as np

import dataset as ds
from screen import Screen, viewport_offset_px

CHECK_FIT = 5


def load(path: Path) -> dict:
    s = json.loads(Path(path).read_text())
    s["snapshots"] = sorted(s.get("snapshots", []), key=lambda r: r["t"])
    return s


def offset_ms(session: dict, data: dict) -> float:
    return ds.synchronize(data["t_ms"], data["bright"], session["flashes"]) if session.get("flashes") else 0.0


def snapshot_index(session: dict, t_ms: np.ndarray) -> np.ndarray:
    times = np.array([r["t"] for r in session["snapshots"]])
    return np.clip(np.searchsorted(times, t_ms, side="right") - 1, 0, len(times) - 1)


def frames(session: dict, t_ms: np.ndarray, px: np.ndarray) -> dict:
    k = snapshot_index(session, t_ms)
    snaps = session["snapshots"]
    off = np.array([viewport_offset_px(r["viewport"]) for r in snaps])[k]
    scroll = np.array([[r.get("scroll_x", 0.0), r.get("scroll_y", 0.0)] for r in snaps])[k]
    client = px - off
    return {"snapshot": k, "client": client, "page": client + scroll,
            "url": np.array([r.get("url", "") for r in snaps], dtype=object)[k]}


def elements_at(session: dict, k: np.ndarray, client: np.ndarray, margin_px: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    names = np.full(len(k), "", dtype=object)
    local = np.full((len(k), 2), np.nan)
    best = np.full(len(k), np.inf)
    for j, snap in enumerate(session["snapshots"]):
        m = np.where(k == j)[0]
        if not len(m):
            continue
        for el in snap.get("aoi", []):
            p = client[m]
            hit = ((p[:, 0] >= el["x"] - margin_px) & (p[:, 0] <= el["x"] + el["w"] + margin_px)
                   & (p[:, 1] >= el["y"] - margin_px) & (p[:, 1] <= el["y"] + el["h"] + margin_px))
            area = el["w"] * el["h"]
            take = m[hit & (area < best[m])]
            names[take] = el["name"]
            best[take] = area
            sx = el.get("natural_w", el["w"]) / max(el["w"], 1e-9)
            sy = el.get("natural_h", el["h"]) / max(el["h"], 1e-9)
            local[take] = (client[take] - [el["x"], el["y"]]) * [sx, sy]
    return names, local


def element_frame(session: dict, k: np.ndarray, client: np.ndarray, name: str) -> np.ndarray:
    out = np.full((len(k), 2), np.nan)
    for j, snap in enumerate(session["snapshots"]):
        m = np.where(k == j)[0]
        el = next((e for e in snap.get("aoi", []) if e["name"] == name), None)
        if len(m) and el:
            s = np.array([el.get("natural_w", el["w"]) / max(el["w"], 1e-9), el.get("natural_h", el["h"]) / max(el["h"], 1e-9)])
            out[m] = (client[m] - [el["x"], el["y"]]) * s
    return out


def check_targets(session: dict, t_ms: np.ndarray, screen: Screen) -> tuple[np.ndarray, np.ndarray]:
    target = np.full((len(t_ms), 2), np.nan)
    point = np.full(len(t_ms), -1)
    for i, c in enumerate(session.get("check", [])):
        snap = session["snapshots"][int(snapshot_index(session, np.array([c["t_start"]]))[0])]
        px = np.array([c["x"], c["y"]]) + viewport_offset_px(snap["viewport"])
        m = (t_ms >= c["t_start"] + ds.FIX_SETTLE_MS) & (t_ms < c["t_end"] - ds.FIX_END_GUARD_MS)
        target[m] = px * screen.mm_per_px
        point[m] = i
    return target, point


def analysis_start_ms(session: dict | None) -> float:
    if session and session.get("check"):
        return float(max(c["t_end"] for c in session["check"]))
    return -np.inf
