import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon

NONE = "(no area)"
COLUMNS = {"screen": ("x_px", "y_px"), "client": ("client_x", "client_y"), "page": ("page_x", "page_y")}
FIELDS = ["aoi", "fixations", "dwell_ms", "dwell_share", "time_to_first_fixation_ms", "first_fixation_ms",
          "mean_fixation_ms", "visits", "revisits", "gaze_ms", "clicks", "time_to_first_click_ms"]


def load_areas(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    return {"space": spec.get("space", "screen"), "margin_px": float(spec.get("margin_px", 0.0)), "areas": spec["areas"]}


def contains(area: dict, xy: np.ndarray, margin: float) -> tuple[np.ndarray, float]:
    if "polygon" in area:
        poly = np.array(area["polygon"], float)
        a = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1)))
        return Polygon(poly).contains_points(np.nan_to_num(xy, nan=-1e9), radius=-margin), a
    if "r" in area:
        r = area["r"] + margin
        return np.hypot(xy[:, 0] - area["cx"], xy[:, 1] - area["cy"]) <= r, np.pi * area["r"] ** 2
    x, y, w, h = area["x"] - margin, area["y"] - margin, area["w"] + 2 * margin, area["h"] + 2 * margin
    return (xy[:, 0] >= x) & (xy[:, 0] <= x + w) & (xy[:, 1] >= y) & (xy[:, 1] <= y + h), area["w"] * area["h"]


def label(xy: np.ndarray, areas: list[dict], margin: float = 0.0) -> np.ndarray:
    names = np.full(len(xy), "", dtype=object)
    best = np.full(len(xy), np.inf)
    for area in areas:
        hit, size = contains(area, xy, margin)
        take = hit & (size < best)
        names[take] = area["name"]
        best[take] = size
    return names


def report(t_ms: np.ndarray, sample_labels: np.ndarray, valid: np.ndarray, fixations: list[dict], clicks: list[dict],
           start_ms: float, names: list[str]) -> list[dict]:
    keep = valid & (t_ms >= start_ms)
    dt = np.diff(t_ms, append=t_ms[-1] + np.median(np.diff(t_ms)))
    t0 = max(start_ms, float(t_ms[0]))
    fix = [f for f in fixations if f["start_ms"] >= start_ms]
    total = sum(f["duration_ms"] for f in fix) or 1.0
    runs = [f["aoi"] or NONE for f in fix]
    visits = {n: sum(1 for j, r in enumerate(runs) if r == n and (j == 0 or runs[j - 1] != n)) for n in set(runs)}
    lab = np.where(sample_labels == "", NONE, sample_labels)
    rows = []
    for n in names + [NONE]:
        mine = [f for f in fix if (f["aoi"] or NONE) == n]
        clicked = [c for c in clicks if (c.get("aoi") or NONE) == n and c["t"] >= start_ms]
        rows.append({"aoi": n, "fixations": len(mine),
                     "dwell_ms": round(sum(f["duration_ms"] for f in mine), 1),
                     "dwell_share": round(100.0 * sum(f["duration_ms"] for f in mine) / total, 1),
                     "time_to_first_fixation_ms": round(mine[0]["start_ms"] - t0, 1) if mine else "",
                     "first_fixation_ms": mine[0]["duration_ms"] if mine else "",
                     "mean_fixation_ms": round(float(np.mean([f["duration_ms"] for f in mine])), 1) if mine else "",
                     "visits": visits.get(n, 0), "revisits": max(visits.get(n, 0) - 1, 0),
                     "gaze_ms": round(float(dt[keep & (lab == n)].sum()), 1), "clicks": len(clicked),
                     "time_to_first_click_ms": round(clicked[0]["t"] - t0, 1) if clicked else ""})
    return rows


def clean(v):
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        return round(v, 5) if np.isfinite(v) else None
    return v


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows([{k: "" if x is None else x for k, x in clean(r).items()} for r in rows])


def read_csv(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def number(v) -> float:
    return float(v) if v not in ("", None) else np.nan


def coordinates(rows: list[dict], space: str) -> np.ndarray:
    if space in COLUMNS:
        cx, cy = COLUMNS[space]
        return np.array([[number(r.get(cx)), number(r.get(cy))] for r in rows]).reshape(-1, 2)
    return np.array([[number(r["element_x"]), number(r["element_y"])] if r.get("element") == space else [np.nan, np.nan]
                     for r in rows]).reshape(-1, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("areas")
    ap.add_argument("--out")
    a = ap.parse_args()
    folder = Path(a.folder)
    spec = load_areas(Path(a.areas))
    gaze = read_csv(folder / "gaze.csv")
    fixes = [r for r in read_csv(folder / "fixations.csv") if r["kind"] == "fixation"]
    sample_labels = label(coordinates(gaze, spec["space"]), spec["areas"], spec["margin_px"])
    fix_labels = label(coordinates(fixes, spec["space"]), spec["areas"], spec["margin_px"])
    meta = json.loads((folder / "gaze.json").read_text())["meta"]
    clicks = [dict(c, aoi=n) for c, n in zip(meta["clicks"], label(coordinates(meta["clicks"], spec["space"]), spec["areas"], 0.0))]
    rows = report(np.array([number(r["t_ms"]) for r in gaze]), sample_labels, np.array([r["status"] == "ok" for r in gaze]),
                  [{"start_ms": number(f["start_ms"]), "duration_ms": number(f["duration_ms"]), "aoi": n} for f, n in zip(fixes, fix_labels)],
                  clicks, meta["analysis_start_ms"], [x["name"] for x in spec["areas"]])
    out = Path(a.out) if a.out else folder / f"aoi_{Path(a.areas).stem}.csv"
    write_csv(out, rows, FIELDS)
    print(f"{len(rows) - 1} areas -> {out}")


if __name__ == "__main__":
    main()
