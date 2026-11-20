import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

import dataset as ds
import screen as sc
from appearance import appearance
from extract import extract
from eye_patches import eye_patches
from gaze_model import NAMES, GazeModel

ROOT = Path(__file__).resolve().parents[1]
RECAL_POINTS = 5
STAGES = ("physical", "linear", "full")


def load_data(videos: list[Path], log: Path, cache: Path, landmarker: Path) -> tuple[dict, float, tuple]:
    name = log.stem
    feats = extract(videos, landmarker, cache, name)
    raw = np.load(feats)
    focal = ds.focal_from_iris(raw["lm"], raw["pose"], raw["ok"])
    app = np.load(appearance(videos, feats, cache, name, focal, landmarker))
    meta = json.loads(log.read_text())
    screen = sc.from_log(meta)
    cam = screen.camera_mm - sc.viewport_offset_px(meta["viewport"]) * screen.mm_per_px
    prior = (float(cam[0]), float(cam[1]))
    data = ds.build(feats, log, ds.Setup(focal_px=focal, camera_x_mm=prior[0], camera_y_mm=prior[1]))
    data.update({"app_mpii": app["mpii"], "app_eth": app["eth"], "app_emb": app["emb"],
                 "patches": np.load(eye_patches(videos, feats, cache, name))})
    data["screen"] = screen
    return data, focal, prior


def evaluate(model: GazeModel, data: dict, idx: np.ndarray, stage: str) -> dict:
    target = ds.screen_points_mm(data["target_mm"][idx], model.setup())
    g = model.camera_gaze(data, idx, stage)
    true = ds.unit(target - data["cyclops"][idx])
    hit = ds.screen_hit_mm(data["cyclops"][idx], g)
    return {"deg": np.degrees(np.arccos(np.clip(np.sum(g * true, -1), -1, 1))),
            "cm": np.linalg.norm((hit - target)[:, :2], axis=-1) / 10.0, "hit": hit}


def stats(e: np.ndarray) -> dict:
    return {"n": int(len(e)), "median": round(float(np.median(e)), 2), "mean": round(float(np.mean(e)), 2),
            "p90": round(float(np.percentile(e, 90)), 2)}


def folds_of(data: dict) -> list[tuple]:
    fix = data["fixation"]
    fi = np.where(fix)[0]
    groups = data["round"] * 1000 + data["point"] + np.where(data["phase"] == "random", 500, 0)
    jobs = [(f"points{j}", fi[tr], fi[te], None) for j, (tr, te) in enumerate(GroupKFold(5).split(fi, groups=groups[fi]))]
    for r in sorted(set(data["round"][fix].tolist())):
        held = fix & (data["round"] == r)
        quick = held & (data["phase"] == "grid") & (data["point"] < RECAL_POINTS)
        jobs.append((f"round{r}", np.where(fix & (data["round"] != r))[0], np.where(held & ~quick)[0], np.where(quick)[0]))
    return jobs


def run_fold(data: dict, focal: float, prior: tuple, tr: np.ndarray, te: np.ndarray, quick: np.ndarray | None, key: str) -> dict:
    model = GazeModel().fit(data, tr, prior)
    model.focal_px = focal
    res = {"test": te.tolist(), "camera_mm": [round(float(v), 2) for v in model.camera_mm]}
    for stage in STAGES:
        e = evaluate(model, data, te, stage)
        res[stage] = {"cm": np.round(e["cm"], 3).tolist(), "deg": np.round(e["deg"], 3).tolist(),
                      "hit_mm": np.round(e["hit"][:, :2], 1).tolist()}
        if quick is not None:
            model.quick_recalibrate(data, quick, stage)
            res[stage + "+recal"] = {"cm": np.round(evaluate(model, data, te, stage)["cm"], 3).tolist()}
            model.recal = np.zeros((3, 2))
    if quick is not None:
        pursuit = np.where(data["pursuit"] & (data["round"] == int(key[5:])))[0]
        e = evaluate(model, data, pursuit, "full")
        res["pursuit"] = {"test": pursuit.tolist(), "cm": np.round(e["cm"], 3).tolist(), "hit_mm": np.round(e["hit"][:, :2], 1).tolist()}
    return res


def report_of(folds: dict, data: dict, focal: float, name: str) -> dict:
    rounds = sorted(k for k in folds if k.startswith("round"))
    points = sorted(k for k in folds if k.startswith("points"))
    report = {"recording": name, "frames": int(len(data["ok"])), "fixation_frames": int(data["fixation"].sum()),
              "sync_offset_ms": data["offset_ms"], "focal_px": round(focal, 1)}
    for stage in STAGES:
        report[f"{stage}_points_never_seen"] = {u: stats(np.concatenate([folds[k][stage][u] for k in points])) for u in ("cm", "deg")}
        per = {r: {"cm": stats(np.array(folds[r][stage]["cm"])), "cm_recal": stats(np.array(folds[r][stage + "+recal"]["cm"]))} for r in rounds}
        report[f"{stage}_pose_never_seen"] = {"mean_of_medians_cm": round(float(np.mean([v["cm"]["median"] for v in per.values()])), 2),
                                              "mean_of_medians_recal_cm": round(float(np.mean([v["cm_recal"]["median"] for v in per.values()])), 2),
                                              "rounds": per}
    report["pursuit_cm"] = {r: stats(np.array(folds[r]["pursuit"]["cm"])) for r in rounds}
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--log", required=True)
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--results", default=str(ROOT / "results" / "data"))
    ap.add_argument("--landmarker", default=str(ROOT / "code" / "face_landmarker.task"))
    a = ap.parse_args()

    cache, models, results = (Path(p).resolve() for p in (a.cache, a.models, a.results))
    for d in (cache, models, results):
        d.mkdir(parents=True, exist_ok=True)
    log = Path(a.log).resolve()
    name = log.stem
    data, focal, prior = load_data([Path(v) for v in a.videos], log, cache, Path(a.landmarker))
    print(f"synchro {data['offset_ms']:.0f} ms, {int(data['fixation'].sum())} fixation frames, focal length {focal:.0f} px", flush=True)

    folds_path = cache / f"{name}_folds.json"
    folds = json.loads(folds_path.read_text()) if folds_path.exists() else {}
    for key, tr, te, quick in folds_of(data):
        if key in folds:
            continue
        folds[key] = run_fold(data, focal, prior, tr, te, quick, key)
        folds_path.write_text(json.dumps(folds))
        print(key + ": " + ", ".join(f"{s} {np.median(folds[key][s]['cm']):.2f} cm" for s in STAGES), flush=True)

    report = report_of(folds, data, focal, name)
    model = GazeModel().fit(data, np.where(data["fixation"])[0], prior)
    model.focal_px = focal
    model.screen = asdict(data["screen"])
    model.save(models / f"{name}.pkl")
    data["screen"].save(models / f"{name}_screen.json")
    report["physical_parameters"] = {n: round(float(v), 4) for n, v in zip(NAMES, model.params)}
    (results / "report.json").write_text(json.dumps(report, indent=1))
    for s in STAGES:
        print(s, report[f"{s}_points_never_seen"]["cm"], report[f"{s}_pose_never_seen"]["mean_of_medians_cm"],
              report[f"{s}_pose_never_seen"]["mean_of_medians_recal_cm"])


if __name__ == "__main__":
    main()
