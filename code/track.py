import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import aoi
import complex_gaze as cg
import dataset as ds
import heatmap
import session as sess
from appearance import appearance
from extract import extract
from eye_patches import eye_patches
from fixations import blinks, one_euro, staircase
from gaze_model import GazeModel, to_head
from screen import Screen

ROOT = Path(__file__).resolve().parents[1]
WARN_CM = 3.0


def render(videos: list[Path], out: Path, data: dict, cam: np.ndarray, stair: np.ndarray, valid: np.ndarray) -> None:
    w, h = (int(v) for v in data["size"])
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    i = 0
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        while True:
            ok, frame = cap.read()
            if not ok or i >= len(valid):
                break
            if valid[i]:
                for idx in (468, 473):
                    o = data["lm"][i, idx, :2]
                    for g, color, th in ((cam[i], (255, 200, 0), 1), (stair[i], (255, 255, 255), 3)):
                        z = max(g[2], 0.25)
                        end = (int(o[0] + g[0] / z * 170), int(o[1] - g[1] / z * 170))
                        cv2.arrowedLine(frame, (int(o[0]), int(o[1])), end, color, th, cv2.LINE_AA, tipLength=0.22)
                yaw, pitch = ds.fick(stair[i])
                cv2.putText(frame, f"yaw {yaw:+.1f}  pitch {pitch:+.1f}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            i += 1
        cap.release()
    writer.release()


def plots(out: Path, t: np.ndarray, raw: np.ndarray, smooth: np.ndarray, stair: np.ndarray, label: np.ndarray,
          z_gaze: np.ndarray, z_head: np.ndarray) -> None:
    fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for k, lab in enumerate(("horizontal (deg)", "vertical (deg)")):
        ax[k].plot(t, raw[:, k], lw=0.5, c="#bbb", label="raw")
        ax[k].plot(t, smooth[:, k], lw=0.8, c="#36c", label="One Euro")
        ax[k].plot(t, stair[:, k], lw=1.4, c="k", label="fixations")
        ax[k].set_ylabel(lab)
    ax[0].legend(loc="upper right")
    ax[1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "timeline.png", dpi=110)
    plt.close(fig)
    fig = plt.figure(figsize=(12, 5))
    z_stair = cg.stereo(ds.from_fick(stair))
    ax3 = fig.add_subplot(1, 2, 1)
    cg.draw_plane(ax3, z_gaze, z_head, None, z_stair[np.r_[True, np.diff(label) != 0] & (label >= 0)])
    ax3.legend(loc="lower left", fontsize=8)
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    cg.draw_trajectory(ax3, t - t[0], z_gaze, None, z_head)
    fig.tight_layout()
    fig.savefig(out / "complex.png", dpi=110)
    plt.close(fig)


def recalibrate(model: GazeModel, data: dict, point: np.ndarray, valid: np.ndarray, screen: Screen) -> dict:
    used = valid & (point >= 0)
    ids = sorted(set(point[used].tolist()))
    if len(ids) < 3:
        return {"points": len(ids)}
    error = lambda idx: float(np.median(np.linalg.norm(
        ds.screen_hit_mm(data["cyclops"][idx], model.camera_gaze(data, idx))[:, :2]
        - ds.screen_points_mm(data["target_mm"][idx], model.setup())[:, :2], axis=-1))) / 10.0
    fit = np.where(used & (point < sess.CHECK_FIT))[0]
    test = np.where(used & (point >= sess.CHECK_FIT))[0]
    res = {"points": len(ids)}
    if len(test) and len(fit):
        res["error_cm_before"] = round(error(test), 2)
        model.quick_recalibrate(data, fit)
        res["error_cm_after"] = round(error(test), 2)
        res["error_px_after"] = round(res["error_cm_after"] * 10.0 / screen.mm_per_px)
        res["held_out_points"] = len(set(point[test].tolist()))
    res["applied"] = bool(res.get("error_cm_after", 0.0) <= res.get("error_cm_before", 0.0))
    model.recal = np.zeros((3, 2))
    if res["applied"]:
        model.quick_recalibrate(data, np.where(used)[0])
    res["measured_error_cm"] = res.get("error_cm_after" if res["applied"] else "error_cm_before")
    if (res["measured_error_cm"] or 0.0) > WARN_CM:
        print(f"warning: measured error on the check points is {res['measured_error_cm']} cm; "
              "check screen.json, lighting and posture, or recalibrate", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--screen-model")
    ap.add_argument("--session")
    ap.add_argument("--aoi", action="append", default=[])
    ap.add_argument("--background")
    ap.add_argument("--heatmap-space", default="screen")
    ap.add_argument("--no-recal", action="store_true")
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--out", default=str(ROOT / "results" / "tracking"))
    ap.add_argument("--landmarker", default=str(ROOT / "code" / "face_landmarker.task"))
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--plots", action="store_true")
    a = ap.parse_args()

    videos = [Path(v).resolve() for v in a.videos]
    name = videos[0].stem
    cache, out = Path(a.cache).resolve(), Path(a.out).resolve() / name
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    feats = extract(videos, Path(a.landmarker), cache, name)
    data = dict(np.load(feats))
    ok = data["ok"]
    focal = ds.focal_from_iris(data["lm"], data["pose"], ok)
    app = np.load(appearance(videos, feats, cache, name, focal, Path(a.landmarker)))
    data.update({"app_mpii": app["mpii"], "app_eth": app["eth"], "app_emb": app["emb"],
                 "patches": np.load(eye_patches(videos, feats, cache, name))})
    model = GazeModel.load(Path(a.model))
    model.adapt(data)
    beside = Path(a.model).with_name(Path(a.model).stem + "_screen.json")
    screen = Screen.load(Path(a.screen_model or beside)) if a.screen_model or not model.screen else Screen(**model.screen)
    model.camera_mm = tuple(screen.camera_mm)
    w, h = (int(v) for v in data["size"])
    eyes = ds.eye_positions_mm(data["lm"], data["pose"], w, h, model.setup())
    data["cyclops"] = (eyes["left"] + eyes["right"]) / 2.0

    session = sess.load(Path(a.session)) if a.session else None
    offset = sess.offset_ms(session, data) if session else 0.0
    t_ms = data["t_ms"] - offset
    t = t_ms / 1000.0
    fps = 1.0 / np.median(np.diff(t))
    valid = ok & ~blinks(data["blend"], ok, fps)
    check = {"points": 0}
    if session and session.get("check") and not a.no_recal:
        data["target_mm"], point = sess.check_targets(session, t_ms, screen)
        check = recalibrate(model, data, point, valid, screen)
    start_ms = sess.analysis_start_ms(session) if session else float(t_ms[0])

    idx = np.where(valid)[0]
    raw = np.full((len(t), 2), np.nan)
    raw[idx] = model.angles(data, idx)
    head = np.full((len(t), 2), np.nan)
    head[idx] = ds.fick(to_head(data["pose"][idx], ds.from_fick(raw[idx])))
    smooth = one_euro(t, raw, valid)
    stair, label, events = staircase(t, raw, valid, fps)
    dec = cg.decompose(data["pose"], ds.from_fick(np.nan_to_num(smooth)))
    z_gaze = np.where(valid, dec["gaze"], np.nan)
    z_eye = np.where(valid, dec["eye"], np.nan)
    z_head = np.where(ok, dec["head"], np.nan)

    mm = screen.mm_from_hit(ds.screen_hit_mm(data["cyclops"], ds.from_fick(smooth)))
    mm[~valid] = np.nan
    px = screen.px_from_mm(mm)
    stair_px = screen.px_from_mm(screen.mm_from_hit(ds.screen_hit_mm(data["cyclops"], ds.from_fick(stair))))
    fixations = []
    for e in events:
        m = label == e["id"]
        fixations.append({"id": e["id"], "kind": e["kind"], "start_ms": round(e["start_s"] * 1000.0, 1),
                          "end_ms": round(e["end_s"] * 1000.0, 1), "duration_ms": e["duration_ms"],
                          "x_px": float(np.nanmean(stair_px[m, 0])), "y_px": float(np.nanmean(stair_px[m, 1])),
                          "yaw_deg": e["yaw_deg"], "pitch_deg": e["pitch_deg"], "saccade_deg": e["saccade_deg"]})
    fxy = np.array([[f["x_px"], f["y_px"]] for f in fixations]).reshape(-1, 2)
    fmid = np.array([(f["start_ms"] + f["end_ms"]) / 2.0 for f in fixations])

    phase = np.where(t_ms < start_ms, "check", "session") if session else np.full(len(t), "session")
    columns = {"t_ms": np.round(t_ms, 1), "frame": np.arange(len(t)), "status": np.where(valid, "ok", np.where(ok, "blink", "no_face")),
               "phase": phase, "x_px": np.round(px[:, 0], 1), "y_px": np.round(px[:, 1], 1),
               "x_mm": np.round(mm[:, 0], 2), "y_mm": np.round(mm[:, 1], 2), "on_screen": screen.inside(px).astype(int),
               "fixation_id": label, "yaw_deg": np.round(smooth[:, 0], 3), "pitch_deg": np.round(smooth[:, 1], 3),
               "head_distance_cm": np.round(np.abs(data["pose"][:, 2, 3]), 2)}
    for f in fixations:
        f["x_mm"], f["y_mm"] = round(f["x_px"] * screen.mm_per_px, 2), round(f["y_px"] * screen.mm_per_px, 2)
    clicks = []
    spaces = {"screen": (px, fxy)}
    if session:
        g = sess.frames(session, t_ms, px)
        fk = sess.snapshot_index(session, fmid)
        snaps = session["snapshots"]
        foff = np.array([sess.viewport_offset_px(snaps[k]["viewport"]) for k in fk]).reshape(-1, 2)
        fscroll = np.array([[snaps[k].get("scroll_x", 0.0), snaps[k].get("scroll_y", 0.0)] for k in fk]).reshape(-1, 2)
        fclient = fxy - foff
        el, local = sess.elements_at(session, g["snapshot"], g["client"])
        fel, flocal = sess.elements_at(session, fk, fclient)
        columns.update({"client_x": np.round(g["client"][:, 0], 1), "client_y": np.round(g["client"][:, 1], 1),
                        "page_x": np.round(g["page"][:, 0], 1), "page_y": np.round(g["page"][:, 1], 1),
                        "element": el, "element_x": np.round(local[:, 0], 1), "element_y": np.round(local[:, 1], 1),
                        "url": g["url"]})
        for j, f in enumerate(fixations):
            f.update({"client_x": round(float(fclient[j, 0]), 1), "client_y": round(float(fclient[j, 1]), 1),
                      "page_x": round(float(fclient[j, 0] + fscroll[j, 0]), 1), "page_y": round(float(fclient[j, 1] + fscroll[j, 1]), 1),
                      "element": fel[j], "element_x": round(float(flocal[j, 0]), 1), "element_y": round(float(flocal[j, 1]), 1),
                      "url": snaps[fk[j]].get("url", "")})
        spaces.update({"client": (g["client"], fclient), "page": (g["page"], fclient + fscroll)})
        for c in [e for e in session.get("events", []) if e.get("type") == "click"]:
            k = int(sess.snapshot_index(session, np.array([c["t"]]))[0])
            cpx = np.array([c["x"], c["y"]]) + sess.viewport_offset_px(snaps[k]["viewport"])
            cel, clocal = sess.elements_at(session, np.array([k]), np.array([[c["x"], c["y"]]]))
            clicks.append({"t": c["t"], "t_ms": c["t"], "x_px": float(cpx[0]), "y_px": float(cpx[1]), "client_x": c["x"], "client_y": c["y"],
                           "page_x": c["x"] + snaps[k].get("scroll_x", 0.0), "page_y": c["y"] + snaps[k].get("scroll_y", 0.0),
                           "element": c.get("element") or cel[0], "element_x": float(clocal[0, 0]), "element_y": float(clocal[0, 1])})

    rows = [aoi.clean({k: v[i] for k, v in columns.items()}) for i in range(len(t))]
    aoi.write_csv(out / "gaze.csv", rows, list(columns))
    ffields = ["id", "kind", "start_ms", "end_ms", "duration_ms", "x_px", "y_px", "x_mm", "y_mm", "yaw_deg", "pitch_deg", "saccade_deg"]
    if session:
        ffields += ["client_x", "client_y", "page_x", "page_y", "element", "element_x", "element_y", "url"]
    for f in fixations:
        f["x_px"], f["y_px"] = round(f["x_px"], 1), round(f["y_px"], 1)
    aoi.write_csv(out / "fixations.csv", fixations, ffields)

    reports = {}
    fix_only = [f for f in fixations if f["kind"] == "fixation"]
    if session and any(s.get("aoi") for s in session["snapshots"]):
        names = sorted({e["name"] for s in session["snapshots"] for e in s.get("aoi", [])})
        reports["elements"] = aoi.report(t_ms, columns["element"], valid, [dict(f, aoi=f["element"]) for f in fix_only],
                                         [dict(c, aoi=c["element"]) for c in clicks], start_ms, names)
    for path in a.aoi:
        spec = aoi.load_areas(Path(path))
        sp = spec["space"]
        if sp in spaces:
            gxy, fx = spaces[sp]
        else:
            gxy = sess.element_frame(session, g["snapshot"], g["client"], sp)
            fx = sess.element_frame(session, fk, fclient, sp)
        fl = aoi.label(fx, spec["areas"], spec["margin_px"])
        kinds = [f["kind"] for f in fixations]
        reports[Path(path).stem] = aoi.report(t_ms, aoi.label(gxy, spec["areas"], spec["margin_px"]), valid,
                                              [dict(f, aoi=n) for f, n, k in zip(fixations, fl, kinds) if k == "fixation"],
                                              [dict(c, aoi=n) for c, n in zip(clicks, aoi.label(aoi.coordinates(clicks, sp), spec["areas"]))],
                                              start_ms, [x["name"] for x in spec["areas"]])
    for k, r in reports.items():
        aoi.write_csv(out / f"aoi_{k}.csv", r, aoi.FIELDS)

    r3 = lambda v: [round(float(x), 3) for x in v] if np.all(np.isfinite(v)) else None
    rc = lambda z: [round(float(z.real), 5), round(float(z.imag), 5)] if np.isfinite(z) else None
    meta = {"videos": [str(v) for v in videos], "model": str(a.model), "screen": asdict(screen), "fps": round(fps, 3),
            "sync_offset_ms": offset, "analysis_start_ms": start_ms, "recalibration": check,
            "session": {k: session[k] for k in ("url", "created", "user_agent") if k in session} if session else None,
            "clicks": clicks, "coordinates": "CSS pixels from the top-left corner of the screen (x to the right, y down); same for mm",
            "angles": "Fick (yaw, pitch), camera frame, degrees",
            "complex": "z = (right + i up) / (1 + forward), stereographic projection, the person is at 0"}
    extra = [{"gaze_raw_deg": r3(raw[i]), "gaze_in_head_deg": r3(head[i]), "head_deg": r3(ds.fick(data["pose"][i, :3, 2])),
              "gaze_complex": rc(z_gaze[i]), "head_complex": rc(z_head[i]), "eye_in_head_complex": rc(z_eye[i])} for i in range(len(t))]
    (out / "gaze.json").write_text(json.dumps(aoi.clean({"meta": meta, "fixations": fixations, "aoi": reports,
                                                         "frames": [dict(r, **x) for r, x in zip(rows, extra)]})))

    hs = a.heatmap_space
    hxy = spaces[hs][1] if hs in spaces else sess.element_frame(session, fk, fclient, hs)
    keep = np.array([f["kind"] == "fixation" and f["start_ms"] >= start_ms for f in fixations], bool)
    bg = Path(a.background) if a.background else None
    size = (screen.width_px, screen.height_px) if hs == "screen" else plt.imread(str(bg)).shape[1::-1] if bg else (
        np.nanmax(hxy[keep, 0]) + 50, np.nanmax(hxy[keep, 1]) + 50)
    heatmap.draw(out / f"heatmap_{hs}.png", hxy[keep], np.array([f["duration_ms"] for f in fixations])[keep], size,
                 10.0 / screen.mm_per_px, bg)
    if a.plots:
        plots(out, t, raw, smooth, stair, label, z_gaze, z_head)
    if a.video:
        render(videos, out / "video.mp4", data, ds.from_fick(smooth), ds.from_fick(stair), valid & np.isfinite(stair[:, 0]))
    fix = [f["duration_ms"] for f in fix_only]
    print(f"{len(fix)} fixations, median {np.median(fix) if fix else float('nan'):.0f} ms, "
          f"recalibration {check} -> {out}")


if __name__ == "__main__":
    main()
