import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cv2

import complex_gaze as cg
import dataset as ds
from calibrate import evaluate, load_data
from fixations import one_euro, staircase
from gaze_model import CONTOUR, EYES, IRIS, GazeModel, network_angles, to_head

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("physical", "linear", "full")
LABEL = {"physical": "Géométrie seule", "linear": "Réseau + calibration linéaire", "full": "Modèle complet",
         "mpii_raw": "MPIIFaceGaze brut", "eth_raw": "ETH-XGaze brut"}
COLOR = {"full": "#000000", "linear": "#5a5a5a", "physical": "#9a9a9a", "mpii_raw": "#000000", "eth_raw": "#7a7a7a"}
LS = {"full": "-", "linear": "--", "physical": ":", "mpii_raw": "-.", "eth_raw": (0, (1, 1, 4, 1))}
HATCH = {"full": None, "linear": "////", "physical": "...."}
INK, INK2, GRID, SURFACE = "#000000", "#000000", "#cccccc", "#ffffff"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.grid": False, "axes.linewidth": 0.8, "xtick.direction": "in",
                     "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "font.family": "serif",
                     "font.serif": ["EB Garamond Lining", "EB Garamond", "Latin Modern Roman"], "mathtext.fontset": "custom",
                     "mathtext.rm": "EB Garamond Lining", "mathtext.it": "EB Garamond Lining:italic", "mathtext.bf": "EB Garamond:bold",
                     "font.size": 10, "legend.frameon": True, "legend.edgecolor": INK, "legend.fancybox": False,
                     "legend.framealpha": 1.0, "lines.linewidth": 1.4, "axes.titlesize": 10})


def screen_cm(hit_mm: np.ndarray, camera_mm) -> np.ndarray:
    return np.c_[camera_mm[0] - hit_mm[:, 0], camera_mm[1] - hit_mm[:, 1]] / 10.0


def write_csv(path: Path, header: list[str], rows) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def quantiles(e: np.ndarray) -> list:
    return [len(e), round(float(np.median(e)), 3), round(float(np.mean(e)), 3), round(float(np.percentile(e, 90)), 3)]


def setup_axes_screen(ax, data: dict) -> None:
    w, h = data["screen_mm"] / 10.0
    ax.add_patch(plt.Rectangle((0, 0), w, h, fill=False, ec=INK2, lw=1.0))
    ax.set_xlim(-3, w + 3)
    ax.set_ylim(h + 3, -6)
    ax.set_aspect("equal")
    ax.set_xlabel("x écran (cm)")
    ax.set_ylabel("y écran (cm)")


def read_frame(videos: list[str], index: int) -> np.ndarray:
    for v in videos:
        cap = cv2.VideoCapture(v)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if index < n:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            cap.release()
            return frame
        index -= n
        cap.release()


def detection_figure(path: Path, frame: np.ndarray, data: dict, i: int, model: GazeModel, hit_mm: np.ndarray) -> None:
    lm = data["lm"][i, :, :2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    for e, ax in zip(("left", "right"), axes):
        c = lm[CONTOUR[e]]
        cx, cy = c.mean(0)
        half = 0.68 * float(np.ptp(c[:, 0]))
        x0, x1, y0, y1 = int(cx - half), int(cx + half), int(cy - 0.62 * half), int(cy + 0.62 * half)
        ax.imshow(gray[y0:y1, x0:x1], cmap="gray", extent=(x0, x1, y1, y0))
        ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], c="#ffffff", lw=0.8, marker="o", ms=3, mfc="#ffffff", mec="#000000", mew=0.5)
        p = lm[IRIS[e][0]]
        rho = float(np.max(np.abs(lm[IRIS[e][1:]] - p)))
        ax.plot(p[0] + rho * np.cos(th), p[1] + rho * np.sin(th), c="#ffffff", lw=1.1)
        ax.plot(lm[IRIS[e][1:], 0], lm[IRIS[e][1:], 1], "s", ms=4.5, mfc="#ffffff", mec="#000000", mew=0.6)
        eye = data["eye_pos"][e][i]
        true = ds.unit(ds.screen_points_mm(data["target_mm"][i:i + 1], model.setup())[0] - eye)
        est = ds.unit(np.r_[hit_mm, 0.0] - eye)
        for g, ls, col in ((est, "-", "#ffffff"), (true, (0, (2, 1.2)), "#000000")):
            ax.annotate("", xy=p + 6 * rho * np.array([g[0], -g[1]]), xytext=p,
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3, ls=ls, shrinkA=0, shrinkB=0))
        ax.plot(*p, "+", ms=12, c="#000000", mew=1.8)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
        ax.set_axis_off()
        ax.set_title("œil droit" if e == "left" else "œil gauche", fontsize=9)
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(top=0.9)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / "data" / "calibration" / "calib_20261025_102754.json"))
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--model", default="")
    ap.add_argument("--figures", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--tables", default=str(ROOT / "results" / "data"))
    ap.add_argument("--timeline-round", type=int, default=1)
    ap.add_argument("--video", nargs="*", default=[str(ROOT / "data" / "calibration" / "calib_20261025_102754.mp4")])
    a = ap.parse_args()

    log, cache = Path(a.log).resolve(), Path(a.cache).resolve()
    figs, tabs = Path(a.figures).resolve(), Path(a.tables).resolve()
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)
    name = log.stem
    landmarker = ROOT / "code" / "face_landmarker.task"
    data, focal, prior = load_data([], log, cache, landmarker)
    folds = json.loads((cache / f"{name}_folds.json").read_text())
    model = GazeModel.load(Path(a.model) if a.model else ROOT / "models" / f"{name}.pkl")
    cam = model.camera_mm
    n = len(data["ok"])
    points = sorted(k for k in folds if k.startswith("points"))
    rounds = sorted(k for k in folds if k.startswith("round"))
    rid = [int(r[5:]) for r in rounds]
    target = data["target_mm"] / 10.0
    head = ds.fick(data["pose"][:, :3, 2])
    dist = np.abs(data["pose"][:, 2, 3])

    pred = {s: np.full((n, 2), np.nan) for s in STAGES}
    err = {s: np.full(n, np.nan) for s in STAGES}
    for k in points:
        te = np.array(folds[k]["test"])
        for s in STAGES:
            pred[s][te] = screen_cm(np.array(folds[k][s]["hit_mm"]), folds[k]["camera_mm"])
            err[s][te] = folds[k][s]["cm"]
    pose_pred = np.full((n, 2), np.nan)
    pose_err = np.full(n, np.nan)
    for r in rounds:
        te = np.array(folds[r]["test"])
        pose_pred[te] = screen_cm(np.array(folds[r]["full"]["hit_mm"]), folds[r]["camera_mm"])
        pose_err[te] = folds[r]["full"]["cm"]
        pt = np.array(folds[r]["pursuit"]["test"])
        pose_pred[pt] = screen_cm(np.array(folds[r]["pursuit"]["hit_mm"]), folds[r]["camera_mm"])
        pose_err[pt] = folds[r]["pursuit"]["cm"]

    fi = np.where(data["fixation"])[0]
    raw_err = {}
    for k in ("mpii", "eth"):
        g = ds.from_fick(network_angles(data, fi)[k])
        hit = ds.screen_hit_mm(data["cyclops"][fi], g)
        tgt = ds.screen_points_mm(data["target_mm"][fi], model.setup())
        raw_err[k + "_raw"] = np.linalg.norm((hit - tgt)[:, :2], axis=-1) / 10.0

    cols = ["frame", "t_s", "face", "phase", "round", "point", "target_x_cm", "target_y_cm", "head_yaw_deg", "head_pitch_deg",
            "head_distance_cm", "fixation", "pursuit"]
    for s in STAGES:
        cols += [f"{s}_x_cm", f"{s}_y_cm", f"{s}_error_cm"]
    cols += ["full_unseen_pose_x_cm", "full_unseen_pose_y_cm", "full_unseen_pose_error_cm"]
    r3 = lambda v: "" if not np.isfinite(v) else round(float(v), 3)
    rows = []
    for i in range(n):
        row = [i, r3(data["t_log"][i] / 1000.0), int(data["ok"][i]), data["phase"][i], int(data["round"][i]), int(data["point"][i]),
               r3(target[i, 0]) if data["phase"][i] != "none" else "", r3(target[i, 1]) if data["phase"][i] != "none" else "",
               r3(head[i, 0]) if data["ok"][i] else "", r3(head[i, 1]) if data["ok"][i] else "", r3(dist[i]) if data["ok"][i] else "",
               int(data["fixation"][i]), int(data["pursuit"][i])]
        for s in STAGES:
            row += [r3(pred[s][i, 0]), r3(pred[s][i, 1]), r3(err[s][i])]
        row += [r3(pose_pred[i, 0]), r3(pose_pred[i, 1]), r3(pose_err[i])]
        rows.append(row)
    write_csv(tabs / "frames.csv", cols, rows)

    summary = []
    for s in STAGES:
        e = np.concatenate([folds[k][s]["cm"] for k in points])
        d = np.concatenate([folds[k][s]["deg"] for k in points])
        summary.append([LABEL[s], "cibles jamais vues", *quantiles(e), round(float(np.median(d)), 3), ""])
        e = np.concatenate([folds[r][s]["cm"] for r in rounds])
        d = np.concatenate([folds[r][s]["deg"] for r in rounds])
        summary.append([LABEL[s], "posture jamais vue", *quantiles(e), round(float(np.median(d)), 3),
                        round(float(np.mean([np.median(folds[r][s]["cm"]) for r in rounds])), 3)])
        e = np.concatenate([folds[r][s + "+recal"]["cm"] for r in rounds])
        summary.append([LABEL[s], "posture jamais vue + 5 points", *quantiles(e), "",
                        round(float(np.mean([np.median(folds[r][s + "+recal"]["cm"]) for r in rounds])), 3)])
    for k, e in raw_err.items():
        summary.append([LABEL[k], "sans calibration", *quantiles(e), "", ""])
    write_csv(tabs / "summary.csv", ["model", "protocol", "n", "median_cm", "mean_cm", "p90_cm", "median_deg", "mean_of_round_medians_cm"], summary)

    rrows = []
    for r, j in zip(rounds, rid):
        m = data["fixation"] & (data["round"] == j)
        rrows.append([j + 1, round(float(np.median(head[m, 0])), 2), round(float(np.median(head[m, 1])), 2), round(float(np.median(dist[m])), 1)]
                     + [round(float(np.median(folds[r][s]["cm"])), 3) for s in STAGES]
                     + [round(float(np.median(folds[r][s + "+recal"]["cm"])), 3) for s in STAGES]
                     + [round(float(np.median(folds[r]["pursuit"]["cm"])), 3)])
    write_csv(tabs / "rounds.csv", ["round", "head_yaw_deg", "head_pitch_deg", "head_distance_cm"]
              + [f"{s}_median_cm" for s in STAGES] + [f"{s}_recal_median_cm" for s in STAGES] + ["pursuit_full_median_cm"], rrows)

    key = data["round"] * 1000 + data["point"] + np.where(data["phase"] == "random", 500, 0)
    trows = []
    for g in np.unique(key[fi]):
        m = data["fixation"] & (key == g)
        i0 = np.where(m)[0][0]
        trows.append([int(data["round"][i0]) + 1, data["phase"][i0], int(data["point"][i0]), r3(target[i0, 0]), r3(target[i0, 1]), int(m.sum())]
                     + [r3(np.median(err[s][m])) for s in STAGES])
    write_csv(tabs / "targets.csv", ["round", "phase", "point", "x_cm", "y_cm", "frames"] + [f"{s}_median_cm" for s in STAGES], trows)
    write_csv(tabs / "physical_parameters.csv", ["parameter", "value"],
              [[k, round(float(v), 4)] for k, v in json.loads((tabs / "report.json").read_text())["physical_parameters"].items()])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    setup_axes_screen(ax, data)
    tr = np.array(trows, dtype=object)
    for ph, mk, lab in (("grid", "s", "grille (13 points)"), ("random", "o", "aléatoires (15 points)")):
        sel = tr[:, 1] == ph
        ax.scatter(tr[sel, 3].astype(float), tr[sel, 4].astype(float), marker=mk, s=26, c=INK if ph == "grid" else SURFACE,
                   ec=INK, lw=0.8, label=lab, zorder=3)
    pm = data["pursuit"]
    ax.plot(target[pm, 0], target[pm, 1], ".", ms=1.0, c="#8a8a8a", label="poursuite", zorder=2)
    ax.plot(cam[0] / 10, cam[1] / 10, marker="v", ms=10, c=INK, ls="none", label="webcam")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig01_protocole.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    gaze_head = ds.fick(to_head(data["pose"], ds.unit(ds.screen_points_mm(data["target_mm"], model.setup()) - data["cyclops"])))
    shades = ["#b0b0b0", "#888888", "#5a5a5a", "#2e2e2e", "#000000"]
    for j, c, mk in zip(rid, shades, "osD^v"):
        m = data["fixation"] & (data["round"] == j)
        ax[0].scatter(head[m, 0], head[m, 1], s=6, c=c, marker=mk, lw=0, label=f"tour {j + 1}")
        ax[1].scatter(gaze_head[m, 0], gaze_head[m, 1], s=6, c=c, marker=mk, lw=0, label=f"tour {j + 1}")
        ax[0].annotate(f"tour {j + 1}", (np.median(head[m, 0]), np.median(head[m, 1])), xytext=(0, 9), textcoords="offset points",
                       ha="center", fontsize=9, color=INK)
    ax[0].set_title("Orientation de la tête (caméra)")
    ax[0].set_xlabel("lacet (°)")
    ax[0].set_ylabel("tangage (°)")
    ax[1].set_title("Direction du regard dans le repère tête")
    ax[1].set_xlabel("lacet (°)")
    ax[1].set_ylabel("tangage (°)")
    ax[1].legend(markerscale=4, loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig02_postures.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for axis, s in zip(ax, STAGES):
        setup_axes_screen(axis, data)
        m = np.isfinite(err[s])
        axis.scatter(pred[s][m, 0], pred[s][m, 1], s=2, c="#7a7a7a", alpha=0.35, lw=0)
        axis.scatter(tr[:, 3].astype(float), tr[:, 4].astype(float), marker="+", s=36, c=INK, lw=1.0)
        axis.set_title(f"{LABEL[s]} : médiane {np.nanmedian(err[s]):.2f} cm")
    fig.tight_layout()
    fig.savefig(figs / "fig03_cibles_jamais_vues.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    curves = [(s, err[s][np.isfinite(err[s])]) for s in STAGES] + list(raw_err.items())
    for s, e in curves:
        x = np.sort(e)
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), c=COLOR[s], lw=1.6 if s in STAGES else 1.0,
                ls=LS[s], label=f"{LABEL[s]} ({np.median(e):.2f} cm)")
    ax.axhline(0.5, c=INK, lw=0.5, ls=":")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1)
    ax.set_xlabel("erreur sur l'écran (cm)")
    ax.set_ylabel("fraction des images")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig04_repartition_erreurs.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(rounds))
    for j, s in enumerate(("physical", "linear", "full")):
        v = [np.median(folds[r][s]["cm"]) for r in rounds]
        vr = [np.median(folds[r][s + "+recal"]["cm"]) for r in rounds]
        ax.bar(xs + (j - 1) * 0.27, v, 0.25, color=SURFACE if HATCH[s] else INK, hatch=HATCH[s], ec=INK, lw=0.8, label=LABEL[s], zorder=3)
        ax.scatter(xs + (j - 1) * 0.27, vr, marker="o", s=22, c=SURFACE, ec=INK, lw=1.0, zorder=4, label="avec 5 points de recalibration" if j == 2 else None)
    hm = lambda j, c: np.median(head[data["fixation"] & (data["round"] == j), c])
    ax.set_xticks(xs, [f"tour {j + 1}\nlacet {hm(j, 0):+.0f}°\ntangage {hm(j, 1):+.0f}°" for j in rid])
    ax.set_ylim(0, 9.5)
    ax.set_ylabel("erreur médiane (cm)")
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(figs / "fig05_posture_jamais_vue.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for axis, s in zip(ax, ("physical", "full")):
        setup_axes_screen(axis, data)
        site = np.where(data["phase"] == "grid", data["point"], key)
        xs_, ys_, v = [], [], []
        for g in np.unique(site[fi]):
            m = data["fixation"] & (site == g) & np.isfinite(err[s])
            xs_.append(np.median(target[m, 0]))
            ys_.append(np.median(target[m, 1]))
            v.append(np.median(err[s][m]))
        v = np.array(v)
        sc = axis.scatter(xs_, ys_, c=v, s=40 + 18 * v, cmap="Greys", vmin=0, vmax=8, ec=INK, lw=0.5)
        axis.set_title(f"{LABEL[s]} : erreur médiane par cible")
    fig.colorbar(sc, ax=ax, label="erreur médiane (cm)", shrink=0.85)
    fig.savefig(figs / "fig06_erreur_sur_ecran.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    edges = np.linspace(-8, 8, 65)
    for k, lab in enumerate(("écart horizontal (cm)", "écart vertical (cm)")):
        for s in ("physical", "linear", "full"):
            m = np.isfinite(err[s])
            ax[k].hist(pred[s][m, k] - target[m, k], bins=edges, histtype="step", lw=1.3, color=COLOR[s], ls=LS[s], density=True, label=LABEL[s])
        ax[k].axvline(0, c=INK, lw=0.5, ls=":")
        ax[k].set_xlabel(lab)
    ax[0].set_ylabel("densité")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig07_ecarts_xy.png", dpi=200)
    plt.close(fig)

    j = a.timeline_round
    tr_idx = np.where(data["fixation"] & (data["round"] != j))[0]
    held = GazeModel().fit(data, tr_idx, prior)
    held.focal_px = focal
    span = np.where((data["round"] == j) & np.isin(data["phase"], ["grid", "random", "pursuit"]))[0]
    span = np.arange(span[0], span[-1] + 1)
    t = data["t_log"][span] / 1000.0
    valid = data["ok"][span] & (np.min(data["blend"][span][:, 9:11], 1) < 0.5)
    ang = np.full((len(span), 2), np.nan)
    ang[valid] = held.angles(data, span[valid])
    fps = 1.0 / np.median(np.diff(t))
    smooth = one_euro(t, ang, valid)
    stair, label, events = staircase(t, ang, valid, fps)
    true = held.target_angles(data, span)
    true[data["phase"][span] == "none"] = np.nan
    fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    for k, lab in enumerate(("horizontal (°)", "vertical (°)")):
        ax[k].plot(t - t[0], ang[:, k], c="#b5b5b5", lw=0.6, label="estimation par image")
        ax[k].plot(t - t[0], true[:, k], c=INK, lw=1.0, ls="--", label="cible")
        ax[k].plot(t - t[0], stair[:, k], c=INK, lw=1.3, label="fixations et saccades")
        ax[k].set_ylabel(lab)
    ax[0].legend(loc="upper right", ncol=3, fontsize=9)
    ax[1].set_xlabel("temps depuis le début du tour (s)")
    fig.tight_layout()
    fig.savefig(figs / "fig08_serie_temporelle.png", dpi=200)
    plt.close(fig)
    write_csv(tabs / f"events_round{j + 1}.csv", ["id", "kind", "start_s", "end_s", "duration_ms", "yaw_deg", "pitch_deg", "saccade_deg"],
              [[e[c] for c in ("id", "kind", "start_s", "end_s", "duration_ms", "yaw_deg", "pitch_deg", "saccade_deg")] for e in events])
    write_csv(tabs / f"timeline_round{j + 1}.csv", ["frame", "t_s", "target_yaw_deg", "target_pitch_deg", "yaw_deg", "pitch_deg",
                                                   "one_euro_yaw_deg", "one_euro_pitch_deg", "staircase_yaw_deg", "staircase_pitch_deg", "event"],
              [[int(span[i]), r3(t[i]), r3(true[i, 0]), r3(true[i, 1]), r3(ang[i, 0]), r3(ang[i, 1]), r3(smooth[i, 0]), r3(smooth[i, 1]),
                r3(stair[i, 0]), r3(stair[i, 1]), int(label[i])] for i in range(len(span))])

    fig, ax = plt.subplots(figsize=(7.5, 5))
    setup_axes_screen(ax, data)
    m = data["pursuit"] & (data["round"] == j)
    ax.plot(target[m, 0], target[m, 1], c=INK, lw=1.2, label="cible")
    ax.plot(pose_pred[m, 0], pose_pred[m, 1], c="#8a8a8a", lw=0.9, label="regard estimé (posture jamais vue)")
    ax.set_title(f"Poursuite, tour {j + 1} : médiane {np.nanmedian(pose_err[m]):.2f} cm")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(figs / "fig09_poursuite.png", dpi=200)
    plt.close(fig)
    zt_all = cg.stereo(ds.from_fick(true))
    z_est = cg.stereo(ds.from_fick(ang))
    z_stair = cg.stereo(ds.from_fick(stair))
    z_smooth = cg.stereo(ds.from_fick(smooth))
    dec = cg.decompose(data["pose"][span], ds.from_fick(np.nan_to_num(ang)))
    z_head = dec["head"]
    z_eye = np.where(valid, dec["eye"], np.nan)
    fix_z = np.array([cg.stereo(ds.from_fick(np.array([e["yaw_deg"], e["pitch_deg"]]))) for e in events if e["kind"] == "fixation"])
    held_targets = np.unique(np.round(zt_all[np.isin(data["phase"][span], ["grid", "random"])], 4))
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    cg.draw_plane(ax, z_est, z_head, held_targets, fix_z)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "fig11_plan_complexe.png", dpi=200)
    plt.close(fig)
    fig = plt.figure(figsize=(11, 5.2))
    ax = fig.add_subplot(projection="3d")
    w = (t - t[0]) >= t[-1] - t[0] - 17.0
    cg.draw_trajectory(ax, t[w] - t[0], z_smooth[w], zt_all[w], z_head[w], stem_every_s=0.2)
    ax.set_box_aspect((2.6, 1, 1), zoom=1.3)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=3, fontsize=8)
    fig.savefig(figs / "fig12_trajectoire_complexe.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    c3 = lambda z: ("", "") if not np.isfinite(z) else (round(float(z.real), 5), round(float(z.imag), 5))
    write_csv(tabs / f"complex_round{j + 1}.csv", ["frame", "t_s", "gaze_re", "gaze_im", "one_euro_re", "one_euro_im", "staircase_re", "staircase_im", "head_re", "head_im",
                                                  "eye_in_head_re", "eye_in_head_im", "target_re", "target_im", "eccentricity_deg"],
              [[int(span[i]), r3(t[i]), *c3(z_est[i]), *c3(z_smooth[i]), *c3(z_stair[i]), *c3(z_head[i]), *c3(z_eye[i]), *c3(zt_all[i]),
                r3(cg.eccentricity_deg(z_est[i]))] for i in range(len(span))])
    print("composition tete-oeil, ecart max:", float(np.nanmax(np.abs(dec["composed"] - dec["gaze"]))))

    if a.video and Path(a.video[0]).exists():
        cand = np.where(data["fixation"] & (data["round"] == 0) & (data["phase"] == "grid"))[0]
        mid = np.abs(target[cand, 1] - np.median(target[cand, 1])) < 3
        cand = cand[mid]
        i = int(cand[np.argmax(np.abs(target[cand, 0] - cam[0] / 10.0))])
        k = next(k for k in points if i in folds[k]["test"])
        hit = np.array(folds[k]["full"]["hit_mm"])[folds[k]["test"].index(i)]
        detection_figure(figs / "fig10_detection.png", read_frame(a.video, i), data, i, model, hit)

    print(f"{len(list(figs.glob('*.png')))} figures, {len(list(tabs.glob('*.csv')))} tables")


if __name__ == "__main__":
    main()
