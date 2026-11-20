import argparse
import multiprocessing as mproc
import tempfile
from pathlib import Path

import cv2
import numpy as np
import ptgaze
import torch
from omegaconf import OmegaConf
from ptgaze.common import Face
from ptgaze.config_schema import AppConfig
from ptgaze.gaze_estimator import GazeEstimator

CHECKPOINT_EVERY = 500
WEIGHTS = Path(__file__).resolve().parent / "weights"
CHECKPOINTS = {"eth-xgaze": "eth-xgaze_resnet18.safetensors", "mpiifacegaze": "mpiifacegaze_resnet_simple.safetensors"}
torch.set_num_threads(1)


def estimator(mode: str, focal: float, w: int, h: int, landmarker: Path) -> GazeEstimator:
    root = Path(ptgaze.__file__).parent
    cfg = OmegaConf.merge(OmegaConf.structured(AppConfig), OmegaConf.load(root / f"data/configs/{mode}.yaml"))
    cfg.PACKAGE_ROOT = root.as_posix()
    cam = Path(tempfile.mkdtemp()) / "camera.yaml"
    cam.write_text(f"image_width: {w}\nimage_height: {h}\ncamera_matrix:\n  rows: 3\n  cols: 3\n"
                   f"  data: [{focal}, 0., {w / 2}, 0., {focal}, {h / 2}, 0., 0., 1.]\n"
                   "distortion_coefficients:\n  rows: 1\n  cols: 5\n  data: [0., 0., 0., 0., 0.]\n")
    cfg.gaze_estimator.camera_params = cam.as_posix()
    cfg.gaze_estimator.normalized_camera_params = (root / f"data/normalized_camera_params/{mode}.yaml").as_posix()
    cfg.gaze_estimator.checkpoint = (WEIGHTS / CHECKPOINTS[mode]).as_posix()
    cfg.face_detector.mediapipe_model_path = landmarker.as_posix()
    cfg.device = "cpu"
    if mode == "mpiifacegaze":
        cfg.model.backbone.pretrained = ""
    return GazeEstimator(cfg)


def to_camera_frame(v: np.ndarray) -> np.ndarray:
    return np.array([v[0], -v[1], -v[2]])


def run_piece(job: tuple) -> Path:
    video, lm, ok, ckpt, idx, focal, landmarker = job
    final = ckpt / f"appearance{idx:02d}.npz"
    if final.exists():
        return final
    partial = ckpt / f"appearance{idx:02d}_partial.npz"
    done = dict(np.load(partial)) if partial.exists() else {}
    buf = {k: list(done[k]) if done else [] for k in ("eth", "mpii", "emb")}
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    eth, mpii = estimator("eth-xgaze", focal, w, h, landmarker), estimator("mpiifacegaze", focal, w, h, landmarker)
    emb = {}
    eth._gaze_estimation_model.fc.register_forward_hook(lambda m, i, o: emb.__setitem__("x", i[0][0].numpy()))
    i = 0
    while i < len(lm):
        okf, frame = cap.read()
        if not okf:
            break
        if i < len(buf["eth"]):
            i += 1
            continue
        g_e, g_m, e = np.full(3, np.nan), np.full(3, np.nan), np.zeros(512)
        if ok[i]:
            pts = lm[i, :468, :2].astype(np.float64)
            bbox = np.array([pts.min(0), pts.max(0)])
            face = Face(bbox, pts)
            eth.estimate_gaze(frame, face)
            g_e, e = to_camera_frame(face.gaze_vector), emb["x"]
            face = Face(bbox, pts)
            mpii.estimate_gaze(frame, face)
            g_m = to_camera_frame(face.gaze_vector)
        buf["eth"].append(g_e)
        buf["mpii"].append(g_m)
        buf["emb"].append(e.astype(np.float16))
        i += 1
        if len(buf["eth"]) % CHECKPOINT_EVERY == 0:
            np.savez(partial, **{k: np.array(v) for k, v in buf.items()})
            print(f"{video.name}: {i}/{len(lm)}", flush=True)
    cap.release()
    np.savez(final, **{k: np.array(v) for k, v in buf.items()})
    partial.unlink(missing_ok=True)
    return final


def appearance(videos: list[Path], features: Path, out: Path, name: str, focal: float, landmarker: Path,
               workers: int = mproc.cpu_count()) -> Path:
    target = out / f"{name}_appearance.npz"
    if target.exists():
        return target
    f = np.load(features)
    lm, ok = f["lm"], f["ok"]
    counts = [int(cv2.VideoCapture(str(v)).get(cv2.CAP_PROP_FRAME_COUNT)) for v in videos]
    bounds = np.r_[0, np.cumsum(counts)]
    ckpt = out / f"{name}_checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    jobs = [(v, lm[bounds[k]:bounds[k + 1]], ok[bounds[k]:bounds[k + 1]], ckpt, k, focal, landmarker) for k, v in enumerate(videos)]
    with mproc.Pool(min(workers, len(jobs))) as pool:
        parts = [dict(np.load(p)) for p in pool.map(run_piece, jobs)]
    np.savez_compressed(target, **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]})
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--features", required=True)
    ap.add_argument("--focal", type=float, required=True)
    ap.add_argument("--out", default="cache")
    ap.add_argument("--name", required=True)
    ap.add_argument("--landmarker", default=str(Path(__file__).resolve().parent / "face_landmarker.task"))
    a = ap.parse_args()
    print(appearance([Path(v) for v in a.videos], Path(a.features), Path(a.out), a.name, a.focal, Path(a.landmarker)))


if __name__ == "__main__":
    main()
