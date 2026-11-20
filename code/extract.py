import argparse
import multiprocessing as mproc
import subprocess
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

CHECKPOINT_EVERY = 500
KEYS = ("t_ms", "ok", "lm", "pose", "blend", "bright")


def first_pts_ms(video: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "frame=pts_time",
                          "-of", "csv=p=0", "-read_intervals", "%+#1", str(video)], capture_output=True, text=True).stdout
    return 1000.0 * float(out.split()[0].strip(","))


def landmarker(model: Path):
    return mp_vision.FaceLandmarker.create_from_options(mp_vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp_vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))


def run_piece(job: tuple) -> Path:
    video, model, ckpt, idx = job
    final = ckpt / f"landmarks{idx:02d}.npz"
    if final.exists():
        return final
    partial = ckpt / f"landmarks{idx:02d}_partial.npz"
    done = dict(np.load(partial)) if partial.exists() else {}
    resume_at = len(done["t_ms"]) if done else 0
    buf = {k: list(done[k]) if done else [] for k in KEYS}
    t0 = first_pts_ms(video)
    det = landmarker(model)
    cap = cv2.VideoCapture(str(video))
    fi, last_ts = 0, -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = t0 + float(cap.get(cv2.CAP_PROP_POS_MSEC))
        if fi < resume_at - 30:
            fi += 1
            continue
        ts = max(int(round(t)), last_ts + 1)
        last_ts = ts
        h, w = frame.shape[:2]
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        if fi >= resume_at:
            found = bool(res.face_landmarks and res.facial_transformation_matrixes)
            lm = np.zeros((478, 3), np.float32)
            pose = np.eye(4, dtype=np.float32)
            blend = np.zeros(52, np.float32)
            if found:
                lm = np.array([[p.x * w, p.y * h, p.z * w] for p in res.face_landmarks[0]], np.float32)
                pose = np.array(res.facial_transformation_matrixes[0], np.float32)
                if res.face_blendshapes:
                    blend = np.array([c.score for c in res.face_blendshapes[0]], np.float32)
            bright = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean())
            for k, v in zip(KEYS, (t, found, lm, pose, blend, bright)):
                buf[k].append(v)
            if len(buf["t_ms"]) % CHECKPOINT_EVERY == 0:
                np.savez(partial, **{k: np.array(v) for k, v in buf.items()})
                print(f"{video.name}: {fi + 1} frames", flush=True)
        fi += 1
    det.close()
    cap.release()
    np.savez(final, **{k: np.array(v) for k, v in buf.items()})
    partial.unlink(missing_ok=True)
    return final


def extract(videos: list[Path], model: Path, out: Path, name: str, workers: int = mproc.cpu_count()) -> Path:
    target = out / f"{name}_features.npz"
    if target.exists():
        return target
    ckpt = out / f"{name}_checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    jobs = [(Path(v).resolve(), Path(model).resolve(), ckpt, i) for i, v in enumerate(videos)]
    with mproc.Pool(min(workers, len(jobs))) as pool:
        parts = [dict(np.load(p)) for p in pool.map(run_piece, jobs)]
    merged = {k: np.concatenate([c[k] for c in parts]) for k in KEYS}
    cap = cv2.VideoCapture(str(videos[0]))
    merged["size"] = np.array([cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)], int)
    np.savez_compressed(target, **merged)
    print(f"{len(merged['t_ms'])} frames, face found in {merged['ok'].mean():.1%} -> {target}")
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--name", default="")
    ap.add_argument("--landmarker", default=str(Path(__file__).resolve().parent / "face_landmarker.task"))
    ap.add_argument("--workers", type=int, default=mproc.cpu_count())
    a = ap.parse_args()
    videos = [Path(v).resolve() for v in a.videos]
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    extract(videos, Path(a.landmarker), out, a.name or videos[0].stem, a.workers)


if __name__ == "__main__":
    main()
