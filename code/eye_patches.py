import argparse
import multiprocessing as mproc
from pathlib import Path

import cv2
import numpy as np

PATCH_W, PATCH_H = 64, 32
CORNER_X = (10.0, 54.0)
CORNERS = {"left": (33, 133), "right": (362, 263)}


def warp(gray: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    src = np.float32([a, b, a + np.array([-(b - a)[1], (b - a)[0]])])
    d = CORNER_X[1] - CORNER_X[0]
    dst = np.float32([[CORNER_X[0], PATCH_H / 2], [CORNER_X[1], PATCH_H / 2], [CORNER_X[0], PATCH_H / 2 + d]])
    return cv2.warpAffine(gray, cv2.getAffineTransform(src, dst), (PATCH_W, PATCH_H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def run_piece(job: tuple) -> Path:
    video, lm, ckpt, idx = job
    final = ckpt / f"patches{idx:02d}.npy"
    if final.exists():
        return final
    cap = cv2.VideoCapture(str(video))
    out = np.zeros((len(lm), 2, PATCH_H, PATCH_W), np.uint8)
    for i in range(len(lm)):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for k, e in enumerate(CORNERS):
            a, b = lm[i, CORNERS[e][0], :2], lm[i, CORNERS[e][1], :2]
            if e == "right":
                a, b = b, a
            out[i, k] = warp(gray, a, b)
    cap.release()
    np.save(final, out)
    return final


def frame_counts(videos: list[Path]) -> list[int]:
    n = []
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        c = 0
        while cap.grab():
            c += 1
        n.append(c)
    return n


def eye_patches(videos: list[Path], features: Path, out: Path, name: str, workers: int = mproc.cpu_count()) -> Path:
    target = out / f"{name}_patches.npy"
    if target.exists():
        return target
    ckpt = out / f"{name}_checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    lm = np.load(features)["lm"]
    counts = frame_counts(videos)
    starts = np.r_[0, np.cumsum(counts)]
    jobs = [(Path(v), lm[starts[i]:starts[i + 1]], ckpt, i) for i, v in enumerate(videos)]
    with mproc.Pool(min(workers, len(jobs))) as pool:
        parts = [np.load(p) for p in pool.map(run_piece, jobs)]
    np.save(target, np.concatenate(parts))
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="cache")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    eye_patches([Path(v).resolve() for v in a.videos], Path(a.features), Path(a.out), a.name)


if __name__ == "__main__":
    main()
