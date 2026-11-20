import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from aoi import coordinates, number, read_csv

CELL_PX = 4


def draw(path: Path, xy: np.ndarray, weight: np.ndarray, size: tuple[float, float], sigma_px: float,
         background: Path | None = None, scanpath: bool = True, areas: list[dict] | None = None) -> None:
    w, h = size
    ok = np.isfinite(xy).all(1) & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    grid = np.zeros((int(np.ceil(h / CELL_PX)), int(np.ceil(w / CELL_PX))))
    np.add.at(grid, ((xy[ok, 1] // CELL_PX).astype(int), (xy[ok, 0] // CELL_PX).astype(int)), weight[ok])
    grid = gaussian_filter(grid, sigma_px / CELL_PX)
    grid = grid / grid.max() if grid.max() > 0 else grid
    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    if background:
        ax.imshow(plt.imread(str(background)), extent=(0, w, h, 0))
    else:
        ax.imshow(np.full((2, 2, 3), 0.95), extent=(0, w, h, 0))
    rgba = plt.get_cmap("turbo")(0.35 + 0.65 * grid)
    rgba[..., 3] = np.clip(1.6 * grid, 0, 0.75)
    ax.imshow(rgba, extent=(0, w, h, 0), interpolation="bilinear")
    for a in areas or []:
        if "x" in a:
            ax.add_patch(plt.Rectangle((a["x"], a["y"]), a["w"], a["h"], fill=False, ec="#2a6fdb", lw=1.5))
            ax.text(a["x"] + 4, a["y"] + 14, a["name"], color="#2a6fdb", fontsize=9)
    if scanpath and ok.sum():
        p = xy[ok]
        ax.plot(p[:, 0], p[:, 1], c="#1f4fa8", lw=1.0, alpha=0.7)
        ax.scatter(p[:, 0], p[:, 1], s=60 + weight[ok] / 4.0, facecolor="white", edgecolor="#1f4fa8", lw=1.2, alpha=0.9, zorder=3)
        for j, (x, y) in enumerate(p[:99]):
            ax.text(x, y, str(j + 1), color="#1f4fa8", fontsize=7, ha="center", va="center", zorder=4)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--space", default="screen")
    ap.add_argument("--background")
    ap.add_argument("--size", nargs=2, type=float)
    ap.add_argument("--sigma-px", type=float, default=40.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    folder = Path(a.folder)
    fixes = [r for r in read_csv(folder / "fixations.csv") if r["kind"] == "fixation"]
    xy = coordinates(fixes, a.space)
    size = a.size or (plt.imread(a.background).shape[1::-1] if a.background else (np.nanmax(xy[:, 0]) + 50, np.nanmax(xy[:, 1]) + 50))
    out = Path(a.out) if a.out else folder / f"heatmap_{a.space}.png"
    draw(out, xy, np.array([number(f["duration_ms"]) for f in fixes]), tuple(size), a.sigma_px, Path(a.background) if a.background else None)
    print(out)


if __name__ == "__main__":
    main()
