import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.spatial.transform import Rotation

MIRROR = np.diag([-1.0, 1.0, 1.0])


def to_person(v: np.ndarray) -> np.ndarray:
    return v @ MIRROR


def stereo(g: np.ndarray) -> np.ndarray:
    g = to_person(g / np.linalg.norm(g, axis=-1, keepdims=True))
    with np.errstate(invalid="ignore"):
        return (g[..., 0] + 1j * g[..., 1]) / (1.0 + g[..., 2])


def inverse_stereo(z: np.ndarray) -> np.ndarray:
    r2 = np.abs(z) ** 2
    g = np.stack([2 * z.real, 2 * z.imag, 1.0 - r2], -1) / (1.0 + r2)[..., None]
    return to_person(g)


def eccentricity_deg(z: np.ndarray) -> np.ndarray:
    return np.degrees(2.0 * np.arctan(np.abs(z)))


def cayley_klein(rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z, w = Rotation.from_matrix(MIRROR @ rot @ MIRROR).as_quat().T
    return w + 1j * z, y - 1j * x


def mobius(a: np.ndarray, b: np.ndarray, z: np.ndarray) -> np.ndarray:
    return (a * z + b) / (-np.conj(b) * z + np.conj(a))


def decompose(pose: np.ndarray, gaze_cam: np.ndarray) -> dict:
    rot = pose[:, :3, :3].astype(float)
    eye = np.einsum("nji,nj->ni", rot, gaze_cam)
    a, b = cayley_klein(rot)
    z_eye = stereo(eye)
    return {"gaze": stereo(gaze_cam), "head": mobius(a, b, np.zeros(len(rot))), "eye": z_eye, "composed": mobius(a, b, z_eye)}


def screen_offset(z: np.ndarray, distance: np.ndarray) -> np.ndarray:
    return 2.0 * distance * z / (1.0 - np.abs(z) ** 2)


def draw_plane(ax, z_gaze: np.ndarray, z_head: np.ndarray | None = None, z_target: np.ndarray | None = None,
               fixations: np.ndarray | None = None, rings_deg=(10, 20, 30)) -> None:
    th = np.linspace(0, 2 * np.pi, 400)
    for e in rings_deg:
        r = np.tan(np.radians(e) / 2)
        ax.plot(r * np.cos(th), r * np.sin(th), c="#bbbbbb", lw=0.6, ls=":")
        ax.text(r * np.cos(np.radians(52)), r * np.sin(np.radians(52)), f"{e}°", fontsize=7, color="#555555")
    lim = np.tan(np.radians(max(rings_deg)) / 2) * 1.08
    ax.plot([-lim, lim], [0, 0], c="#bbbbbb", lw=0.5)
    ax.plot([0, 0], [-lim, lim], c="#bbbbbb", lw=0.5)
    if z_head is not None:
        ax.plot(z_head.real, z_head.imag, c="#9a9a9a", lw=1.0, ls="--", label="tête (normale du visage)")
    ax.plot(z_gaze.real, z_gaze.imag, c="#b5b5b5", lw=0.5, label="regard, image par image")
    if z_target is not None:
        ax.scatter(z_target.real, z_target.imag, marker="+", s=40, c="#000000", lw=0.9, label="cibles", zorder=4)
    if fixations is not None:
        ax.scatter(fixations.real, fixations.imag, s=14, facecolors="#000000", edgecolors="#000000", label="fixations", zorder=5)
        k = int(np.argmax(np.abs(fixations)))
        ax.annotate("", xy=(fixations[k].real, fixations[k].imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#000000", lw=1.2, shrinkA=0, shrinkB=2))
    ax.plot(0, 0, marker="o", ms=7, mfc="#ffffff", mec="#000000", mew=1.2, zorder=6)
    ax.annotate("personne", (0, 0), xytext=(-8, 9), textcoords="offset points", ha="right", fontsize=8, bbox=dict(boxstyle="square,pad=0.15", fc="#ffffff", ec="none"), zorder=7)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}\,z$ (vers la droite)")
    ax.set_ylabel(r"$\mathrm{Im}\,z$ (vers le haut)")


def draw_trajectory(ax, t: np.ndarray, z: np.ndarray, z_target: np.ndarray | None = None, z_head: np.ndarray | None = None,
                    stem_every_s: float = 0.4) -> None:
    ax.plot(t, np.zeros_like(t), np.zeros_like(t), c="#000000", lw=0.8)
    for s in np.arange(t[0], t[-1], stem_every_s):
        i = int(np.argmin(np.abs(t - s)))
        if np.isfinite(z[i]):
            ax.plot([t[i], t[i]], [0, z[i].real], [0, z[i].imag], c="#9a9a9a", lw=0.5)
    if z_head is not None:
        ax.plot(t, z_head.real, z_head.imag, c="#9a9a9a", lw=0.9, ls="--", label="tête")
    if z_target is not None:
        ax.plot(t, z_target.real, z_target.imag, c="#000000", lw=0.7, ls=":", label="cible")
    ax.plot(t, z.real, z.imag, c="#000000", lw=1.3, label="regard")
    ax.set_xlabel("temps (s)", labelpad=10)
    ax.set_ylabel(r"$\mathrm{Re}\,z$", labelpad=8)
    ax.set_zlabel(r"$\mathrm{Im}\,z$", labelpad=8)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.zaxis.set_major_locator(MaxNLocator(3))
    ax.view_init(elev=18, azim=-62)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#ffffff")
        axis.pane.set_edgecolor("#000000")
        axis._axinfo["grid"]["color"] = "#e0e0e0"
