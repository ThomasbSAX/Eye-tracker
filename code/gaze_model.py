import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

import dataset as ds

LIMBUS_RADIUS_MM = 5.9
LIMBUS_DEPTH_RATIO = 1.763
MAX_SIN_GAZE = 0.97
EMB_COMPONENTS = 16
PATCH_COMPONENTS = 24
RESIDUAL_ALPHA = 0.1
LINEAR_RIDGE = 0.1
EYES = ("left", "right")
IRIS = {"left": list(range(468, 473)), "right": list(range(473, 478))}
CANTHI = {"left": (133, 33, 159, 145), "right": (362, 263, 386, 374)}
CORNERS = {"left": (33, 133), "right": (362, 263)}
CONTOUR = {
    "left": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "right": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
}
EYE_BLENDSHAPES = slice(9, 23)
NAMES = ["rest_yaw_left", "rest_pitch_left", "rest_yaw_right", "rest_pitch_right", "gain_x", "gain_y",
         "eyeball_depth_mm", "lid_gain"]
P = {n: i for i, n in enumerate(NAMES)}


def rotation_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    axis = np.cross(u, v)
    s, c = float(np.linalg.norm(axis)), float(np.clip(u @ v, -1, 1))
    if s < 1e-12:
        return np.eye(3)
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def to_head(pose: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("nji,nj->ni", pose[:, :3, :3].astype(float), v)


def to_camera(pose: np.ndarray, v: np.ndarray) -> np.ndarray:
    return np.einsum("nij,nj->ni", pose[:, :3, :3].astype(float), v)


def eye_frame(lm: np.ndarray, eye: str) -> np.ndarray:
    a, b = lm[:, CORNERS[eye][0], :2], lm[:, CORNERS[eye][1], :2]
    center = (a + b) / 2.0
    d = b - a
    span = np.linalg.norm(d, axis=-1, keepdims=True)
    ex = d / span
    ey = np.stack([-ex[:, 1], ex[:, 0]], -1)
    pts = lm[:, IRIS[eye] + CONTOUR[eye], :2] - center[:, None]
    return np.stack([np.einsum("npk,nk->np", pts, ex), np.einsum("npk,nk->np", pts, ey)], -1) / span[:, None]


def measurements(data: dict) -> dict:
    lm, pose = data["lm"], data["pose"]
    dist = np.maximum(np.abs(pose[:, 2, 3]), 1e-3)
    rot = pose[:, :3, :3].astype(float)
    m = {"rot": rot, "behind": np.stack([-rot[:, 0, 2], rot[:, 1, 2]], -1)}
    for e in EYES:
        c = lm[:, IRIS[e][0], :2].astype(float)
        rim = lm[:, IRIS[e][1:], :2] - c[:, None]
        r = np.maximum(np.abs(rim[..., 0]).max(1), np.abs(rim[..., 1]).max(1))
        inn, out, up, lo = CANTHI[e]
        width = np.linalg.norm(lm[:, out, :2] - lm[:, inn, :2], axis=-1)
        m[e] = {"iris": c, "canthi": ((lm[:, inn, :2] + lm[:, out, :2]) / 2).astype(float), "r_dist": r * dist,
                "dist": dist, "open": np.linalg.norm(lm[:, lo, :2] - lm[:, up, :2], axis=-1) / np.maximum(width, 1e-6)}
    return m


def patch_matrix(data: dict, idx: np.ndarray) -> np.ndarray:
    p = data["patches"][idx].astype(np.float32).reshape(len(idx), 2, -1)
    p = (p - p.mean(-1, keepdims=True)) / (p.std(-1, keepdims=True) + 1.0)
    return p.reshape(len(idx), -1)


def network_angles(data: dict, idx: np.ndarray) -> dict:
    out = {}
    for k in ("mpii", "eth"):
        g = np.nan_to_num(data[f"app_{k}"][idx].astype(float))
        g[np.linalg.norm(g, axis=1) < 0.5] = [0.0, 0.0, 1.0]
        g = ds.unit(g)
        out[k] = ds.fick(g)
        out[k + "_head"] = ds.fick(to_head(data["pose"][idx], g))
    return out


class GazeModel:
    def __init__(self):
        self.params = None
        self.k_radius = None
        self.open_median = None
        self.linear = None
        self.pca = None
        self.patch_pca = None
        self.scaler = None
        self.residual = None
        self.recal = np.zeros((3, 2))
        self.focal_px = None
        self.camera_mm = None
        self.screen = None

    def setup(self) -> ds.Setup:
        return ds.Setup(focal_px=self.focal_px or 1000.0, camera_x_mm=self.camera_mm[0], camera_y_mm=self.camera_mm[1])

    def adapt(self, data: dict) -> None:
        m = measurements(data)
        ok = data["ok"]
        self.k_radius = {e: float(np.median(m[e]["r_dist"][ok])) for e in EYES}
        self.open_median = {e: float(np.median(m[e]["open"][ok])) for e in EYES}

    def _physical(self, m: dict, idx: np.ndarray, params: np.ndarray) -> dict:
        out = {}
        for k, e in enumerate(EYES):
            d = m[e]
            a = self.k_radius[e] / d["dist"][idx]
            q = d["canthi"][idx] + (params[P["eyeball_depth_mm"]] * a / LIMBUS_RADIUS_MM)[:, None] * m["behind"][idx]
            delta = (d["iris"][idx] - q) / (LIMBUS_DEPTH_RATIO * a[:, None])
            delta[:, 1] -= params[P["lid_gain"]] * (d["open"][idx] - self.open_median[e])
            n = np.linalg.norm(delta, axis=-1, keepdims=True)
            delta = np.where(n > MAX_SIN_GAZE, delta * MAX_SIN_GAZE / np.maximum(n, 1e-9), delta)
            g_cam = np.stack([delta[:, 0], -delta[:, 1], np.sqrt(1 - np.sum(delta ** 2, -1))], -1)
            g_head = np.einsum("nji,nj->ni", m["rot"][idx], g_cam)
            rc = rotation_between(ds.from_fick(params[2 * k: 2 * k + 2]), np.array([0.0, 0.0, 1.0]))
            out[e] = ds.from_fick(ds.fick(g_head @ rc.T) * params[[P["gain_x"], P["gain_y"]]])
        return out

    def _true_head(self, data: dict, m: dict, idx: np.ndarray, eye: str) -> np.ndarray:
        target = ds.screen_points_mm(data["target_mm"][idx], self.setup())
        return np.einsum("nji,nj->ni", m["rot"][idx], ds.unit(target - data["eye_pos"][eye][idx]))

    def fit_geometry(self, data: dict, idx: np.ndarray, camera_mm: tuple[float, float]) -> None:
        self.adapt(data)
        self.camera_mm = tuple(camera_mm)
        m = measurements(data)
        truth = {e: ds.fick(self._true_head(data, m, idx, e)) for e in EYES}
        p = np.array([0, 0, 0, 0, 1.0, 1.0, 10.0, 0.0])
        for k, e in enumerate(EYES):
            p[2 * k: 2 * k + 2] = np.median(ds.fick(self._physical(m, idx, p)[e]), 0)

        def res(x, w):
            g = self._physical(m, idx, x)
            r = np.concatenate([(ds.fick(g[e]) - truth[e]).ravel() for e in EYES])
            return r * w if w is not None else r

        lo = np.array([-45, -45, -45, -45, 0.5, 0.5, 0, -3])
        hi = np.array([45, 45, 45, 45, 3, 4, 20, 3])
        w = None
        for _ in range(4):
            p = least_squares(res, np.clip(p, lo + 1e-6, hi - 1e-6), bounds=(lo, hi), args=(w,), x_scale="jac").x
            w = (1 + (res(p, None) / 2.0) ** 2) ** -0.25
        self.params = p

    def physical_angles(self, data: dict, idx: np.ndarray) -> np.ndarray:
        g = self._physical(measurements(data), idx, self.params)
        return ds.fick(to_camera(data["pose"][idx], ds.unit(g["left"] + g["right"])))

    def target_angles(self, data: dict, idx: np.ndarray) -> np.ndarray:
        target = ds.screen_points_mm(data["target_mm"][idx], self.setup())
        return ds.fick(ds.unit(target - data["cyclops"][idx]))

    def _linear_inputs(self, data: dict, idx: np.ndarray) -> np.ndarray:
        return np.c_[network_angles(data, idx)["mpii"], np.ones(len(idx))]

    def _residual_inputs(self, data: dict, idx: np.ndarray) -> np.ndarray:
        a = network_angles(data, idx)
        lm = data["lm"][idx]
        eyes = [eye_frame(lm, e).reshape(len(idx), -1) for e in EYES]
        return np.nan_to_num(np.c_[eyes[0], eyes[1], data["blend"][idx][:, EYE_BLENDSHAPES], a["mpii_head"], a["eth_head"],
                                   self.pca.transform(data["app_emb"][idx].astype(np.float32)),
                                   self.patch_pca.transform(patch_matrix(data, idx))])

    def fit(self, data: dict, idx: np.ndarray, camera_mm: tuple[float, float]) -> "GazeModel":
        self.fit_geometry(data, idx, camera_mm)
        T = self.target_angles(data, idx)
        A = self._linear_inputs(data, idx)
        reg = LINEAR_RIDGE * len(A) / 100.0 * np.diag([1.0, 1.0, 0.0])
        self.linear = np.linalg.solve(A.T @ A + reg, A.T @ T)
        self.pca = PCA(EMB_COMPONENTS).fit(data["app_emb"][idx].astype(np.float32))
        self.patch_pca = PCA(PATCH_COMPONENTS).fit(patch_matrix(data, idx))
        X = self._residual_inputs(data, idx)
        self.scaler = StandardScaler().fit(X)
        self.residual = KernelRidge(alpha=RESIDUAL_ALPHA, kernel="rbf", gamma=0.5 / X.shape[1]).fit(
            self.scaler.transform(X), T - A @ self.linear)
        self.recal = np.zeros((3, 2))
        return self

    def angles(self, data: dict, idx: np.ndarray, stage: str = "full") -> np.ndarray:
        if stage == "physical":
            a = self.physical_angles(data, idx)
        else:
            a = self._linear_inputs(data, idx) @ self.linear
            if stage == "full":
                a = a + self.residual.predict(self.scaler.transform(self._residual_inputs(data, idx)))
        return a + np.c_[a, np.ones(len(a))] @ self.recal

    def quick_recalibrate(self, data: dict, idx: np.ndarray, stage: str = "full", ridge: float = 0.05) -> None:
        self.recal = np.zeros((3, 2))
        pred = self.angles(data, idx, stage)
        A = np.c_[pred, np.ones(len(pred))]
        self.recal = np.linalg.solve(A.T @ A + np.diag([ridge, ridge, 0.0]) * len(A), A.T @ (self.target_angles(data, idx) - pred))

    def camera_gaze(self, data: dict, idx: np.ndarray, stage: str = "full") -> np.ndarray:
        return ds.from_fick(self.angles(data, idx, stage))

    def save(self, path: Path) -> None:
        Path(path).write_bytes(pickle.dumps(self.__dict__))

    @classmethod
    def load(cls, path: Path) -> "GazeModel":
        m = cls()
        m.__dict__.update(pickle.loads(Path(path).read_bytes()))
        return m
