import math

import numpy as np

BLINK_BLENDSHAPES = (9, 10)
MAX_GAP_S = 0.1


def one_euro(t: np.ndarray, x: np.ndarray, valid: np.ndarray, min_cutoff: float = 1.0, beta: float = 0.3,
             d_cutoff: float = 1.0) -> np.ndarray:
    alpha = lambda c, dt: 1.0 / (1.0 + 1.0 / (2 * math.pi * c * dt))
    y = np.full_like(x, np.nan)
    state = None
    for i in range(len(t)):
        if not valid[i]:
            state = None
            continue
        if state is None or not 0 < t[i] - state[0] <= MAX_GAP_S:
            state = (t[i], x[i], np.zeros(x.shape[1]))
            y[i] = x[i]
            continue
        dt = t[i] - state[0]
        dx = state[2] + alpha(d_cutoff, dt) * ((x[i] - state[1]) / dt - state[2])
        xh = state[1] + alpha(min_cutoff + beta * np.linalg.norm(dx), dt) * (x[i] - state[1])
        state = (t[i], xh, dx)
        y[i] = xh
    return y


def blinks(blend: np.ndarray, ok: np.ndarray, fps: float, threshold: float = 0.5, duration_ms: float = 150.0) -> np.ndarray:
    cand = ok & (np.min(blend[:, BLINK_BLENDSHAPES], 1) >= threshold)
    half = int(round(duration_ms / 1000 * fps / 2))
    return np.convolve(cand.astype(float), np.ones(2 * half + 1), "same") > 0


def smooth_step(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    a = np.where(s > 0.0, np.exp(-1.0 / np.maximum(s, 1e-12)), 0.0)
    b = np.where(s < 1.0, np.exp(-1.0 / np.maximum(1.0 - s, 1e-12)), 0.0)
    return a / (a + b)


def noise_sd(blocks: list[np.ndarray]) -> np.ndarray:
    d2 = [y[2:] - 2.0 * y[1:-1] + y[:-2] for y in blocks if len(y) >= 3]
    if not d2:
        return np.ones(2)
    return np.maximum(1.4826 * np.median(np.abs(np.concatenate(d2)), axis=0) / math.sqrt(6.0), 1e-3)


def segment(t: np.ndarray, z: np.ndarray, penalty: float, slope_penalty: float, min_len: int) -> list[tuple[int, int, int]]:
    n = len(z)
    if n < 2 * min_len:
        return [(0, n, 0)]
    tc = t - t[0]
    c = lambda x: np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(x, axis=0)])
    s1, s_t, s_tt = c(np.ones(n)), c(tc), c(tc * tc)
    s_y, s_yy, s_ty = c(z), c(z * z), c(tc[:, None] * z)
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    arg = np.zeros(n + 1, dtype=int)
    deg = np.zeros(n + 1, dtype=int)
    for j in range(min_len, n + 1):
        i = np.arange(0, j - min_len + 1)
        cnt = s1[j] - s1[i]
        sy, syy = s_y[j] - s_y[i], s_yy[j] - s_yy[i]
        cost0 = (syy - sy * sy / cnt[:, None]).sum(axis=1)
        st, stt, sty = s_t[j] - s_t[i], s_tt[j] - s_tt[i], s_ty[j] - s_ty[i]
        det = cnt * stt - st * st
        ok = det > 1e-12
        safe = np.where(ok, det, 1.0)[:, None]
        explained = (stt[:, None] * sy * sy - 2.0 * st[:, None] * sy * sty + cnt[:, None] * sty * sty) / safe
        cost1 = np.where(ok, (syy - explained).sum(axis=1) + slope_penalty, np.inf)
        total = best[i] + np.maximum(np.minimum(cost0, cost1), 0.0) + penalty
        k = int(np.argmin(total))
        best[j], arg[j], deg[j] = total[k], int(i[k]), int(cost1[k] < cost0[k])
    out, j = [], n
    while j > 0:
        out.append((int(arg[j]), j, int(deg[j])))
        j = int(arg[j])
    return out[::-1]


def staircase(t: np.ndarray, gaze: np.ndarray, valid: np.ndarray, fps: float, penalty_factor: float = 3.0,
              min_fixation_s: float = 0.1, noise_floor_deg: float = 0.5, pursuit_factor: float = 2.0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    idx = np.where(valid)[0]
    cuts = np.where(np.diff(t[idx]) > MAX_GAP_S)[0] + 1
    blocks = [b for b in np.split(idx, cuts) if len(b)]
    sigma = np.maximum(noise_sd([gaze[b] for b in blocks]), noise_floor_deg)
    penalty = penalty_factor * math.log(max(len(idx), 2))
    min_len = max(1, int(round(min_fixation_s * fps)))
    stair = np.full_like(gaze, np.nan)
    label = np.full(len(t), -1)
    events = []
    for b in blocks:
        tb, yb = t[b], gaze[b]
        models = []
        for a, e, d in segment(tb, yb / sigma, penalty, pursuit_factor * penalty, min_len):
            coef = np.polyfit(tb[a:e] - tb[a], yb[a:e], 1) if d == 1 else np.vstack([np.zeros(2), yb[a:e].mean(axis=0)])
            models.append((a, e, d, coef))
        value = lambda k, tt: np.outer(tt - tb[models[k][0]], models[k][3][0]) + models[k][3][1]
        plateau = np.vstack([value(k, tb[a:e]) for k, (a, e, _, _) in enumerate(models)])
        out = plateau.copy()
        for k in range(1, len(models)):
            a = models[k][0]
            tau = 0.5 * (tb[a - 1] + tb[a])
            amplitude = float(np.linalg.norm(value(k, np.array([tau]))[0] - value(k - 1, np.array([tau]))[0]))
            duration = 0.021 + 0.0022 * amplitude
            win = np.where(np.abs(tb - tau) < duration / 2.0)[0]
            if len(win):
                s = smooth_step((tb[win] - (tau - duration / 2.0)) / duration)[:, None]
                out[win] = value(k - 1, tb[win]) * (1.0 - s) + value(k, tb[win]) * s
        for k, (a, e, d, _) in enumerate(models):
            label[b[a:e]] = len(events)
            events.append({"id": len(events), "kind": "pursuit" if d else "fixation",
                           "start_s": round(float(tb[a]), 4), "end_s": round(float(tb[e - 1]), 4),
                           "duration_ms": round(float(tb[e - 1] - tb[a]) * 1000.0 + 1000.0 / fps, 1),
                           "yaw_deg": round(float(plateau[a:e, 0].mean()), 3), "pitch_deg": round(float(plateau[a:e, 1].mean()), 3),
                           "saccade_deg": round(float(np.linalg.norm(plateau[a] - plateau[a - 1])), 3) if k else None})
        stair[b] = out
    return stair, label, events
