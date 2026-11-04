"""The detector's ball in pitch metres, and combined with the tracking's ball.

The detector (vision/detect_spans.py) gives up to 10 pixel candidates per frame on the frames it
ran on. Per frame:

  1. candidates that sit still -- within STILL_PX of a candidate STILL_FRAMES frames away on two
     sides (before and after, or twice before, or twice after) -- are dropped: in live play a
     ball does not stay on the same pixel for most of a second, while fixed marks on the image (a
     scoreboard digit, a painted spot) do. This needs no knowledge of where such marks are in our
     video, and works inside short detected spans.
  2. all remaining candidates are taken back to the pitch through the frame's inverse homography,
     and one per frame is chosen by `follow`: the most confident sequence of candidates that
     moves like a ball. Physics only -- consecutive picks may not be further apart than a ball
     at MAX_SPEED can travel -- and no use of the tracking, so the detector can still correct it.
     Picking the single most confident candidate per frame instead let the detector jump to
     other objects (a sock, a spare ball) on many frames.

`fused_ball` then uses it only where it is confident (conf.json, chosen on outside test tiles):
where the tracking has no ball, it fills it; where the two are more than DISAGREE m apart, the
detector wins (the tracking's ball is the one that jumps; 3 m is well beyond either's noise).
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, "starter")]
from games import GAME, GAMES  # noqa: E402
from load_homography import load as load_homography  # noqa: E402

STILL_PX = 3.0
STILL_FRAMES = 10
DISAGREE = 3.0
MAX_SPEED = 45.0    # m/s; faster than any kick (the same limit as possession/rules.py)
FPS = 25
BRIDGE = 3          # frames; a brief occlusion the ball can be followed through


def load_detections(folder, n):
    """(n, 10, 3) candidates (x px, y px, conf) for every frame; NaN where not run or none."""
    dets = np.full((n, 10, 3), np.nan, dtype=np.float32)
    for path in sorted(glob.glob(os.path.join(folder, "det_*.npz"))):
        data = np.load(path)
        dets[data["frames"]] = data["dets"]
    return dets


def drop_still(dets):
    """Candidates that sit still (see the module docstring): NaN."""
    out = dets.copy()
    xy = dets[..., :2]
    n, k = len(dets), STILL_FRAMES

    def near(shift):
        """(n, 10): a candidate within STILL_PX `shift` frames away (False off the ends)."""
        other = np.full_like(xy, np.nan)
        if shift > 0:
            other[:-shift] = xy[shift:]
        else:
            other[-shift:] = xy[:n + shift]
        d = np.hypot(xy[:, :, None, 0] - other[:, None, :, 0], xy[:, :, None, 1] - other[:, None, :, 1])
        return (np.nan_to_num(d, nan=np.inf) < STILL_PX).any(2)

    before, after = near(-k), near(k)
    still = (before & after) | (before & near(-2 * k)) | (after & near(2 * k))
    out[still] = np.nan
    return out


def best(dets):
    """(n, 3) the most confident candidate per frame; NaN where none."""
    pick = np.argmax(np.nan_to_num(dets[..., 2], nan=-1.0), axis=1)
    return dets[np.arange(len(dets)), pick]


def to_pitch(px, py):
    """Pixels -> pitch metres through each frame's inverse homography. NaN in, NaN out."""
    matrices, _, valid = load_homography()
    inv = np.full_like(matrices, np.nan)
    inv[valid] = np.linalg.inv(matrices[valid])
    den = inv[:, 2, 0] * px + inv[:, 2, 1] * py + inv[:, 2, 2]
    x = (inv[:, 0, 0] * px + inv[:, 0, 1] * py + inv[:, 0, 2]) / den
    y = (inv[:, 1, 0] * px + inv[:, 1, 1] * py + inv[:, 1, 2]) / den
    length, width = GAMES[GAME]["pitch"]
    return np.stack([x * length, y * width], axis=1)


def candidates_to_pitch(dets):
    """(n, k, 2) every candidate in pitch metres. NaN in, NaN out."""
    matrices, _, valid = load_homography()
    inv = np.full_like(matrices, np.nan)
    inv[valid] = np.linalg.inv(matrices[valid])
    inv = inv[:, None]
    px, py = dets[..., 0], dets[..., 1]
    den = inv[..., 2, 0] * px + inv[..., 2, 1] * py + inv[..., 2, 2]
    x = (inv[..., 0, 0] * px + inv[..., 0, 1] * py + inv[..., 0, 2]) / den
    y = (inv[..., 1, 0] * px + inv[..., 1, 1] * py + inv[..., 1, 2]) / den
    length, width = GAMES[GAME]["pitch"]
    return np.stack([x * length, y * width], axis=-1)


def follow(xy, conf, threshold):
    """Index of the chosen candidate per frame, or -1 for none. Viterbi over the candidates.

    Cost of a frame: -log(conf) for a candidate, -log(threshold) for none (so on its own a
    candidate below the threshold never beats none). A candidate may follow one of up to BRIDGE
    frames earlier (the frames between are none) if a ball at MAX_SPEED could cover the distance
    in that time, and costs nothing extra; picking up a candidate otherwise -- after a longer
    gap, at the start, or another object -- costs two frames of none. So switching to another
    object must be paid for, while following the ball, through brief occlusions, is not.
    """
    n, k = conf.shape
    none_cost = -np.log(threshold)
    start = 2 * none_cost
    step = MAX_SPEED / FPS
    emit = np.where(np.isnan(conf), np.inf, -np.log(np.clip(np.nan_to_num(conf), 1e-6, 1)))
    costs = np.full((n, k + 1), np.inf)            # best path cost ending in each state
    back_t = np.zeros((n, k + 1), dtype=np.int32)  # predecessor frame and state
    back_s = np.zeros((n, k + 1), dtype=np.int16)
    costs[0, :k] = emit[0] + start
    costs[0, k] = none_cost
    back_t[0] = -1
    for t in range(1, n):
        prev = int(np.argmin(costs[t - 1]))
        best = np.full(k, costs[t - 1, prev] + start)          # pick up a new candidate
        bt = np.full(k, t - 1)
        bs = np.full(k, prev)
        for g in range(1, min(BRIDGE, t) + 1):
            d = np.hypot(xy[t][:, None, 0] - xy[t - g][None, :, 0],
                         xy[t][:, None, 1] - xy[t - g][None, :, 1])
            ok = np.nan_to_num(d, nan=np.inf) <= step * g
            via = np.where(ok, costs[t - g, None, :k] + (g - 1) * none_cost, np.inf)
            j = np.argmin(via, axis=1)
            c = via[np.arange(k), j]
            better = c < best
            best[better], bt[better], bs[better] = c[better], t - g, j[better]
        costs[t, :k] = best + emit[t]
        back_t[t, :k], back_s[t, :k] = bt, bs
        costs[t, k] = costs[t - 1, prev] + none_cost
        back_t[t, k], back_s[t, k] = t - 1, prev
    out = np.full(n, -1, dtype=np.int16)
    t, s = n - 1, int(np.argmin(costs[n - 1]))
    while t >= 0:
        if s < k:
            out[t] = s
        t, s = back_t[t, s], back_s[t, s]
    return out


def detector_ball(folder, n, conf):
    """(n, 2) the detector's ball in metres where confident (>= conf), NaN elsewhere."""
    dets = drop_still(load_detections(folder, n))
    xy = candidates_to_pitch(dets)
    ball = np.full((n, 2), np.nan)
    ran = ~np.isnan(dets[:, :, 2]).all(1)
    for a, b in _runs(ran):
        pick = follow(xy[a:b], dets[a:b, :, 2], conf)
        rows = np.flatnonzero(pick >= 0)
        ball[a + rows] = xy[a + rows, pick[rows]]
    return ball


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def read_conf(path):
    with open(path) as handle:
        return float(json.load(handle)["conf"])


def fused_ball(track, det):
    """The tracking's ball, filled or corrected by the detector's (both (n, 2) metres)."""
    out = track.copy()
    has_det = ~np.isnan(det[:, 0])
    far = np.nan_to_num(np.hypot(*(det - track).T), nan=np.inf) > DISAGREE
    use = has_det & (np.isnan(track[:, 0]) | far)
    out[use] = det[use]
    return out
