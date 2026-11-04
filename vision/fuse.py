"""The detector's ball in pitch metres, and combined with the tracking's ball.

The detector (vision/detect_spans.py) gives up to 10 pixel candidates per frame on the frames it
ran on. Per frame:

  1. candidates that sit still -- within STILL_PX of a candidate STILL_FRAMES frames away on two
     sides (before and after, or twice before, or twice after) -- are dropped: in live play a
     ball does not stay on the same pixel for most of a second, while fixed marks on the image (a
     scoreboard digit, a painted spot) do. This needs no knowledge of where such marks are in our
     video, and works inside short detected spans.
  2. the most confident remaining candidate is taken back to the pitch through the frame's
     inverse homography.

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


def detector_ball(folder, n, conf):
    """(n, 2) the detector's ball in metres where confident (>= conf), NaN elsewhere."""
    top = best(drop_still(load_detections(folder, n)))
    ball = to_pitch(top[:, 0], top[:, 1])
    ball[~(np.nan_to_num(top[:, 2]) >= conf)] = np.nan
    return ball


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
