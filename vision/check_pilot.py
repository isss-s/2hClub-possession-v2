"""Stop-or-go check on the pilot detections, before spending units on the whole match.

    python vision/check_pilot.py external/det_pilot external/conf.json

Automatic, from the provided data only (no labels of any kind): on the pilot frames, how often
the detector finds a confident ball, and how often it agrees with the tracking's ball where the
tracking has one. The pass marks were set before any detection was seen:

    coverage   a confident ball on at least 50% of pilot frames
    agreement  where both have a ball, at least 75% within 2 m of each other
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "possession")]
import fuse  # noqa: E402
from match import read_tracking  # noqa: E402

MIN_COVERAGE = 0.50
MIN_AGREEMENT = 0.75
AGREE_M = 2.0


def main(folder, conf_path):
    track, _, _ = read_tracking()
    n = len(track)
    frames = np.zeros(n, dtype=bool)
    for path in glob.glob(os.path.join(folder, "det_*.npz")):
        frames[np.load(path)["frames"]] = True
    conf = fuse.read_conf(conf_path)
    det = fuse.detector_ball(folder, n, conf)
    found = frames & ~np.isnan(det[:, 0])
    both = found & ~np.isnan(track[:, 0])
    dist = np.hypot(*(det[both] - track[both]).T)
    coverage = found.sum() / frames.sum()
    agreement = float(np.mean(dist < AGREE_M)) if len(dist) else 0.0
    print("pilot frames {}, confidence threshold {:.2f} (from outside test tiles)".format(
        frames.sum(), conf))
    print("coverage   {:.1%} of frames have a confident ball   (need {:.0%})".format(
        coverage, MIN_COVERAGE))
    print("agreement  {:.1%} within {:.0f} m of the tracking's ball (need {:.0%}); "
          "median distance {:.2f} m".format(agreement, AGREE_M, MIN_AGREEMENT,
                                           float(np.median(dist)) if len(dist) else np.nan))
    print("would correct the tracking on {:.1%} of frames (more than {:.0f} m apart)".format(
        float(np.mean(dist > fuse.DISAGREE)) * both.sum() / frames.sum() if len(dist) else 0.0,
        fuse.DISAGREE))
    go = coverage >= MIN_COVERAGE and agreement >= MIN_AGREEMENT
    print("GO: run the whole match" if go else "STOP: keep Stage 1")
    return go


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
