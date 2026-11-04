"""Which frames the ball detector runs on, and which of their tiles.

    python vision/frame_spans.py             # -> external/spans_live.json, external/spans_pilot.json

The detector's ball is only used to decide who controls the ball during live play (the stoppage
rules keep the tracking's ball), so it is run on the frames possession/match.py calls live, plus
MARGIN frames either side, inside the two halves of play. They are cut into spans of SPAN frames;
each span lists the tiles of vision/detect_spans.py within SEARCH_PX of where the tracking puts the
ball in any of its frames (or last put it, if it has lost it). That is about 12 m: wide enough to
catch the tracking's ball being in the wrong place, which is what the detector is for. A span with
no tracked ball before it gets all tiles.

The pilot is PILOT_SPANS spans spread evenly over the live spans.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, "possession"), os.path.join(ROOT, "starter")]
import rules  # noqa: E402
from detect_spans import ORIGINS, TILE  # noqa: E402
from games import GAME, GAMES  # noqa: E402
from load_homography import load as load_homography  # noqa: E402
from match import OUT as LABELS, read_tracking  # noqa: E402

MARGIN = 25
SPAN = 50
SEARCH_PX = 200
PILOT_SPANS = 60
WIDTH, HEIGHT = 1920, 1080


def ball_pixels(ball, matrices):
    """(n, 2) pixel position of the tracking's ball; NaN where it is missing."""
    length, width = GAMES[GAME]["pitch"]
    x, y = ball[:, 0] / length, ball[:, 1] / width
    m = matrices
    den = m[:, 2, 0] * x + m[:, 2, 1] * y + m[:, 2, 2]
    return np.stack([(m[:, 0, 0] * x + m[:, 0, 1] * y + m[:, 0, 2]) / den,
                     (m[:, 1, 0] * x + m[:, 1, 1] * y + m[:, 1, 2]) / den], -1)


def tiles_near(px):
    """Indices into ORIGINS of the tiles within SEARCH_PX of any pixel point in `px` (k, 2)."""
    px = px[~np.isnan(px).any(1)]
    if not len(px):
        return list(range(len(ORIGINS)))
    out = [k for k, (x0, y0) in enumerate(ORIGINS)
           if ((px[:, 0] > x0 - SEARCH_PX) & (px[:, 0] < x0 + TILE + SEARCH_PX) &
               (px[:, 1] > y0 - SEARCH_PX) & (px[:, 1] < y0 + TILE + SEARCH_PX)).any()]
    return out or list(range(len(ORIGINS)))


def main():
    labels = pd.read_csv(LABELS)["label"].values.astype(str)
    ball, _, _ = read_tracking()
    matrices, _, valid = load_homography()
    px = ball_pixels(ball, matrices)
    held = pd.DataFrame(px).ffill().values          # where the tracking last had the ball
    halves = np.zeros(len(labels), dtype=bool)
    for half, (a, b) in GAMES[GAME]["halves"].items():
        off = GAMES[GAME]["offsets"][half]
        halves[a - off:b - off] = True
    live = (labels != "D") & halves
    grown = live.copy()
    for s in range(1, MARGIN + 1):
        grown[s:] |= live[:-s]
        grown[:-s] |= live[s:]
    spans = []
    for start, end in rules.runs(grown & halves):
        for a in range(start, end, SPAN):
            b = min(a + SPAN, end)
            seen = np.where(np.isnan(px[a:b]), held[a:b], px[a:b])[valid[a:b]]
            spans.append({"start": int(a), "end": int(b), "tiles": tiles_near(seen)})
    pick = np.linspace(0, len(spans) - 1, PILOT_SPANS).round().astype(int)
    pilot = [spans[i] for i in sorted(set(pick))]
    for name, chosen in (("spans_live.json", spans), ("spans_pilot.json", pilot)):
        frames = sum(s["end"] - s["start"] for s in chosen)
        tiles = sum((s["end"] - s["start"]) * len(s["tiles"]) for s in chosen)
        path = os.path.join(ROOT, "external", name)
        with open(path, "w") as handle:
            json.dump(chosen, handle)
        print("{}: {} spans, {} frames, {:.2f} tiles per frame (of {})".format(
            name, len(chosen), frames, tiles / frames, len(ORIGINS)))


if __name__ == "__main__":
    main()
