"""Seven Bundesliga matches with per-frame ball status and possession: more outside data to tune on.

    Bassek, Rein, Weber, Memmert (2025). An integrated dataset of spatiotemporal and event data in
    elite soccer. Scientific Data 12, 195. CC BY 4.0. https://doi.org/10.1038/s41597-025-04505-y
    Files (DFL position and match-information XML) into external/idsse/, from
    https://doi.org/10.6084/m9.figshare.28196177 or its mirror
    https://huggingface.co/datasets/pysport/idsse-data

Each ball frame carries BallStatus (1 in play, 0 not) and BallPossession (1 home, 2 guest), set
by the league's data provider. That is the same kind of label as our match's ground truth (dead
where the tracking marks the ball dead, otherwise the team it names), so the labels here are read
straight from the files: 'D' where BallStatus is 0, else 'H' or 'A'.

The tracking is 25 fps and complete: the ball is never missing. Our match's tracking loses the
ball on 41% of frames, so these matches are used only for the settings that decide who has the
ball in live play; the stoppage settings stay with Metrica's games (see tune.py).
"""
import glob
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "external", "idsse")
CACHE = os.path.join(ROOT, "external", "cache")
GAMES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
HOME, AWAY = 0, 1
HALF_GAP = 1500     # frames of nothing put between the halves, so no rule looks across them

FRAMESET = re.compile(r'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" PersonId="([^"]*)"')
FRAME = re.compile(r'<Frame N="(\d+)" [^>]*?X="(-?[\d.]+)" Y="(-?[\d.]+)"')
BALL = re.compile(r'BallPossession="(\d)" BallStatus="(\d)"')


def _file(game, kind):
    hits = glob.glob(os.path.join(DATA, "DFL_{}_*{}.xml".format(kind, game)))
    if not hits:
        raise FileNotFoundError("no {} file for {} in {}".format(kind, game, DATA))
    return hits[0]


def _parse(game):
    with open(_file(game, "02_01"), encoding="utf-8") as handle:
        info = handle.read()
    home = re.search(r'HomeTeamId="([^"]+)"', info).group(1)
    guest = re.search(r'GuestTeamId="([^"]+)"', info).group(1)
    pitch = (float(re.search(r'PitchX="([\d.]+)"', info).group(1)),
             float(re.search(r'PitchY="([\d.]+)"', info).group(1)))
    tracks = {}                     # (section, team, person) -> [(n, x, y, possession, status)]
    key = None
    with open(_file(game, "04_03"), encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                if key is None:
                    continue
                m = FRAME.match(line)
                b = BALL.search(line) if key[1] == "BALL" else None
                tracks[key].append((int(m.group(1)), float(m.group(2)), float(m.group(3)),
                                    int(b.group(1)) if b else 0, int(b.group(2)) if b else 0))
            elif line.startswith("<FrameSet "):
                section, team, person = FRAMESET.match(line).groups()
                key = (section, team, person) if team in (home, guest, "BALL") else None
                if key:
                    tracks[key] = []
    slots = sorted({(t, p) for _, t, p in tracks if t != "BALL"}, key=lambda s: (s[0] != home, s[1]))
    team = np.array([HOME if t == home else AWAY for t, _ in slots])
    parts = []
    for section in ("firstHalf", "secondHalf"):
        rows = np.array(tracks[next(k for k in tracks if k[0] == section and k[1] == "BALL")])
        first, n = int(rows[0, 0]), len(rows)
        ball = rows[:, 1:3] + np.array(pitch) / 2
        labels = np.where(rows[:, 4] == 0, "D", np.where(rows[:, 3] == 1, "H", "A"))
        players = np.full((n, len(slots), 2), np.nan)
        for k, (t, p) in enumerate(slots):
            if (section, t, p) in tracks and tracks[(section, t, p)]:
                r = np.array(tracks[(section, t, p)])
                i = r[:, 0].astype(int) - first
                ok = (i >= 0) & (i < n)
                players[i[ok], k] = r[ok, 1:3] + np.array(pitch) / 2
        parts.append((ball, players, labels))
    gap = HALF_GAP
    ball = np.concatenate([parts[0][0], np.full((gap, 2), np.nan), parts[1][0]])
    players = np.concatenate([parts[0][1], np.full((gap, len(slots), 2), np.nan), parts[1][1]])
    labels = np.concatenate([parts[0][2], np.full(gap, "X"), parts[1][2]])
    return ball.astype(np.float32), players.astype(np.float32), team, labels, np.array(pitch)


def read(game):
    """ball (n, 2), players (n, slots, 2) in metres from a corner; team (slots,); labels (n,) of
    'H'/'A'/'D' ('X' between the halves); pitch (length, width)."""
    path = os.path.join(CACHE, "idsse_{}.npz".format(game))
    if not os.path.exists(path):
        ball, players, team, labels, pitch = _parse(game)
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(path, ball=ball, players=players, team=team, labels=labels, pitch=pitch)
    data = np.load(path)
    return data["ball"], data["players"], data["team"], data["labels"], tuple(data["pitch"])
