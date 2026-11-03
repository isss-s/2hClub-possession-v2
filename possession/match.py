"""Possession for our match (FC Barcelona vs Athletic Club) with the settings chosen on Metrica's games.

    python possession/match.py           # writes evaluation/results/2hClub.csv

Nothing here is fitted to this match: it reads the provided tracking, applies the rules in
rules.py with possession/settings.json (written by tune.py from Metrica's sample games), and
writes the file.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
import rules  # noqa: E402
from baseline import team_of_each_track, write_submission  # noqa: E402
from games import GAME, GAMES, require, source_frames  # noqa: E402

OUT = os.path.join(ROOT, "evaluation", "results", "2hClub.csv")


def read_tracking():
    """Ball (n, 2) and players (n, slots, 2) in metres, NaN where not tracked; team (slots,)."""
    length, width = GAMES[GAME]["pitch"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    n = source_frames("tactical")
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.full((n, len(team), 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            frame, groups, ball_field = line.split(":")
            i = int(frame) - 1
            for slot, group in enumerate(groups.split(";")):
                if group[0] != "N":
                    values = group.split(",")
                    players[i, slot, 0] = float(values[0]) * length
                    players[i, slot, 1] = float(values[1]) * width
            if ball_field[0] != "N":
                values = ball_field.split(",")
                ball[i, 0] = float(values[0]) * length
                ball[i, 1] = float(values[1]) * width
    return ball, players, np.asarray(team)


if __name__ == "__main__":
    with open(os.path.join(HERE, "settings.json")) as handle:
        settings = json.load(handle)
    ball, players, team = read_tracking()
    labels = rules.label(ball, players, team, GAMES[GAME]["pitch"], settings)
    print("wrote", os.path.relpath(write_submission(OUT, labels), ROOT))
