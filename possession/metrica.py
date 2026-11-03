"""Metrica Sports' public sample games: the outside data we tune on.

    git clone --depth 1 https://github.com/metrica-sports/sample-data external/metrica-sample-data

Sample Games 1 and 2 come with tracking (players and ball, 25 fps) and event data, both anonymous
and in the same 0..1 pitch coordinates as our match's tracking (105 x 68 m). Source: Metrica
Sports, https://github.com/metrica-sports/sample-data.

Possession labels for these games are built from their event data, by code, with no person
deciding any frame:

    - a team has the ball from its pass, recovery, set piece or shot until the other team's;
      tackles, aerial duels and ball losses do not hand the ball over by themselves, a recovery
      or the other team's next pass does
    - the ball is dead from the end of the event before each restart (set piece or goal kick)
      until the restart itself
    - frames before the first and after the last event of each half are not graded (X)
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "external", "metrica-sample-data", "data")
PITCH = (105.0, 68.0)
GAMES = (1, 2)
HOME, AWAY = 0, 1

OWNING = ("PASS", "RECOVERY", "SET PIECE", "SHOT")


def _path(game, kind):
    return os.path.join(DATA, "Sample_Game_{0}".format(game),
                        "Sample_Game_{0}_{1}.csv".format(game, kind))


def read_tracking(game):
    """Ball (n, 2) and players (n, slots, 2) in metres, team (slots,), period (n,)."""
    length, width = PITCH
    frames, teams = [], []
    for side, code in (("Home", HOME), ("Away", AWAY)):
        raw = pd.read_csv(_path(game, "RawTrackingData_{}_Team".format(side)), skiprows=2)
        names = [c for c in raw.columns if c.startswith("Player")]
        xy = np.stack([raw[[c, raw.columns[raw.columns.get_loc(c) + 1]]].values for c in names], 1)
        frames.append(xy)
        teams += [code] * len(names)
        if code == HOME:
            period = raw["Period"].values
            ball = raw[["Ball", raw.columns[raw.columns.get_loc("Ball") + 1]]].values
    players = np.concatenate(frames, axis=1).astype(np.float32)
    players[..., 0] *= length
    players[..., 1] *= width
    ball = ball.astype(np.float32) * np.array([length, width], dtype=np.float32)
    return ball, players, np.array(teams), period


def read_events(game):
    events = pd.read_csv(_path(game, "RawEventsData"))
    events["team"] = np.where(events["Team"] == "Home", HOME, AWAY)
    events["start"] = events["Start Frame"] - 1       # 0-based, like our tracking rows
    events["end"] = np.maximum(events["End Frame"], events["Start Frame"]) - 1
    return events


def is_restart(events):
    goal_kick = (events["Type"] == "PASS") & events["Subtype"].fillna("").str.startswith("GOAL KICK")
    return (events["Type"] == "SET PIECE") | goal_kick


def labels(game, n):
    """Per frame: 'H', 'A', 'D' or 'X' (not graded), from the event data."""
    events = read_events(game)
    out = np.full(n, "X", dtype="<U1")
    restart = is_restart(events).values
    for _, half in events.groupby("Period"):
        idx = half.index.values
        first, last = half["start"].min(), half["end"].max()
        owner = None
        for row, i in enumerate(idx):
            e = events.loc[i]
            nxt = events.loc[idx[row + 1], "start"] if row + 1 < len(idx) else last + 1
            if e["Type"] in OWNING:
                owner = "H" if e["team"] == HOME else "A"
            if owner is not None:
                out[e["start"]:nxt] = owner
        for row, i in enumerate(idx):
            if restart[i] and row > 0 and events.loc[i, "start"] > first:
                stop = events.loc[idx[row - 1], "end"] + 1
                out[stop:events.loc[i, "start"]] = "D"
    return out
