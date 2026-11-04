"""Metrica Sports' public sample games: the outside data we tune on.

    git clone --depth 1 https://github.com/metrica-sports/sample-data external/metrica-sample-data

Sample Games 1 to 3 come with tracking (players and ball, 25 fps) and event data, anonymous and
in 0..1 pitch coordinates (105 x 68 m), like our match's tracking. Games 1 and 2 are CSV files;
Game 3 is in the FIFA EPTS format with JSON events and is read with kloppy. Source: Metrica
Sports, https://github.com/metrica-sports/sample-data.

Possession labels for these games are built from their event data, by code, with no person
deciding any frame:

    - a team has the ball from its pass, carry, recovery, set piece or shot until the other
      team's; tackles, aerial duels and ball losses do not hand the ball over by themselves, a
      recovery or the other team's next pass does
    - the ball is dead from the end of the event before each restart (set piece or goal kick)
      until the restart itself
    - frames before the first and after the last event of each half are not graded (X)
"""
import json
import os
import pickle
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "external", "metrica-sample-data", "data")
CACHE = os.path.join(ROOT, "external", "cache")
PITCH = (105.0, 68.0)
GAMES = (1, 2, 3)
HOME, AWAY = 0, 1

OWNING = ("PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT")


def _folder(game):
    return os.path.join(DATA, "Sample_Game_{}".format(game))


def _csv(game, kind):
    return os.path.join(_folder(game), "Sample_Game_{0}_{1}.csv".format(game, kind))


def _read_tracking_csv(game):
    frames, teams = [], []
    for side, code in (("Home", HOME), ("Away", AWAY)):
        raw = pd.read_csv(_csv(game, "RawTrackingData_{}_Team".format(side)), skiprows=2)
        names = [c for c in raw.columns if c.startswith("Player")]
        xy = np.stack([raw[[c, raw.columns[raw.columns.get_loc(c) + 1]]].values for c in names], 1)
        frames.append(xy)
        teams += [code] * len(names)
        if code == HOME:
            ball = raw[["Ball", raw.columns[raw.columns.get_loc("Ball") + 1]]].values
    return ball, np.concatenate(frames, axis=1), np.array(teams)


def _read_tracking_epts(game):
    from kloppy import metrica
    prefix = os.path.join(_folder(game), "Sample_Game_{}_".format(game))
    data = metrica.load_tracking_epts(meta_data=prefix + "metadata.xml",
                                      raw_data=prefix + "tracking.txt")
    slots, teams = {}, []
    for code, side in enumerate(data.metadata.teams):     # the first team is the home side
        for player in side.players:
            slots[player.player_id] = len(teams)
            teams.append(code)
    n = max(f.frame_id for f in data.frames)
    ball = np.full((n, 2), np.nan)
    players = np.full((n, len(teams), 2), np.nan)
    for f in data.frames:
        i = f.frame_id - 1
        if f.ball_coordinates is not None:
            ball[i] = f.ball_coordinates.x, f.ball_coordinates.y
        for player, point in f.players_coordinates.items():
            if point is not None:
                players[i, slots[player.player_id]] = point.x, point.y
    return ball, players, np.array(teams)


def read_tracking(game):
    """Ball (n, 2) and players (n, slots, 2) in metres, NaN where missing; team (slots,)."""
    path = os.path.join(CACHE, "game{}.pkl".format(game))
    if os.path.exists(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)
    reader = _read_tracking_csv if game in (1, 2) else _read_tracking_epts
    ball, players, team = reader(game)
    scale = np.array(PITCH, dtype=np.float32)
    out = ball.astype(np.float32) * scale, players.astype(np.float32) * scale, team
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(out, handle)
    return out


def read_events(game):
    """One row per event: type, subtype, team (HOME/AWAY), start and end (0-based frames), period."""
    if game in (1, 2):
        raw = pd.read_csv(_csv(game, "RawEventsData"))
        events = pd.DataFrame({
            "type": raw["Type"], "subtype": raw["Subtype"].fillna(""),
            "team": np.where(raw["Team"] == "Home", HOME, AWAY),
            "start": raw["Start Frame"], "end": raw["End Frame"], "period": raw["Period"]})
    else:
        with open(os.path.join(_folder(game), "Sample_Game_{}_events.json".format(game))) as handle:
            raw = json.load(handle)["data"]
        with open(os.path.join(_folder(game), "Sample_Game_{}_metadata.xml".format(game))) as handle:
            home = re.search(r'<Team id="([^"]+)"', handle.read()).group(1)   # the first team
        rows = []
        for e in raw:
            sub = e.get("subtypes")
            if isinstance(sub, list):
                sub = " ".join(s["name"] for s in sub)
            elif isinstance(sub, dict):
                sub = sub["name"]
            rows.append({"type": e["type"]["name"], "subtype": sub or "",
                         "team": HOME if e["team"]["id"] == home else AWAY,
                         "start": e["start"]["frame"], "end": e["end"]["frame"] or 0,
                         "period": e["period"]})
        events = pd.DataFrame(rows)
    events["start"] = events["start"] - 1       # 0-based, like the tracking rows
    events["end"] = np.maximum(events["end"] - 1, events["start"])
    return events.sort_values(["period", "start"], kind="stable").reset_index(drop=True)


def is_restart(events):
    goal_kick = (events["type"] == "PASS") & events["subtype"].str.contains("GOAL KICK")
    return ((events["type"] == "SET PIECE") | goal_kick).values


def labels(game, n):
    """Per frame: 'H', 'A', 'D' or 'X' (not graded), from the event data."""
    events = read_events(game)
    out = np.full(n, "X", dtype="<U1")
    restart = is_restart(events)
    for _, half in events.groupby("period"):
        idx = half.index.values
        first, last = half["start"].min(), half["end"].max()
        owner = None
        for row, i in enumerate(idx):
            nxt = events.loc[idx[row + 1], "start"] if row + 1 < len(idx) else last + 1
            if events.loc[i, "type"] in OWNING:
                owner = "H" if events.loc[i, "team"] == HOME else "A"
            if owner is not None:
                out[events.loc[i, "start"]:nxt] = owner
        for row, i in enumerate(idx):
            if restart[i] and row > 0 and events.loc[i, "start"] > first:
                stop = events.loc[idx[row - 1], "end"] + 1
                out[stop:events.loc[i, "start"]] = "D"
    return out
