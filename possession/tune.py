"""Choose the rule settings on Metrica's sample games, never on our match.

    python possession/tune.py            # writes possession/settings.json

Coordinate search: each setting in turn is tried over its range with the others fixed, and the
value with the best mean spellF1 over the sample games is kept if it gains at least MIN_GAIN on the
mean and costs no single game more than MAX_LOSS; this repeats until nothing changes. The search
runs from STARTS starting points -- the football-reasoned values, and STARTS - 1 drawn at random
from the ranges (seed SEED) -- and the best end point is kept. The scorer is the official one
(evaluation/possession_metrics.py), run on the event-based labels of metrica.py.

Multiple starts were adopted because, judged by leaving one game out, they beat the single start
(0.696 vs 0.690 on the left-out games, no game more than 0.008 worse).
"""
import random
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import metrica  # noqa: E402
import possession_metrics as pm  # noqa: E402
import rules  # noqa: E402

SETTINGS = os.path.join(HERE, "settings.json")
MIN_GAIN = 0.005   # a change must gain this much mean spellF1 to be kept; smaller gains are noise
MAX_LOSS = 0.01    # ...and must not cost any single game more than this
STARTS = 8
SEED = 0

# Starting point: values from football reasoning, before looking at any score.
START = {
    "radius": 1.5,          # m; a player in control has the ball at his feet
    "switch": 25,           # frames; "a player keeps the ball for a few seconds" -- 1 s
    "gap_long": 125,        # frames; the ball is rarely hidden behind players for 5 s
    "out_margin": 0.5,      # m past the line, for tracking noise
    "slow_window": 125,     # frames each side for the median speed
    "slow": 3.0,            # m/s; a ball placed for a restart barely moves
    "min_dead": 300,        # frames; a 12 s slow stretch
    "min_stop": 50,         # frames; stoppages long enough to wait for a restart kick
    "kick": 4.0,            # m/s; the restart kick
    "max_wait": 250,        # frames; at most 10 s
    "touch": 3.0,           # m/s; velocity change of a touch
    "touch_radius": 2.5,    # m
    "touches": 3,           # in a row by the other team; 0 turns the touch rule off
}

SEARCH = {
    "radius": [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0],
    "switch": [5, 10, 15, 20, 25, 35, 50, 75],
    "gap_long": [25, 50, 75, 100, 125, 150, 200, 250, 375],
    "out_margin": [0.0, 0.5, 1.0, 2.0],
    "slow": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    "min_dead": [125, 200, 300, 450, 600],
    "min_stop": [25, 50, 75, 125],
    "kick": [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0],
    "max_wait": [125, 250, 375, 500],
    "touches": [0, 2, 3, 4, 5],
    "touch": [1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    "touch_radius": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
}


def load_games():
    games = []
    for g in metrica.GAMES:
        ball, players, team = metrica.read_tracking(g)
        gt = pm.label_codes(metrica.labels(g, len(ball)))
        games.append((ball, players, team, gt, pm.valid_runs(gt)))
    return games


def score(games, p):
    """Mean spellF1 and macroF1 over the games."""
    spell, macro = [], []
    for ball, players, team, gt, valid in games:
        pred = pm.label_codes(rules.label(ball, players, team, metrica.PITCH, p))
        spell.append(pm.spell_scores(gt, pred, valid)["mean_f1"])
        macro.append(pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))))
    return float(np.mean(spell)), float(np.mean(macro)), spell


def search(games, p, verbose=True):
    p = dict(p)
    best, _, per_game = score(games, p)
    if verbose:
        print("start  spellF1 {:.4f}".format(best))
    for sweep in range(4):
        changed = False
        for key, values in SEARCH.items():
            for v in values:
                if v == p[key]:
                    continue
                trial = dict(p, **{key: v})
                s, _, trial_games = score(games, trial)
                worst = min(np.subtract(trial_games, per_game))
                if s > best + MIN_GAIN and worst > -MAX_LOSS:
                    best, p, per_game, changed = s, trial, trial_games, True
                    if verbose:
                        print("sweep {} {:>12} = {:<6} spellF1 {:.4f}".format(sweep, key, v, best))
        if not changed:
            break
    return p, best


def starting_points():
    rng = random.Random(SEED)
    points = [dict(START)]
    for _ in range(STARTS - 1):
        p = dict(START)
        for key, values in SEARCH.items():
            p[key] = rng.choice(values)
        points.append(p)
    return points


def search_all(games, verbose=False):
    """The best end point of `search` over all starting points."""
    best_p, best_s = None, -1.0
    for k, p0 in enumerate(starting_points()):
        p, s = search(games, p0, verbose=False)
        if verbose:
            print("start {}: spellF1 {:.4f}".format(k, s), flush=True)
        if s > best_s:
            best_p, best_s = p, s
    return best_p, best_s


def leave_one_out(games):
    """For each game: tune on the others, score this one. The fair estimate for a new match."""
    rows = []
    for i in range(len(games)):
        p, _ = search_all(games[:i] + games[i + 1:])
        rows.append((score([games[i]], START)[0], score([games[i]], p)[0]))
    return rows


if __name__ == "__main__":
    games = load_games()
    print("leave one game out (tuned on the other games):")
    for g, (before, after) in zip(metrica.GAMES, leave_one_out(games)):
        print("  game {}: starting values {:.3f} -> tuned {:.3f}".format(g, before, after))
    p, best = search_all(games, verbose=True)
    spell, macro, per_game = score(games, p)
    print("chosen:", json.dumps(p))
    print("spellF1 {:.4f} (per game {})  macroF1 {:.4f}".format(
        spell, ", ".join("{:.3f}".format(s) for s in per_game), macro))
    with open(SETTINGS, "w") as handle:
        json.dump(p, handle, indent=2)
        handle.write("\n")
    print("wrote", os.path.relpath(SETTINGS, ROOT))
