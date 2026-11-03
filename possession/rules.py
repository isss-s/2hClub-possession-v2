"""Football rules that turn ball and player positions into possession, one label per frame.

Every function takes plain arrays, so the same rules run on Metrica's sample games (where their
settings are chosen) and on our match (where they are only applied):

    ball     (n, 2) metres, NaN where the ball was not tracked
    players  (n, slots, 2) metres, NaN where a slot is empty
    team     (slots,) HOME or AWAY

The rules, in the order `label` applies them:

    control       a team is on the ball when one of its players is within `radius` m of it
    dead_frames   the ball outside the lines, or missing for longer than `gap_long` frames
    slow_ball     the ball barely moving for a long stretch (placed for a restart)
    until_kick    a stoppage lasts until the ball is kicked again
    touches       a sharp change in the ball's velocity next to a player
    walk          possession carried forward; it changes after the other team has been on the
                  ball for `switch` frames in a row, or touched it `touches` times in a row
"""
import numpy as np
import pandas as pd

HOME, AWAY, NOBODY, LOOSE = 0, 1, -1, -2
FPS = 25
MAX_BALL_SPEED = 45.0  # m/s; faster than any kick, so a tracking jump
STEP = 3               # frames before and after a touch over which the ball's velocity is measured



def runs(mask):
    """(start, end) of every run of True, end exclusive."""
    edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def drop_short(mask, min_len):
    """Keep only the True runs that last at least `min_len` frames."""
    out = np.zeros_like(mask)
    for start, end in runs(mask):
        if end - start >= min_len:
            out[start:end] = True
    return out


def nearest(ball, players, team):
    """(n,) distance to the nearest home player and to the nearest away player."""
    dist = np.sqrt(((players - ball[:, None, :]) ** 2).sum(-1))
    dist = np.where(np.isnan(dist), np.inf, dist)
    home = np.where(team[None, :] == HOME, dist, np.inf).min(1)
    away = np.where(team[None, :] == AWAY, dist, np.inf).min(1)
    return home, away


def control(ball, players, team, radius):
    """Per frame: HOME/AWAY if that team has a player within `radius` m, LOOSE if neither,
    NOBODY if the ball is missing. The nearer team wins when both are close."""
    home, away = nearest(ball, players, team)
    out = np.full(len(ball), LOOSE, dtype=np.int8)
    close_home, close_away = home < radius, away < radius
    out[close_home & ~close_away] = HOME
    out[close_away & ~close_home] = AWAY
    both = close_home & close_away
    out[both] = np.where(home[both] <= away[both], HOME, AWAY)
    out[np.isnan(ball[:, 0])] = NOBODY
    return out


def dead_frames(ball, pitch, gap_long, out_margin):
    """The ball outside the lines by more than `out_margin` m, or missing for over `gap_long`
    frames (shorter gaps are the ball hidden behind players)."""
    length, width = pitch
    seen = ~np.isnan(ball[:, 0])
    with np.errstate(invalid="ignore"):
        dead = seen & ((ball[:, 0] < -out_margin) | (ball[:, 0] > length + out_margin) |
                       (ball[:, 1] < -out_margin) | (ball[:, 1] > width + out_margin))
    for start, end in runs(~seen):
        if start > 0 and end < len(ball) and end - start > gap_long:
            dead[start:end] = True
    return dead


def slow_ball(ball, window, threshold, min_len):
    """A median ball speed under `threshold` m/s over +-`window` frames, for `min_len` frames."""
    speed = np.full(len(ball), np.nan)
    speed[1:] = np.hypot(*np.diff(ball, axis=0).T) * FPS
    median = pd.Series(speed).rolling(2 * window + 1, center=True, min_periods=window).median()
    return drop_short((median < threshold).values, min_len)


def ball_speed(ball):
    """Ball speed in m/s, median over 3 frames. NaN where the ball is missing."""
    speed = np.full(len(ball), np.nan)
    speed[1:] = np.hypot(*np.diff(ball, axis=0).T) * FPS
    speed[speed > MAX_BALL_SPEED] = np.nan
    return pd.Series(speed).rolling(3, center=True, min_periods=1).median().values


def until_kick(stopped, speed, min_stop, kick, max_wait):
    """`stopped`, with each run of at least `min_stop` frames extended to the restart kick:
    the first frame the ball moves faster than `kick` m/s, at most `max_wait` frames later."""
    out = stopped.copy()
    for start, end in runs(stopped):
        if end - start < min_stop:
            continue
        stop = min(end + max_wait, len(stopped))
        kicks = np.flatnonzero(speed[end:stop] > kick)
        out[end:end + kicks[0] if len(kicks) else stop] = True
    return out


def touches(ball, players, team, change, radius):
    """Per frame: the team that touched the ball there (HOME/AWAY), else -1. A touch is a jump
    of more than `change` m/s between the ball's velocity over the STEP frames before and the
    STEP frames after, credited to the nearest player if within `radius` m."""
    n = len(ball)
    v_in = np.full((n, 2), np.nan)
    v_out = np.full((n, 2), np.nan)
    v_in[STEP:] = (ball[STEP:] - ball[:-STEP]) / STEP * FPS
    v_out[:-STEP] = v_in[STEP:]
    jump = np.nan_to_num(np.hypot(*(v_out - v_in).T))
    peak = jump > change
    for s in range(1, STEP + 1):            # keep the sharpest frame of each change
        peak[s:] &= jump[s:] >= jump[:-s]
        peak[:-s] &= jump[:-s] > jump[s:]
    out = np.full(n, -1, dtype=np.int8)
    for f in np.flatnonzero(peak):
        dist = np.hypot(*(players[f] - ball[f]).T)
        dist = np.where(np.isnan(dist), np.inf, dist)
        who = np.argmin(dist)
        if dist[who] < radius:
            out[f] = team[who]
    return out


def walk(ctrl, dead, switch, touched=None, need=0):
    """Carry the possessing team forward through loose balls and passes in flight.

    Possession changes after the other team has been on the ball for `switch` frames in a row,
    dated back to where that run began; or, if `need` > 0, after `need` touches in a row by the
    other team with none by the team in possession between them, dated back to the first. After
    a stoppage the first team on the ball (or touching it) has it.
    """
    out = np.full(len(ctrl), NOBODY, dtype=np.int8)
    current, run, candidate = NOBODY, 0, NOBODY
    streak_team, streak, streak_start = -1, 0, 0
    for i in range(len(ctrl)):
        if dead[i]:
            current, run, streak_team, streak = NOBODY, 0, -1, 0
            continue
        t = touched[i] if need else -1
        if t >= 0:
            if current == NOBODY:
                current = t
            elif t == current:
                streak_team, streak = -1, 0
            else:
                if t == streak_team:
                    streak += 1
                else:
                    streak_team, streak, streak_start = t, 1, i
                if streak >= need:
                    current = t
                    out[streak_start:i] = current
                    streak_team, streak, run = -1, 0, 0
        c = ctrl[i]
        if c >= 0:
            if current == NOBODY:
                current, run = c, 0
            elif c != current:
                if c == candidate:
                    run += 1
                else:
                    candidate, run = c, 1
                if run >= switch:
                    current = c
                    out[i - run + 1:i] = current
                    run = 0
            else:
                run = 0
        out[i] = current
    return out


def label(ball, players, team, pitch, p, touch_ball=None):
    """'H'/'A'/'D' per frame. `p` holds the settings (possession/settings.json). `touch_ball` is the ball
    the touch rule looks at (default: `ball`)."""
    ctrl = control(ball, players, team, p["radius"])
    dead = dead_frames(ball, pitch, p["gap_long"], p["out_margin"])
    slow = slow_ball(ball, p["slow_window"], p["slow"], p["min_dead"])
    dead |= until_kick(dead | slow, ball_speed(ball), p["min_stop"], p["kick"], p["max_wait"]) & ~slow
    touched = None
    if p["touches"]:
        tb = ball if touch_ball is None else touch_ball
        touched = touches(tb, players, team, p["touch"], p["touch_radius"])
    codes = walk(ctrl, dead, p["switch"], touched, p["touches"])
    labels = np.where(codes == HOME, "H", np.where(codes == AWAY, "A", "D"))
    labels[slow] = "D"
    return labels
