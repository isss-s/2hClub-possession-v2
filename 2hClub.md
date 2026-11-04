# 2hClub — Technical document

**More than a Hack 2027, Phase 1: Ball Possession**
Team: 2hClub · Submission file: `2hClub.csv`, made by `possession/match.py`

---

## 1. Approach

We turn the provided tracking into possession with **a few football rules**, and we choose the
rules' settings **on other games**: Metrica Sports' public sample games, whose event data gives a
possession label for every frame. Nothing is fitted to FC Barcelona vs Athletic Club. The rules
are applied to it unchanged.

Why rules rather than a learned model:

- **Every decision can be explained.** "Possession changed because Athletic had the ball at their
  feet for a second" can be checked by anyone watching the video.
- **There are few settings, so little to overfit.** We tune 13 numbers on about 425,000 labelled
  frames from three other games.
- **They carry over between games.** Settings tuned on two sample games also improve the
  third (section 5), which is the closest test we have of how they will do on our match.

## 2. Training data

The rules say no labels on the provided game, so all tuning uses outside data:

| dataset | source | who labelled it | how much | how we use it |
|---|---|---|---|---|
| Metrica Sports sample games 1, 2 and 3: tracking and event data | https://github.com/metrica-sports/sample-data (public; "acknowledge the source") | events: Metrica Sports; possession labels: built by our code from those events | 3 matches, 424,886 graded frames, 1,058 possessions | choose the rule settings |

**Possession labels from events** (`possession/metrica.py`), made by code with no person
deciding any frame:

- A team has the ball from its pass, carry, recovery, set piece or shot until the other team's.
  Challenges, aerial duels and ball losses don't hand the ball over by themselves. A recovery or
  the other team's next pass does. This follows the challenge's definition: a touch without
  control isn't a change of possession.
- The ball is dead from the end of the event before each restart (a set piece or a goal kick)
  until the restart itself.

Frames before kick-off and between the halves aren't graded.

The tracking of the sample games resembles ours where it matters. They lose the ball on 32–41% of
frames (ours: 41%), and their ball speeds have a similar median (5–6.5 m/s; ours 5 m/s) and 95th
percentile (17–18 m/s; ours 17 m/s). These
numbers come from automatic statistics of the tracking files only.

**We used no hand labels, no ground truth and no human-produced data about the provided game.**

## 3. The possession rules (`possession/rules.py`)

| rule | what it does | setting (tuned on Metrica) |
|---|---|---|
| **On the ball** | a team is on the ball when one of its players is within `radius` of it; the nearer team wins if both are | `radius` 2.5 m |
| **Change of possession** | the other team has been on the ball for `switch` frames in a row; dated back to where that run began | `switch` 25 frames (1 s) |
| **Touches** | a sharp change in the ball's velocity (more than `touch` m/s, 3 frames before vs 3 after) next to a player (within `touch_radius`); `touches` touches in a row by the other team, with none by the team in possession, also switch possession | 3 m/s, 2.5 m, 2 touches |
| **Loose ball / pass in flight** | nobody is on the ball: the last team keeps it (there is no "contested" state) | — |
| **Ball out of play** | the ball more than `out_margin` past a line: dead | 0.5 m |
| **Ball missing** | a gap longer than `gap_long`: dead; shorter gaps are the ball hidden behind players, and play goes on | 125 frames (5 s) |
| **Restart kick** | a stoppage of at least `min_stop` frames lasts until the ball first moves faster than `kick`, for at most `max_wait` | 2 s, 4 m/s, 10 s |
| **Slow ball** | the ±5 s median ball speed under `slow` for 12 s: dead | 0.5 m/s, so in practice almost never |
| **Restarts** | after a stoppage, possession goes to the first team on the ball | — |
| **Abstention** | we never output `X` | — |

**How the settings were chosen** (`possession/tune.py`):

1. Start from values set by football reasoning before looking at any score. For example, a
   player in control has the ball at their feet, and a restart is a kick.
2. Try each setting in turn over a range, with the others fixed. Keep a new value only if it
   raises the mean spellF1 over the three sample games by at least 0.005 (smaller gains are
   noise) **and** costs no single game more than 0.01.
3. Repeat until nothing changes.

The official scorer (`evaluation/possession_metrics.py`) does the scoring. The chosen values are
committed in `possession/settings.json`.

## 4. Assets used

| asset | used? | how |
|---|---|---|
| Tactical tracking (ATD), players | **yes** | each team's nearest player to the ball |
| Tactical tracking (ATD), ball | **yes** | position, speed, out of play, gaps, touches |
| Tactical video | no | — |
| Homography | no | — |
| Smart Tagging | no | — |
| Broadcast video / tracking | no | not synchronised with the tactical timeline |
| Public 5-minute ground truth | pipeline check only | run once on the final file, to confirm that frames line up and labels mean what we think (section 5); never used to choose anything |
| External: Metrica Sports sample games | **yes** | choosing the settings (section 2) |

## 5. Results

**On the sample games** (event-based labels, official scorer):

| | spellF1 | macroF1 |
|---|---|---|
| starting values | 0.627 | 0.845 |
| tuned values | **0.684** (games 1/2/3: 0.669, 0.708, 0.673) | **0.862** |

**Do the settings carry over to a game they weren't tuned on?** Leave one game out: tune on the
other two games, score the one left out.

| left out | starting values | tuned on the other two |
|---|---|---|
| game 1 | 0.593 | **0.688** |
| game 2 | 0.666 | **0.708** |
| game 3 | 0.622 | **0.673** |
| mean | 0.627 | **0.690** |

The left-out game improves every time, by +0.04 to +0.10, which is our best estimate of the gain on
a new match. Tuning on two games and tuning on all three choose the same settings.

**Shape of our output on the provided match**: about 3.2 possessions a minute, mean 10.1 s,
median 7.0 s. The sample games' event labels give 3.4–3.7 a minute, mean 10.6–12.2 s.

**Pipeline check on the public 5 minutes:** spellF1 0.400, macroF1 0.684. The official baselines
score 0.038–0.424 spellF1 and 0.597–0.621 macroF1 there. The home share of alive time is within
3 points of the truth, which confirms the frames are aligned. We ran this once, on the final
file, and changed nothing afterwards. Five minutes (17 possessions) is far too little to rank
methods anyway.

## 6. Limitations and next steps

- **Different labels.** The ground truth comes from corrected tracking. Our tuning labels come
  from event data, a different source with its own idea of when control starts. Possession
  changes may be dated slightly differently.
- **Different tracking.** The sample games' ball moves more smoothly from frame to frame, so
  velocity-based rules find far more touches there (about 18 a minute) than on our match (about
  1). On our match the touch rule therefore rarely fires, and most changes come from the 1-second
  control rule.
- **Too much dead ball.** On the sample games our rules call 41–43% of frames dead where the events
  say 30–34%, and on our match 46%. Most of it is the tracking losing the ball for over 5 s while
  play goes on.
- **Ball missing on 41% of frames.** We rely on the tracking's ball alone. A ball detector
  trained on outside football footage, and run on the tactical video, is our next step.
- **Ball in the air.** The tracking puts the ball on the ground plane, so a high ball "lands"
  metres away from where it really is.
- **Only three sample games for tuning**, all from the same tracking provider.

## 7. Reproduction

See [README.md](README.md). In short:

1. Clone Metrica's sample data into `external/` (Game 3 is read with kloppy).
2. Run `python possession/tune.py`, which writes `settings.json`.
3. Run `python possession/match.py`, which writes the file.

Everything runs on a CPU in about a minute, and the same inputs always give the same file.
