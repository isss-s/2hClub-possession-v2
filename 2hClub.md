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
| **On the ball** | a team is on the ball when one of its players is within `radius` of it; the nearer team wins if both are | `radius` 2.0 m |
| **Change of possession** | the other team has been on the ball for `switch` frames in a row; dated back to where that run began | `switch` 20 frames (0.8 s) |
| **Touches** | a sharp change in the ball's velocity (more than `touch` m/s, 3 frames before vs 3 after) next to a player (within `touch_radius`); `touches` touches in a row by the other team, with none by the team in possession, also switch possession | 1 m/s, 4 m, 2 touches |
| **Loose ball / pass in flight** | nobody is on the ball: the last team keeps it (there is no "contested" state) | — |
| **Ball out of play** | the ball more than `out_margin` past a line: dead | 0.5 m |
| **Ball missing** | a gap longer than `gap_long`: dead; shorter gaps are the ball hidden behind players, and play goes on | 75 frames (3 s) |
| **Restart kick** | a stoppage of at least `min_stop` frames lasts until the ball first moves faster than `kick`, for at most `max_wait` | 1 s, 4 m/s, 20 s |
| **Slow ball** | the ±5 s median ball speed under `slow` for `min_dead` frames: dead | 0.5 m/s for 8 s, so in practice almost never |
| **Restarts** | after a stoppage, possession goes to the first team on the ball | — |
| **Abstention** | we never output `X` | — |

**How the settings were chosen** (`possession/tune.py`):

1. Start from values set by football reasoning before looking at any score. For example, a
   player in control has the ball at their feet, and a restart is a kick.
2. Try each setting in turn over a range, with the others fixed. Keep a new value only if it
   raises the mean spellF1 over the three sample games by at least 0.005 (smaller gains are
   noise) **and** costs no single game more than 0.01.
3. Repeat until nothing changes.
4. Do this from 8 starting points: the football-reasoned values, and 7 drawn at random from the
   ranges (fixed seed). Keep the best end point.

Step 4 was added late, and only after a test we fixed in advance. Judged by leaving one game
out, it had to beat the single start by at least 0.005, with no left-out game more than 0.01
worse. It did: 0.696 against 0.690, with its worst game 0.008 lower.

The official scorer (`evaluation/possession_metrics.py`) does the scoring. The chosen values are
committed in `possession/settings.json`.

## 4. Assets used

| asset | used? | how |
|---|---|---|
| Tactical tracking (ATD), players | **yes** | each team's nearest player to the ball |
| Tactical tracking (ATD), ball | **yes** | position, speed, out of play, gaps, touches |
| Tactical video | tried, dropped | a ball detector trained on outside footage, run on two pilot samples; it did not pass the check we set in advance (section 6) |
| Homography | tried, dropped | with the detector: pixels → pitch metres |
| Smart Tagging | no | — |
| Broadcast video / tracking | no | not synchronised with the tactical timeline |
| Public 5-minute ground truth | pipeline check only | run on each final file *after* its settings were fixed, to confirm that frames line up and labels mean what we think (section 5); never used to choose anything |
| External: Metrica Sports sample games | **yes** | choosing the settings (section 2) |

## 5. Results

**On the sample games** (event-based labels, official scorer):

| | spellF1 | macroF1 |
|---|---|---|
| starting values | 0.627 | 0.845 |
| tuned values | **0.706** (games 1/2/3: 0.697, 0.725, 0.698) | **0.868** |

**Do the settings carry over to a game they weren't tuned on?** Leave one game out: tune on the
other two games, score the one left out.

| left out | starting values | tuned on the other two |
|---|---|---|
| game 1 | 0.593 | **0.680** |
| game 2 | 0.666 | **0.709** |
| game 3 | 0.622 | **0.698** |
| mean | 0.627 | **0.696** |

The left-out game improves every time, by +0.04 to +0.09. That is our best estimate of the gain on
a new match.

**Shape of our output on the provided match** (both halves): 3.6 possessions a minute, mean
8.7 s, median 5.3 s. The sample games' event labels give 3.4–4.0 a minute, mean 10.6–12.2 s,
median 5.3–6.2 s.

**Pipeline check on the public 5 minutes:** spellF1 0.467, macroF1 0.681. The official baselines
score 0.038–0.424 spellF1 and 0.597–0.621 macroF1 there. The home share of alive time is within
1.3 points of the truth, which confirms the frames are aligned. We ran it only after each file's
settings were fixed, and never changed anything because of it. Five minutes (17 possessions) is
far too little to rank methods anyway.

## 6. Tried and dropped: a ball detector trained on other matches (`vision/`)

The tracking loses the ball on 41% of frames and sometimes puts it metres off. So we trained our
own ball detector, using **only footage of other matches**:

| step | file | what |
|---|---|---|
| data | `vision/build_external.py` | two CC BY 4.0 Roboflow Universe datasets: [football-ball-detection](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg) v4 and [football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) v20 (ball class only), labelled by Roboflow. Each image is un-stretched to 16:9 and scaled so its ball is 4–9 px, like ours, then cut into 640 tiles: 3,804 training tiles, 2,536 with a ball |
| train | `vision/train_external_colab.ipynb` | YOLO26s (COCO-pretrained), image size 1280, 25 epochs, Colab T4 |
| threshold | same | the lowest confidence with precision ≥ 0.90 on the outside test tiles: **0.20** (precision 0.92, recall 0.89) |
| frames | `vision/frame_spans.py` | the live frames of our output ±1 s (92,034), and per 2 s span only the tiles within ~12 m of the tracking's ball |
| detect | `vision/detect_spans.py`, `detect_colab.ipynb` | Colab L4, 10.8 frames/s |
| fuse | `vision/fuse.py` | drop candidates that sit still (fixed marks on the image); follow the candidates that move like a ball (no faster than 45 m/s, through gaps of up to 3 frames) |

**The check, set before any detection was seen.** Running the whole match would cost about 12
Colab units, so we first ran a pilot of about 2,900 frames. We would go ahead only if:

- the detector found a confident ball on at least 50% of pilot frames; and
- where both balls exist, at least 75% of detections were within 2 m of the tracking's ball.

This is an automatic check with no labels; we never compared anything with the video by eye.

| pilot | fusion | ball found | within 2 m of the tracking | impossible jumps (> 45 m/s) | decision |
|---|---|---|---|---|---|
| 1 (2,857 frames) | most confident candidate | 95.9% | **57.5%** | 12.5% of steps | stop |
| 2 (2,873 other frames) | motion filter, written after pilot 1 from physics alone | 88.1% | **63.3%** | 0.3% of steps | stop |

The motion filter removed the detector's jumps between objects. Even so, the two balls stayed
more than 3 m apart on about a quarter of the frames. Without labels we can't tell which one is
right there, so we kept the tracking's ball. The detector code stays in the repository as
documented work; it does not touch the submitted file.

## 7. Limitations and next steps

- **Different labels.** The ground truth comes from corrected tracking. Our tuning labels come
  from event data, a different source with its own idea of when control starts. Possession
  changes may be dated slightly differently.
- **Different tracking.** The sample games' ball moves more smoothly from frame to frame, so
  velocity-based rules find more touches there (19–24 a minute with our settings) than on our
  match (14 a minute).
- **Too much dead ball.** On the sample games our rules call 32–43% of frames dead where the events
  say 26–34%, and on our match 47%. Most of it is the tracking losing the ball for over 5 s while
  play goes on. We tried calling such a gap dead only while the players walk, since in the
  sample games players move at about 1.2 m/s in these live gaps against 0.9 m/s in stoppages,
  and their speeds are distributed like ours. Every setting we tried (thresholds 0.9–1.5 m/s,
  windows of 1–5 s, gap limits of 1–5 s) lowered the sample games' mean spellF1, by 0.005 to
  0.02, so it is not used. Without the ball, the team in possession during such a gap is unknown
  anyway, so calling it live mostly adds frames with the wrong team.
- **Fragments.** We tried merging very short stoppages and possessions into their neighbours.
  Judged by leaving one game out, it gained nothing (0.6956 to 0.6958) and cost one game 0.011, so
  it is not used.
- **Ball missing on 41% of frames.** We rely on the tracking's ball alone. Our detector trained on
  outside footage (section 6) disagreed with the tracking too often to be trusted without labels.
  A detector trained on footage from the same camera system, or a few hand-labelled frames of
  *other* matches from that camera, would be the next step.
- **Ball in the air.** The tracking puts the ball on the ground plane, so a high ball "lands"
  metres away from where it really is.
- **Only three sample games for tuning**, all from the same tracking provider. We looked at seven
  more: the Bundesliga matches of Bassek et al. (2025, CC BY 4.0), whose tracking carries a ball
  status and a possession team on every frame (`possession/idsse.py` reads them). We did not tune
  on them. Their possession follows a different convention from this challenge: 3.8–5.3
  possessions a minute with a median of 3–5 s, against the 3.4 a minute and 14.2 s mean that the
  starter kit reports for the public ground truth. Metrica's event-based labels (3.4–4.0 a
  minute, mean 10.6–12.2 s) are much closer to that, and settings tuned towards the Bundesliga
  labels would change possession too readily.

## 8. Reproduction

See [README.md](README.md). In short:

1. Clone Metrica's sample data into `external/` (Game 3 is read with kloppy).
2. Run `python possession/tune.py`, which writes `settings.json`.
3. Run `python possession/match.py`, which writes the file.

Everything runs on a CPU in about a minute, and the same inputs always give the same file.
