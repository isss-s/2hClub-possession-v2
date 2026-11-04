"""Ball-detector training tiles from outside footage only, sized like our match's ball.

    python vision/build_external.py          # external/roboflow/* -> external/tiles/, tiles.zip

Inputs (CC BY 4.0, downloaded from Roboflow Universe in YOLOv8 format into external/roboflow/):
    ball/      football-ball-detection v4     https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg
    players/   football-players-detection v20 https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc
               (only its ball class is used)

Both exports are 16:9 match frames stretched to a square. The detector will see our match as
640 x 640 tiles of a 1920 x 1080 frame, upscaled 2x, with a ball of a few pixels. So each outside
image is un-stretched back to 16:9 and scaled so that its ball is BALL_PX pixels across (drawn at
random per tile), then placed in a 640 x 640 tile: cropped around the ball if the scaled frame is
larger, padded with grey if smaller. Images without a ball become negative tiles.
"""
import glob
import os
import random
import shutil
import zipfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "external", "roboflow")
OUT = os.path.join(ROOT, "external", "tiles")
TILE = 640
BALL_PX = (4.0, 9.0)        # ball diameter in our frames is about 6 px at 1920 x 1080
VARIANTS = 2                # tiles per image with a ball, each at its own random size
NEG_PER_POS = 0.5           # ball-free tiles per ball tile
SPLITS = {"train": "train", "valid": "val", "test": "test"}
SEED = 0


def read_labels(path):
    """Ball boxes (cx, cy, w, h) in 0-1 of the stretched image. Class 0 is ball in both sets."""
    if not os.path.exists(path):
        return []
    rows = [line.split() for line in open(path) if line.strip()]
    return [tuple(map(float, r[1:5])) for r in rows if int(r[0]) == 0]


def ball_px_1920(box):
    """Ball diameter in px of the original 1920 x 1080 frame."""
    return float(np.sqrt(box[2] * 1920 * box[3] * 1080))


def make_tile(img, boxes, target, typical, rng):
    """One 640 tile and its YOLO label lines, ball scaled to `target` px."""
    size = ball_px_1920(boxes[0]) if boxes else typical
    s = target / size
    w, h = max(1, int(round(1920 * s))), max(1, int(round(1080 * s)))
    frame = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    tile = np.full((TILE, TILE, 3), 114, dtype=np.uint8)
    if boxes:
        cx, cy = boxes[0][0] * w, boxes[0][1] * h
        x0 = int(np.clip(cx - rng.uniform(32, TILE - 32), 0, max(0, w - TILE))) if w > TILE else 0
        y0 = int(np.clip(cy - rng.uniform(32, TILE - 32), 0, max(0, h - TILE))) if h > TILE else 0
    else:
        x0 = rng.randint(0, max(0, w - TILE))
        y0 = rng.randint(0, max(0, h - TILE))
    crop = frame[y0:y0 + TILE, x0:x0 + TILE]
    ox = rng.randint(0, TILE - crop.shape[1])
    oy = rng.randint(0, TILE - crop.shape[0])
    tile[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    lines = []
    for bx, by, bw, bh in boxes:
        px, py = bx * w - x0 + ox, by * h - y0 + oy
        side = max(ball_px_1920((bx, by, bw, bh)) * s, 2.0)
        if ox <= px < ox + crop.shape[1] and oy <= py < oy + crop.shape[0]:
            lines.append("0 {:.6f} {:.6f} {:.6f} {:.6f}".format(
                px / TILE, py / TILE, side / TILE, side / TILE))
    return tile, lines


def main():
    rng = random.Random(SEED)
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    counts = {}
    for split, name in SPLITS.items():
        os.makedirs(os.path.join(OUT, "images", name))
        os.makedirs(os.path.join(OUT, "labels", name))
        pos, neg = [], []
        for dataset in ("ball", "players"):
            for path in sorted(glob.glob(os.path.join(SRC, dataset, split, "images", "*.jpg"))):
                label = path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
                boxes = read_labels(os.path.splitext(label)[0] + ".txt")
                (pos if boxes else neg).append((path, boxes))
        sizes = [ball_px_1920(b[0]) for _, b in pos]
        typical = float(np.median(sizes))
        rng.shuffle(neg)
        jobs = [(p, b, v) for p, b in pos for v in range(VARIANTS)]
        jobs += [(p, b, 0) for p, b in neg[:int(len(jobs) * NEG_PER_POS)]]
        n_ball = 0
        for k, (path, boxes, v) in enumerate(jobs):
            img = cv2.imread(path)
            tile, lines = make_tile(img, boxes, rng.uniform(*BALL_PX), typical, rng)
            stem = "{}_{:05d}".format(name, k)
            cv2.imwrite(os.path.join(OUT, "images", name, stem + ".jpg"), tile,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(os.path.join(OUT, "labels", name, stem + ".txt"), "w") as handle:
                handle.write("\n".join(lines) + ("\n" if lines else ""))
            n_ball += bool(lines)
        counts[name] = (len(jobs), n_ball)
    with open(os.path.join(OUT, "data.yaml"), "w") as handle:
        handle.write("path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
                     "names:\n  0: ball\n")
    for name, (n, b) in counts.items():
        print("{:5}  {} tiles, {} with a ball".format(name, n, b))
    zpath = os.path.join(ROOT, "external", "tiles.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for path in sorted(glob.glob(os.path.join(OUT, "**", "*"), recursive=True)):
            if os.path.isfile(path):
                z.write(path, os.path.relpath(path, OUT))
    print("wrote", os.path.relpath(zpath, ROOT), "{:.0f} MB".format(os.path.getsize(zpath) / 1e6))


if __name__ == "__main__":
    main()
