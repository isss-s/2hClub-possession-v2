"""Ball detector over chosen spans of the tactical video, only on the tiles where the players are.

    python vision/detect_spans.py --spans spans_pilot.json --weights best.pt --out det/ --device cuda --half
    vision/detect_colab.ipynb runs this same file on a GPU

Each 1920 x 1080 frame is cut into 8 overlapping 640 tiles, upscaled 2x on the GPU (the 1280 input
the detector was trained on); only the tiles a span lists are run (vision/frame_spans.py picks the
tiles with players in them). The top MAX_PER_FRAME candidates down to MIN_CONF are kept, so that
everything downstream can change without running the detector again.

Output, one file per CHUNK spans, so that a stopped run resumes where it was:
    <out>/det_<first span>.npz   frames (n,) and dets (n, MAX_PER_FRAME, 3: x px, y px, conf), NaN-padded
Derived data: never commit.
"""
import argparse
import json
import os
import queue
import threading

import cv2
import numpy as np

TILE = 640
TILE_X = (0, 427, 853, 1280)
TILE_Y = (0, 440)
ORIGINS = [(x0, y0) for y0 in TILE_Y for x0 in TILE_X]
MIN_CONF = 0.01
MERGE_PX = 8
MAX_PER_FRAME = 10
CHUNK = 100


def merge(points):
    """points: list of (x, y, conf). Greedy merge of hits closer than MERGE_PX, best first."""
    kept = []
    for x, y, conf in sorted(points, key=lambda p: -p[2]):
        if all(np.hypot(x - kx, y - ky) >= MERGE_PX for kx, ky, _ in kept):
            kept.append((x, y, conf))
    return kept[:MAX_PER_FRAME]


def read_spans(video, spans, out_queue):
    """Decode every frame of `spans`, in order, onto `out_queue` as (span index, frame, image)."""
    cap = cv2.VideoCapture(video)
    pos = None
    for k, span in enumerate(spans):
        if pos != span["start"]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, span["start"])
        for frame in range(span["start"], span["end"]):
            ok, img = cap.read()
            if not ok:
                break
            out_queue.put((k, frame, img))
        pos = span["end"]
    out_queue.put(None)


def tiles_on_device(frames, origins, device, imgsz):
    """BGR uint8 frames -> (len(frames) * len(origins), 3, imgsz, imgsz) RGB in 0-1, frame-major."""
    import torch
    import torch.nn.functional as F
    batch = torch.from_numpy(np.stack(frames)).to(device)
    batch = batch[..., [2, 1, 0]].permute(0, 3, 1, 2).float().div_(255)
    tiles = torch.stack([batch[:, :, y0:y0 + TILE, x0:x0 + TILE] for x0, y0 in origins], 1)
    tiles = tiles.reshape(-1, 3, TILE, TILE)
    if imgsz != TILE:
        # Bilinear with half-pixel centres: what cv2.INTER_LINEAR in the training letterbox does.
        tiles = F.interpolate(tiles, size=(imgsz, imgsz), mode="bilinear", align_corners=False)
    return tiles.contiguous()


def detect_batch(model, frames, origins, device, imgsz, half):
    results = model.predict(tiles_on_device(frames, origins, device, imgsz), imgsz=imgsz,
                            conf=MIN_CONF, half=half, device=device, max_det=MAX_PER_FRAME,
                            verbose=False)
    scale = TILE / imgsz
    out = []
    for f in range(len(frames)):
        points = []
        for k, (x0, y0) in enumerate(origins):
            boxes = results[f * len(origins) + k].boxes
            for (cx, cy, _, _), conf in zip(boxes.xywh.cpu().numpy(), boxes.conf.cpu().numpy()):
                points.append((float(cx * scale + x0), float(cy * scale + y0), float(conf)))
        out.append(merge(points))
    return out


def run(model, video, spans, device, imgsz, half, batch_frames):
    """Detections for every frame of `spans`: frames (n,), dets (n, MAX_PER_FRAME, 3)."""
    n = sum(s["end"] - s["start"] for s in spans)
    frames = np.zeros(n, dtype=np.int64)
    dets = np.full((n, MAX_PER_FRAME, 3), np.nan, dtype=np.float32)
    q = queue.Queue(maxsize=batch_frames * 4)
    threading.Thread(target=read_spans, args=(video, spans, q), daemon=True).start()
    row, pending, done = 0, [], False
    while not done:
        item = q.get()
        done = item is None
        flush = pending and (done or len(pending) == batch_frames or item[0] != pending[0][0])
        if flush:
            origins = [ORIGINS[t] for t in spans[pending[0][0]]["tiles"]]
            found = detect_batch(model, [img for _, _, img in pending], origins, device, imgsz, half)
            for (_, frame, _), points in zip(pending, found):
                frames[row] = frame
                for j, point in enumerate(points):
                    dets[row, j] = point
                row += 1
            pending = []
        if not done:
            pending.append(item)
    return frames[:row], dets[:row]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spans", required=True, help="JSON from vision/frame_spans.py")
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True, help="folder for det_<k>.npz")
    parser.add_argument("--imgsz", type=int, default=1280, help="must match training")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--batch-frames", type=int, default=4)
    args = parser.parse_args()

    from ultralytics import YOLO
    with open(args.spans) as handle:
        spans = json.load(handle)
    model = YOLO(args.weights)
    os.makedirs(args.out, exist_ok=True)
    for first in range(0, len(spans), CHUNK):
        path = os.path.join(args.out, "det_{:05d}.npz".format(first))
        if os.path.exists(path):
            continue
        frames, dets = run(model, args.video, spans[first:first + CHUNK], args.device,
                           args.imgsz, args.half, args.batch_frames)
        np.savez_compressed(path + ".tmp.npz", frames=frames, dets=dets)
        os.replace(path + ".tmp.npz", path)
        print("wrote", path, len(frames), "frames", flush=True)


if __name__ == "__main__":
    main()
