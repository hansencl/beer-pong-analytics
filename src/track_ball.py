"""
=============================================================================
 Beer Pong ball tracker — YOLO detection + trajectory + hit/miss
=============================================================================
Runs an Ultralytics YOLO model on every frame, keeps the "sports ball"
detection (COCO class 32) that best continues the current flight, and writes:

    * an annotated video   (trail, cups, live HIT / MISS banner)
    * a per-frame CSV      (frame, t, x, y, conf, source, shot)
    * a per-shot CSV       (launch angle, apex, flight time, outcome, ...)

A small, fast ball is hard for a generic detector, so an optional HSV colour
fallback fills frames where YOLO finds nothing.

Usage
-----
    # 1) click the cup centres once (saved to JSON, reused afterwards)
    python -m src.track_ball -i data/throw.mp4 --select-cups data/cups.json

    # 2) track + classify
    python -m src.track_ball -i data/throw.mp4 -o data/throw_tracked.mp4 \
        --cups data/cups.json --csv data/trajectory.csv

    # colour-only mode (no neural net, very fast)
    python -m src.track_ball -i data/throw.mp4 --detector hsv
=============================================================================
"""

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:  # works both as `python -m src.track_ball` and `python src/track_ball.py`
    from .trajectory import (Cup, classify_shot, load_cups, save_cups,
                             split_shots, summarize)
except ImportError:
    from trajectory import (Cup, classify_shot, load_cups, save_cups,
                            split_shots, summarize)

SPORTS_BALL = 32  # COCO class id

# Default HSV range: a white / pale ping-pong ball. Tune for your footage
# (src/beerpong_science.py --calibrate has an interactive picker).
HSV_LOW = (0, 0, 190)
HSV_HIGH = (179, 70, 255)

TRAIL_COLOR = (0, 255, 255)
CUP_COLOR = (60, 60, 255)
HIT_COLOR = (80, 255, 80)
MISS_COLOR = (60, 60, 255)


# =============================================================================
# Detectors — each returns a list of (x, y, r, conf)
# =============================================================================

class YoloDetector:
    def __init__(self, weights: str, conf: float, imgsz: int, device=None):
        from ultralytics import YOLO  # imported lazily: heavy dependency
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def __call__(self, frame):
        res = self.model.predict(frame, classes=[SPORTS_BALL], conf=self.conf,
                                 imgsz=self.imgsz, device=self.device,
                                 verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(),
                                       res.boxes.conf.cpu().numpy()):
            out.append(((x1 + x2) / 2, (y1 + y2) / 2,
                        max(x2 - x1, y2 - y1) / 2, float(c)))
        return out


class HsvDetector:
    def __init__(self, low=HSV_LOW, high=HSV_HIGH, min_area=20,
                 max_area=3000, min_circularity=0.5):
        self.low, self.high = np.array(low), np.array(high)
        self.min_area, self.max_area = min_area, max_area
        self.min_circ = min_circularity
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def __call__(self, frame):
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.low, self.high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if not self.min_area <= area <= self.max_area:
                continue
            perim = cv2.arcLength(c, True)
            circ = 4 * math.pi * area / (perim * perim) if perim else 0
            if circ < self.min_circ:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            out.append((x, y, r, 0.3 * circ))  # low pseudo-confidence
        return out


# =============================================================================
# Association: pick the detection that best continues the flight
# =============================================================================

class BallTrack:
    """Constant-velocity gating so we follow ONE ball, not every round blob."""

    def __init__(self, gate_px: float = 120, max_gap: int = 8):
        self.gate = gate_px
        self.max_gap = max_gap
        self.last = None   # (frame, x, y)
        self.vel = (0.0, 0.0)

    def predict(self, frame_idx):
        if self.last is None:
            return None
        f, x, y = self.last
        dt = frame_idx - f
        if dt > self.max_gap:
            self.last, self.vel = None, (0.0, 0.0)
            return None
        return x + self.vel[0] * dt, y + self.vel[1] * dt

    def choose(self, frame_idx, detections):
        if not detections:
            return None
        pred = self.predict(frame_idx)
        if pred is None:
            best = max(detections, key=lambda d: d[3])
        else:
            def cost(d):
                return math.hypot(d[0] - pred[0], d[1] - pred[1]) - 50 * d[3]
            best = min(detections, key=cost)
            if math.hypot(best[0] - pred[0], best[1] - pred[1]) > self.gate:
                return None
        if self.last is not None:
            f, x, y = self.last
            dt = max(1, frame_idx - f)
            self.vel = ((best[0] - x) / dt, (best[1] - y) / dt)
        self.last = (frame_idx, best[0], best[1])
        return best


# =============================================================================
# Cup selection UI
# =============================================================================

def select_cups(video_path: str, out_json: str, default_r: int = 25) -> None:
    """Click cup centres on the first frame. +/- change radius, u undo,
    ENTER/s save, ESC cancel."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read {video_path}")

    cups: list[Cup] = []
    state = {"r": default_r}
    win = "select cups"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            cups.append(Cup(id=len(cups), x=x, y=y, r=state["r"]))

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        view = frame.copy()
        for c in cups:
            cv2.circle(view, (int(c.x), int(c.y)), int(c.r), CUP_COLOR, 2)
            cv2.putText(view, str(c.id), (int(c.x) - 6, int(c.y) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, CUP_COLOR, 1)
        cv2.putText(view, f"click cups | r={state['r']} (+/-) | u=undo | "
                          "ENTER=save | ESC=cancel", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(win, view)
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("+"), ord("=")):
            state["r"] += 2
            for c in cups:
                c.r = state["r"]
        elif k == ord("-"):
            state["r"] = max(4, state["r"] - 2)
            for c in cups:
                c.r = state["r"]
        elif k == ord("u") and cups:
            cups.pop()
        elif k in (13, 10, ord("s")):
            save_cups(cups, out_json)
            print(f"[info] saved {len(cups)} cups -> {out_json}")
            break
        elif k == 27:
            print("[info] cancelled")
            break
    cv2.destroyAllWindows()


# =============================================================================
# Main loop
# =============================================================================

def draw_overlay(frame, trail, cups, banner):
    for c in cups:
        cv2.circle(frame, (int(c.x), int(c.y)), int(c.r), CUP_COLOR, 2)
    pts = list(trail)
    for i in range(1, len(pts)):
        thick = 1 + int(4 * i / len(pts))
        cv2.line(frame, pts[i - 1], pts[i], TRAIL_COLOR, thick, cv2.LINE_AA)
    if pts:
        cv2.circle(frame, pts[-1], 8, TRAIL_COLOR, 2, cv2.LINE_AA)
    if banner:
        text, color = banner
        cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6,
                    (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6,
                    color, 2, cv2.LINE_AA)
    return frame


def run(args) -> None:
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cups = load_cups(args.cups) if args.cups else []
    yolo = (YoloDetector(args.model, args.conf, args.imgsz, args.device)
            if args.detector in ("yolo", "both") else None)
    hsv = HsvDetector() if args.detector in ("hsv", "both") else None
    track = BallTrack(gate_px=args.gate, max_gap=args.max_gap)

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))

    rows, trail = [], deque(maxlen=args.trail)
    current = []          # detections of the shot in progress
    banner, banner_until = None, -1
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets, source = (yolo(frame), "yolo") if yolo else ([], "")
        if not dets and hsv:
            dets, source = hsv(frame), "hsv"
        ball = track.choose(idx, dets)

        if ball is not None:
            x, y, r, c = ball
            rows.append(dict(frame=idx, t=idx / fps, x=x, y=y, r=r,
                             conf=c, source=source))
            current.append(rows[-1])
            trail.append((int(x), int(y)))
        elif current and idx - current[-1]["frame"] > args.max_gap:
            # flight ended -> live verdict for the overlay
            if cups and len(current) >= args.min_points:
                outcome, cup_id, _ = classify_shot(pd.DataFrame(current), cups,
                                                   args.tolerance)
                label = f"HIT! cup {cup_id}" if outcome == "hit" else "MISS"
                banner = (label, HIT_COLOR if outcome == "hit" else MISS_COLOR)
                banner_until = idx + int(fps * 1.5)
            current, trail = [], deque(maxlen=args.trail)

        if idx > banner_until:
            banner = None
        draw_overlay(frame, trail, cups, banner)
        if writer:
            writer.write(frame)
        if args.preview:
            cv2.imshow("beer pong tracker", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
        idx += 1
        if idx % 50 == 0:
            print(f"\r[info] frame {idx}/{total}  detections={len(rows)}",
                  end="", flush=True)

    print()
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    df = pd.DataFrame(rows, columns=["frame", "t", "x", "y", "r", "conf",
                                     "source"])
    df = split_shots(df, max_gap=args.max_gap, min_points=args.min_points)
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"[info] trajectory -> {args.csv} ({len(df)} points)")

    summary = summarize(df, cups or None, args.px_per_meter, args.tolerance)
    if not summary.empty:
        if args.csv:
            shots_csv = str(Path(args.csv).with_name(
                Path(args.csv).stem + "_shots.csv"))
            summary.to_csv(shots_csv, index=False)
            print(f"[info] shot summary -> {shots_csv}")
        with pd.option_context("display.max_columns", None,
                               "display.width", 160):
            print(summary.round(2).to_string(index=False))
        if "outcome" in summary:
            hits = (summary["outcome"] == "hit").sum()
            print(f"[result] {hits}/{len(summary)} hits "
                  f"({100 * hits / len(summary):.0f}%)")
    else:
        print("[warn] no complete shots found — try --detector both, a lower "
              "--conf, or tune HSV_LOW/HSV_HIGH.")
    if args.output:
        print(f"[info] annotated video -> {args.output}")


def main():
    ap = argparse.ArgumentParser(description="Beer pong ball tracking + "
                                             "hit/miss detection")
    ap.add_argument("-i", "--input", required=True, help="input video")
    ap.add_argument("-o", "--output", help="annotated output video (.mp4)")
    ap.add_argument("--csv", default="data/trajectory.csv",
                    help="per-frame trajectory CSV (a *_shots.csv is written "
                         "next to it)")
    ap.add_argument("--cups", help="cups JSON (create with --select-cups)")
    ap.add_argument("--select-cups", metavar="JSON",
                    help="click cup positions on frame 0, save to JSON, exit")
    ap.add_argument("--detector", choices=["yolo", "hsv", "both"],
                    default="both", help="yolo, colour mask, or yolo with "
                                         "colour fallback (default)")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="Ultralytics weights (downloaded on first use)")
    ap.add_argument("--conf", type=float, default=0.15,
                    help="YOLO confidence threshold (balls are small: keep low)")
    ap.add_argument("--imgsz", type=int, default=960,
                    help="YOLO inference size; larger helps small balls")
    ap.add_argument("--device", default=None, help="e.g. cpu, 0, mps")
    ap.add_argument("--gate", type=float, default=120,
                    help="max px jump from the predicted position")
    ap.add_argument("--max-gap", type=int, default=8,
                    help="frames without the ball before a shot ends")
    ap.add_argument("--min-points", type=int, default=5,
                    help="min detections for a valid shot")
    ap.add_argument("--tolerance", type=float, default=1.2,
                    help="hit if landing is within tolerance * cup radius")
    ap.add_argument("--px-per-meter", type=float, default=None,
                    help="scale for km/h launch speed (optional)")
    ap.add_argument("--trail", type=int, default=40, help="trail length")
    ap.add_argument("--preview", action="store_true", help="live window")
    args = ap.parse_args()

    if args.select_cups:
        select_cups(args.input, args.select_cups)
        return
    run(args)


if __name__ == "__main__":
    main()
