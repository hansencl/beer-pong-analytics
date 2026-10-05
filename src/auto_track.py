"""
=============================================================================
 Automatic ball tracking — no clicking, no colour calibration
=============================================================================
The fully automatic counterpart to the manual keyframe mode in
beerpong_science.py (--keyframes). Instead of *you* telling the program where
the ball is, it uses one physical fact: a thrown ball is the only small thing
in the scene that moves along a parabola.

Pipeline
--------
1. MOTION CANDIDATES  background subtraction (MOG2) AND frame differencing
                      -> small, roughly round moving blobs in every frame
                      (optionally + YOLO "sports ball" detections)
2. TRACKLETS          link candidates frame-to-frame with a constant-velocity
                      prediction; many short tracklets (hands, cups, noise)
3. PICK THE THROW     score every tracklet by length, distance travelled and
                      how well it fits y = a t^2 + b t + c with gravity DOWN
4. GROW BY PHYSICS    RANSAC-style: refit the parabola and pull in any
                      candidate from ANY frame that lies on it — recovers the
                      ball where tracklets broke (blur, occlusion by the hand)
5. POSITIONS          measured points + gap interpolation -> {frame: (x, y)}
                      (same format as the manual keyframe mode)

The flight ends where the single parabola stops explaining the data, i.e. at
the first bounce or when the ball drops into a cup — exactly the impact point
the HUD and hit/miss logic need.

Usage
-----
    # standalone: CSV (compatible with src/trajectory.py) + debug video
    python -m src.auto_track -i data/throw.mp4 --csv data/auto_trajectory.csv \
        --debug data/auto_debug.mp4

    # inside the HUD renderer
    python src/beerpong_science.py -i data/throw.mp4 -o data/science.mp4 --auto
=============================================================================
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# CONFIG — defaults are relative to frame size, so they rarely need touching
# =============================================================================

MIN_AREA_FRAC = 0.000008    # smallest blob, as a fraction of the frame area
MAX_AREA_FRAC = 0.004       # largest blob (rejects people / arms)
MIN_CIRCULARITY = 0.35      # motion blur stretches the ball -> stay permissive
MAX_ASPECT = 3.5            # max bbox elongation (blurred ball is elongated)
WARMUP_FRAMES = 5           # frames for the background model to settle
SHAKE_FACTOR = 8            # a frame with > SHAKE_FACTOR x median candidates
SHAKE_MIN = 30              # (and > SHAKE_MIN) is camera shake -> skipped

MAX_GAP = 4                 # frames a tracklet may coast without a detection
MIN_TRACK_LEN = 5           # shortest tracklet / flight considered at all
SEED_LEN = 5                # points per RANSAC seed window
SEED_MIN_SPAN = 0.03        # seed must move this far (fraction of diag)
INLIER_TOL_FRAC = 0.012     # parabola inlier distance, fraction of frame diag
GROW_ITERS = 4              # physics-growing iterations


@dataclass
class Candidate:
    frame: int
    x: float
    y: float
    r: float
    score: float = 0.5
    source: str = "motion"


@dataclass
class Tracklet:
    pts: list = field(default_factory=list)   # list[Candidate]

    @property
    def last(self):
        return self.pts[-1]

    def predict(self, frame):
        p = self.last
        if len(self.pts) >= 2:
            q = self.pts[-2]
            dt = max(1, p.frame - q.frame)
            vx, vy = (p.x - q.x) / dt, (p.y - q.y) / dt
        else:
            vx = vy = 0.0
        d = frame - p.frame
        return p.x + vx * d, p.y + vy * d, math.hypot(vx, vy)


# =============================================================================
# 1. Motion candidates
# =============================================================================

class MotionDetector:
    def __init__(self, w, h):
        area = w * h
        self.min_area = max(4.0, MIN_AREA_FRAC * area)
        self.max_area = MAX_AREA_FRAC * area
        self.bg = cv2.createBackgroundSubtractorMOG2(history=120,
                                                     varThreshold=24,
                                                     detectShadows=False)
        self.prev = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def __call__(self, frame):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                (5, 5), 0)
        fg = self.bg.apply(frame)
        if self.prev is None:
            self.prev = gray
            return [], fg
        diff = cv2.absdiff(gray, self.prev)
        self.prev = gray
        _, diff = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        # A pixel must be "new" w.r.t. the background AND changing right now:
        # kills slow drifts (lighting) and static clutter at the same time.
        mask = cv2.bitwise_and(fg, diff)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.dilate(mask, self.kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if not self.min_area <= area <= self.max_area:
                continue
            perim = cv2.arcLength(c, True)
            circ = 4 * math.pi * area / (perim * perim) if perim else 0.0
            _, _, bw, bh = cv2.boundingRect(c)
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if circ < MIN_CIRCULARITY or aspect > MAX_ASPECT:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            out.append((x, y, r, circ))
        return out, mask


def collect_candidates(video_path, use_yolo=False, yolo_model="yolov8n.pt",
                       yolo_conf=0.1, progress=True):
    """Pass over the video once. Returns (candidates_by_frame, meta)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    motion = MotionDetector(w, h)
    yolo = None
    if use_yolo:
        from ultralytics import YOLO
        yolo = YOLO(yolo_model)

    cands, idx = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        found = []
        blobs, _ = motion(frame)
        if idx >= WARMUP_FRAMES:
            found += [Candidate(idx, x, y, r, 0.5 * circ, "motion")
                      for x, y, r, circ in blobs]
        if yolo is not None:
            res = yolo.predict(frame, classes=[32], conf=yolo_conf,
                               verbose=False)[0]
            for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(),
                                           res.boxes.conf.cpu().numpy()):
                found.append(Candidate(idx, (x1 + x2) / 2, (y1 + y2) / 2,
                                       max(x2 - x1, y2 - y1) / 2,
                                       0.5 + float(c), "yolo"))
        cands[idx] = found
        idx += 1
        if progress and idx % 50 == 0:
            print(f"\r[auto] scanning frame {idx}/{total}", end="", flush=True)
    cap.release()

    # Camera shake makes the WHOLE image move: hundreds of blobs that line up
    # into fake "flights". Such frames carry no usable ball signal -> drop them.
    counts = [len(v) for v in cands.values()] or [0]
    limit = max(SHAKE_MIN, SHAKE_FACTOR * float(np.median(counts)))
    shaky = [f for f, v in cands.items() if len(v) > limit]
    for f in shaky:
        cands[f] = []
    if progress:
        print(f"\r[auto] scanned {idx} frames, "
              f"{sum(map(len, cands.values()))} motion candidates"
              + (f", {len(shaky)} shaky frames skipped" if shaky else ""))
    return cands, dict(fps=fps, w=w, h=h, n_frames=idx)


# =============================================================================
# 2. Tracklets
# =============================================================================

def build_tracklets(cands, diag):
    base_gate = 0.04 * diag
    active, done = [], []
    for f in sorted(cands):
        dets = list(cands[f])
        # retire tracklets that have coasted too long
        still = []
        for t in active:
            (done if f - t.last.frame > MAX_GAP else still).append(t)
        active = still

        # greedy nearest assignment, cheapest pairs first
        pairs = []
        for ti, t in enumerate(active):
            px, py, speed = t.predict(f)
            gate = base_gate + 0.6 * speed * (f - t.last.frame)
            for di, d in enumerate(dets):
                dist = math.hypot(d.x - px, d.y - py)
                if dist <= gate:
                    pairs.append((dist, ti, di))
        used_t, used_d = set(), set()
        for _, ti, di in sorted(pairs):
            if ti in used_t or di in used_d:
                continue
            active[ti].pts.append(dets[di])
            used_t.add(ti)
            used_d.add(di)
        for di, d in enumerate(dets):
            if di not in used_d:
                active.append(Tracklet([d]))
    return [t for t in done + active if len(t.pts) >= MIN_TRACK_LEN]


# =============================================================================
# 3 + 4. Physics: pick the throw and grow it
# =============================================================================

def fit_motion(pts):
    """x(f) linear, y(f) quadratic. Returns (px, py) polynomials or None."""
    if len(pts) < 4:
        return None
    f = np.array([p.frame for p in pts], float)
    x = np.array([p.x for p in pts])
    y = np.array([p.y for p in pts])
    return np.polyfit(f, x, 1), np.polyfit(f, y, 2)


def residuals(model, pts):
    px, py = model
    f = np.array([p.frame for p in pts], float)
    return np.hypot(np.array([p.x for p in pts]) - np.polyval(px, f),
                    np.array([p.y for p in pts]) - np.polyval(py, f))


def seed_windows(tracklets, diag, length=SEED_LEN, stride=2):
    """Short, clean, clearly-moving pieces of tracklets.

    Whole tracklets are often polluted (the ball tracklet hops onto the arm
    after the ball leaves the frame), so we seed from small windows instead
    and let physics decide how far each one really extends.
    """
    tol = INLIER_TOL_FRAC * diag
    for t in tracklets:
        for i in range(0, len(t.pts) - length + 1, stride):
            win = t.pts[i:i + length]
            span = math.hypot(win[-1].x - win[0].x, win[-1].y - win[0].y)
            if span < SEED_MIN_SPAN * diag:      # jitter, not a flight
                continue
            model = fit_motion(win)
            if np.sqrt(np.mean(residuals(model, win) ** 2)) > 0.5 * tol:
                continue
            yield win


def score_flight(pts, diag):
    """Many inliers, along a long path, bending the way gravity bends."""
    if len(pts) < MIN_TRACK_LEN:
        return -1.0
    model = fit_motion(pts)
    a = model[1][0]                      # y curvature; gravity -> a > 0
    if a <= 0:                           # curving UP on screen: not a throw
        return -1.0
    if np.polyval(np.polyder(model[1]), pts[-1].frame) <= 0:
        return -1.0                      # still rising at the end: a wind-up
                                         # / hand lift, not a landed throw
    f = np.arange(pts[0].frame, pts[-1].frame + 1)
    curve = np.column_stack([np.polyval(model[0], f), np.polyval(model[1], f)])
    path = float(np.sum(np.hypot(*np.diff(curve, axis=0).T)))
    rmse = float(np.sqrt(np.mean(residuals(model, pts) ** 2)))
    coverage = len(pts) / len(f)         # fraction of frames with a detection
    return len(pts) * (path / diag) * coverage / (1.0 + rmse / (0.005 * diag))


def grow_flight(seed_pts, cands, diag, n_frames):
    """Pull in candidates from all frames that lie on the fitted parabola.

    Grows outward from the seed only while frames keep supporting the model,
    so it stops at the bounce / cup instead of jumping to unrelated motion.
    """
    tol = INLIER_TOL_FRAC * diag
    pts = sorted(seed_pts, key=lambda p: p.frame)
    for _ in range(GROW_ITERS):
        model = fit_motion(pts)
        if model is None:
            break
        px, py = model
        by_frame = {}
        lo, hi = pts[0].frame, pts[-1].frame
        # extend in both directions until MAX_GAP frames in a row lack support
        for direction, start in ((-1, lo), (1, hi)):
            miss, f = 0, start
            while 0 <= f < n_frames and miss <= MAX_GAP:
                ex, ey = np.polyval(px, f), np.polyval(py, f)
                best = None
                for c in cands.get(f, []):
                    d = math.hypot(c.x - ex, c.y - ey)
                    if d <= tol and (best is None or d < best[0]):
                        best = (d, c)
                if best:
                    by_frame[f] = best[1]
                    miss = 0
                elif not (lo <= f <= hi):
                    miss += 1
                f += direction
        # inside the span, re-select the best candidate per frame too
        for f in range(lo, hi + 1):
            ex, ey = np.polyval(px, f), np.polyval(py, f)
            near = [(math.hypot(c.x - ex, c.y - ey), c)
                    for c in cands.get(f, [])]
            near = [n for n in near if n[0] <= tol]
            if near:
                by_frame[f] = min(near, key=lambda n: n[0])[1]
        new_pts = [by_frame[f] for f in sorted(by_frame)]
        if len(new_pts) < 4 or [p.frame for p in new_pts] == \
                [p.frame for p in pts]:
            pts = new_pts if len(new_pts) >= 4 else pts
            break
        pts = new_pts
    return pts


# =============================================================================
# 5. Positions
# =============================================================================

def to_positions(pts, mode="hybrid", model=None):
    """{frame: (x, y)} for every frame of the flight.

    hybrid : measured points, gaps linearly interpolated (default)
    fit    : the smooth parabola everywhere (prettiest, least literal)
    """
    frames = np.array([p.frame for p in pts])
    all_f = np.arange(frames[0], frames[-1] + 1)
    if mode == "fit":
        model = model or fit_motion(pts)
        xs, ys = np.polyval(model[0], all_f), np.polyval(model[1], all_f)
    else:
        xs = np.interp(all_f, frames, [p.x for p in pts])
        ys = np.interp(all_f, frames, [p.y for p in pts])
    return {int(f): (int(round(x)), int(round(y)))
            for f, x, y in zip(all_f, xs, ys)}


def find_flights(cands, tracklets, diag, n_frames, max_flights=1,
                 min_rel_score=0.25):
    """RANSAC over seed windows: grow each, keep the best-scoring flights.

    After a flight is accepted its frames are blocked so the next one is a
    different throw (for videos with several shots).
    """
    grown = []
    seen = set()
    for win in seed_windows(tracklets, diag):
        pts = grow_flight(win, cands, diag, n_frames)
        key = (pts[0].frame, pts[-1].frame, len(pts))
        if key in seen:
            continue
        seen.add(key)
        s = score_flight(pts, diag)
        if s > 0:
            grown.append((s, pts))
    grown.sort(key=lambda g: g[0], reverse=True)

    flights, used = [], set()
    for s, pts in grown:
        if len(flights) >= max_flights:
            break
        if flights and s < min_rel_score * flights[0][0]:
            break
        frames = set(range(pts[0].frame, pts[-1].frame + 1))
        if frames & used:
            continue
        flights.append((s, pts))
        used |= frames
    return [pts for _, pts in sorted(flights, key=lambda f: f[1][0].frame)]


def auto_track(video_path, use_yolo=False, mode="hybrid", max_throws=1,
               verbose=True, **yolo_kw):
    """Run the whole pipeline. Returns a result dict or None if no throw.

    result = {positions: {frame: (x, y)}, radius, points: [Candidate],
              model, flights: [ {positions, radius, points, model} ], fps,
              candidates, tracklets, meta}
    The top-level positions/radius/points/model are the FIRST flight, so a
    single-throw clip can use them directly.
    """
    cands, meta = collect_candidates(video_path, use_yolo, progress=verbose,
                                     **yolo_kw)
    diag = math.hypot(meta["w"], meta["h"])
    tracklets = build_tracklets(cands, diag)
    found = find_flights(cands, tracklets, diag, meta["n_frames"], max_throws)
    if verbose:
        print(f"[auto] {len(tracklets)} tracklets -> {len(found)} throw(s)")
    if not found:
        return None

    flights = []
    for pts in found:
        model = fit_motion(pts)
        radius = max(4, int(np.median([p.r for p in pts])))
        flights.append(dict(positions=to_positions(pts, mode, model),
                            radius=radius, points=pts, model=model))
        if verbose:
            rmse = float(np.sqrt(np.mean(residuals(model, pts) ** 2)))
            print(f"[auto]   frames {pts[0].frame}-{pts[-1].frame}: "
                  f"{len(pts)} detections, fit RMSE {rmse:.1f}px")
    return dict(**flights[0], flights=flights, fps=meta["fps"],
                candidates=cands, tracklets=tracklets, meta=meta)


# =============================================================================
# Outputs
# =============================================================================

def save_csv(result, path):
    import pandas as pd
    rows = []
    for shot, fl in enumerate(result["flights"]):
        measured = {p.frame: p for p in fl["points"]}
        for f, (x, y) in sorted(fl["positions"].items()):
            p = measured.get(f)
            rows.append(dict(frame=f, t=f / result["fps"], x=x, y=y,
                             r=p.r if p else fl["radius"],
                             conf=p.score if p else 0.0,
                             source=p.source if p else "interpolated",
                             shot=shot))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[auto] trajectory -> {path} ({len(rows)} frames, "
          f"{len(result['flights'])} throw(s))")


def write_debug_video(video_path, result, out_path):
    """Grey = every motion candidate, blue = tracklets, yellow = the throw,
    red = parabola fit. The fastest way to see WHY the tracker chose a path."""
    cap = cv2.VideoCapture(video_path)
    m = result["meta"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                          m["fps"], (m["w"], m["h"]))
    curves = []
    for fl in result["flights"]:
        px, py = fl["model"]
        f_lo, f_hi = min(fl["positions"]), max(fl["positions"])
        curves.append(np.array([(np.polyval(px, f), np.polyval(py, f))
                                for f in range(f_lo, f_hi + 1)], np.int32))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for c in result["candidates"].get(idx, []):
            cv2.circle(frame, (int(c.x), int(c.y)), max(3, int(c.r)),
                       (160, 160, 160), 1)
        for t in result["tracklets"]:
            seg = [(int(p.x), int(p.y)) for p in t.pts if p.frame <= idx]
            if len(seg) >= 2 and t.pts[-1].frame >= idx - 15:
                cv2.polylines(frame, [np.array(seg, np.int32)], False,
                              (255, 140, 0), 1)
        for fl, curve in zip(result["flights"], curves):
            pos = fl["positions"]
            f_lo, f_hi = min(pos), max(pos)
            if idx < f_lo:
                continue
            cv2.polylines(frame, [curve], False, (0, 0, 255), 1, cv2.LINE_AA)
            trail = [pos[f] for f in range(f_lo, min(idx, f_hi) + 1)]
            if len(trail) >= 2:
                cv2.polylines(frame, [np.array(trail, np.int32)], False,
                              (0, 255, 255), 2, cv2.LINE_AA)
            if idx in pos:
                cv2.circle(frame, pos[idx], fl["radius"] + 4, (0, 255, 255),
                           2, cv2.LINE_AA)
        cv2.putText(frame, f"frame {idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        out.write(frame)
        idx += 1
    cap.release()
    out.release()
    print(f"[auto] debug video -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Automatic beer pong ball "
                                             "tracking (motion + physics)")
    ap.add_argument("-i", "--input", required=True, help="input video")
    ap.add_argument("--csv", default="data/auto_trajectory.csv",
                    help="output trajectory CSV (src/trajectory.py format)")
    ap.add_argument("--debug", help="write a debug video showing candidates, "
                                    "tracklets and the chosen throw")
    ap.add_argument("--yolo", action="store_true",
                    help="add YOLO 'sports ball' detections as candidates")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO weights")
    ap.add_argument("--mode", choices=["hybrid", "fit"], default="hybrid",
                    help="hybrid = measured + interpolated, fit = parabola")
    ap.add_argument("--max-throws", type=int, default=20,
                    help="max number of throws to extract from the video")
    args = ap.parse_args()

    res = auto_track(args.input, use_yolo=args.yolo, mode=args.mode,
                     max_throws=args.max_throws, yolo_model=args.model)
    if res is None:
        raise SystemExit("[auto] no throw found. Try --yolo, a static camera, "
                         "or the manual --keyframes mode in beerpong_science.py")
    if args.csv:
        save_csv(res, args.csv)
    if args.debug:
        write_debug_video(args.input, res, args.debug)


if __name__ == "__main__":
    main()
