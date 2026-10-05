"""
Trajectory analysis for beer pong throws.

Pure NumPy / pandas — no OpenCV required — so it can be imported from the
notebook or unit tests without a video.

A trajectory is a DataFrame with (at least) the columns:

    frame : int     video frame index
    t     : float   time in seconds
    x, y  : float   ball centre in pixels (image coords, y grows DOWN)
    shot  : int     shot id (one contiguous flight); -1 = not assigned

Main entry points
-----------------
    split_shots(df, max_gap)        -> df with a `shot` column
    shot_metrics(shot_df, ...)      -> dict of per-shot physics metrics
    classify_shot(shot_df, cups)    -> ("hit" | "miss", cup_id | None, landing)
    summarize(df, cups, ...)        -> one row per shot
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Cups
# =============================================================================

@dataclass
class Cup:
    """A cup rim approximated as a circle in image coordinates."""
    id: int
    x: float
    y: float
    r: float

    def contains(self, px: float, py: float, tolerance: float = 1.0) -> bool:
        return math.hypot(px - self.x, py - self.y) <= self.r * tolerance


def load_cups(path: str | Path) -> list[Cup]:
    """Load cups from JSON: [{"x": .., "y": .., "r": ..}, ...]."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Cup(id=c.get("id", i), x=c["x"], y=c["y"], r=c["r"])
            for i, c in enumerate(raw)]


def save_cups(cups: list[Cup], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([c.__dict__ for c in cups], f, indent=2)


# =============================================================================
# Shot segmentation
# =============================================================================

def split_shots(df: pd.DataFrame, max_gap: int = 8,
                min_points: int = 5) -> pd.DataFrame:
    """Assign a `shot` id to each detection.

    A new shot starts whenever the ball is lost for more than `max_gap`
    frames. Shots with fewer than `min_points` detections are marked -1
    (noise / false positives).
    """
    df = df.sort_values("frame").reset_index(drop=True).copy()
    if df.empty:
        df["shot"] = pd.Series(dtype=int)
        return df

    new_shot = df["frame"].diff().fillna(max_gap + 1) > max_gap
    df["shot"] = new_shot.cumsum() - 1

    sizes = df.groupby("shot")["frame"].transform("size")
    df.loc[sizes < min_points, "shot"] = -1

    # Re-number the valid shots 0..N-1
    valid = sorted(s for s in df["shot"].unique() if s >= 0)
    remap = {old: new for new, old in enumerate(valid)}
    df["shot"] = df["shot"].map(lambda s: remap.get(s, -1))
    return df


# =============================================================================
# Physics
# =============================================================================

def add_kinematics(shot: pd.DataFrame, smooth: int = 3) -> pd.DataFrame:
    """Add smoothed position, velocity (px/s) and speed columns."""
    shot = shot.sort_values("frame").copy()
    win = max(1, smooth)
    shot["xs"] = shot["x"].rolling(win, center=True, min_periods=1).mean()
    shot["ys"] = shot["y"].rolling(win, center=True, min_periods=1).mean()
    t = shot["t"].to_numpy()
    if len(shot) >= 2:
        shot["vx"] = np.gradient(shot["xs"].to_numpy(), t)
        shot["vy"] = np.gradient(shot["ys"].to_numpy(), t)
    else:
        shot["vx"] = 0.0
        shot["vy"] = 0.0
    shot["speed"] = np.hypot(shot["vx"], shot["vy"])
    return shot


def fit_parabola(shot: pd.DataFrame) -> np.ndarray | None:
    """Least-squares fit y(t) = a t^2 + b t + c. Returns [a, b, c].

    In image coordinates gravity points DOWN, so a clean ballistic flight
    has a > 0.
    """
    if len(shot) < 3:
        return None
    t = shot["t"].to_numpy() - shot["t"].iloc[0]
    return np.polyfit(t, shot["y"].to_numpy(), 2)


def find_landing(shot: pd.DataFrame) -> tuple[float, float, int]:
    """Estimate where the ball came down.

    Heuristic: the first point *after the apex* where the ball stops falling
    (vertical velocity flips from down to up => bounce, or the track simply
    ends => ball dropped into a cup / out of frame). Returns (x, y, frame).
    """
    k = add_kinematics(shot)
    ys = k["ys"].to_numpy()
    vy = k["vy"].to_numpy()
    apex = int(np.argmin(ys))  # smallest y == highest point on screen

    for i in range(apex + 1, len(k)):
        if vy[i - 1] > 0 and vy[i] <= 0:  # was falling, now rising/stopped
            row = k.iloc[i]
            return float(row["xs"]), float(row["ys"]), int(row["frame"])

    row = k.iloc[-1]
    return float(row["xs"]), float(row["ys"]), int(row["frame"])


def shot_metrics(shot: pd.DataFrame, px_per_meter: float | None = None,
                 launch_frames: int = 4) -> dict:
    """Per-shot metrics: launch angle/speed, apex, flight time, fit quality."""
    k = add_kinematics(shot)
    n = min(launch_frames, len(k))
    vx0 = k["vx"].iloc[:n].mean()
    vy0 = k["vy"].iloc[:n].mean()
    # flip y so a throw going up the screen has a positive angle
    launch_angle = math.degrees(math.atan2(-vy0, abs(vx0))) if n else float("nan")
    launch_speed_px = math.hypot(vx0, vy0) if n else float("nan")

    apex_idx = int(k["ys"].idxmin())
    coeffs = fit_parabola(shot)
    if coeffs is not None:
        t = shot["t"].to_numpy() - shot["t"].iloc[0]
        resid = shot["y"].to_numpy() - np.polyval(coeffs, t)
        rmse = float(np.sqrt(np.mean(resid ** 2)))
    else:
        rmse = float("nan")

    out = {
        "n_points": len(shot),
        "start_frame": int(shot["frame"].iloc[0]),
        "end_frame": int(shot["frame"].iloc[-1]),
        "flight_time_s": float(shot["t"].iloc[-1] - shot["t"].iloc[0]),
        "launch_angle_deg": launch_angle,
        "launch_speed_px_s": launch_speed_px,
        "apex_x": float(k.loc[apex_idx, "xs"]),
        "apex_y": float(k.loc[apex_idx, "ys"]),
        "parabola_a": float(coeffs[0]) if coeffs is not None else float("nan"),
        "parabola_rmse_px": rmse,
    }
    if px_per_meter:
        out["launch_speed_kmh"] = launch_speed_px / px_per_meter * 3.6
    return out


# =============================================================================
# Hit / miss
# =============================================================================

def classify_shot(shot: pd.DataFrame, cups: list[Cup],
                  tolerance: float = 1.2):
    """Return ("hit"|"miss", cup_id or None, (x, y, frame) landing)."""
    lx, ly, lf = find_landing(shot)
    best, best_d = None, float("inf")
    for cup in cups:
        d = math.hypot(lx - cup.x, ly - cup.y)
        if cup.contains(lx, ly, tolerance) and d < best_d:
            best, best_d = cup, d
    if best is None:
        return "miss", None, (lx, ly, lf)
    return "hit", best.id, (lx, ly, lf)


def summarize(df: pd.DataFrame, cups: list[Cup] | None = None,
              px_per_meter: float | None = None,
              tolerance: float = 1.2) -> pd.DataFrame:
    """One row per shot with metrics and (if cups are given) the outcome."""
    rows = []
    for shot_id, shot in df[df["shot"] >= 0].groupby("shot"):
        m = {"shot": shot_id, **shot_metrics(shot, px_per_meter)}
        if cups:
            outcome, cup_id, (lx, ly, lf) = classify_shot(shot, cups, tolerance)
            m.update(outcome=outcome, cup=cup_id,
                     landing_x=lx, landing_y=ly, landing_frame=lf)
        rows.append(m)
    return pd.DataFrame(rows)


# =============================================================================
# Synthetic data (for the notebook / tests when you have no video yet)
# =============================================================================

def synthetic_throws(n_shots: int = 6, fps: float = 60.0, seed: int = 7,
                     cups: list[Cup] | None = None) -> pd.DataFrame:
    """Generate noisy ballistic throws aimed at `cups` (some hit, some miss)."""
    rng = np.random.default_rng(seed)
    if cups is None:
        cups = default_cup_rack()
    rows, frame = [], 0
    g = 2200.0  # px/s^2 — roughly 9.81 m/s^2 at ~225 px/m
    for s in range(n_shots):
        x0, y0 = 150 + rng.normal(0, 10), 520 + rng.normal(0, 10)
        target = cups[rng.integers(len(cups))]
        miss = rng.random() < 0.4
        tx = target.x + (rng.choice([-1, 1]) * rng.uniform(40, 90) if miss
                         else rng.normal(0, target.r * 0.3))
        ty = target.y
        T = rng.uniform(0.75, 0.95)
        vx = (tx - x0) / T
        vy = (ty - y0 - 0.5 * g * T ** 2) / T
        for i in range(int(T * fps) + 1):
            t = i / fps
            rows.append({
                "frame": frame + i,
                "t": (frame + i) / fps,
                "x": x0 + vx * t + rng.normal(0, 1.5),
                "y": y0 + vy * t + 0.5 * g * t ** 2 + rng.normal(0, 1.5),
                "conf": float(rng.uniform(0.4, 0.95)),
                "source": "synthetic",
            })
        frame += int(T * fps) + int(fps)  # 1 s pause between throws
    return pd.DataFrame(rows)


def default_cup_rack(x: float = 1050, y: float = 600, r: float = 22) -> list[Cup]:
    """A 10-cup triangle rack as seen from the side-ish camera (demo only)."""
    cups, i = [], 0
    for row in range(4):
        for col in range(4 - row):
            cups.append(Cup(id=i, x=x + row * 2 * r * 0.87,
                            y=y - (4 - row - 1) * r + col * 2 * r, r=r))
            i += 1
    return cups
