"""
=============================================================================
 BEER PONG: "FOR SCIENCE" — Sci-Fi HUD Overlay Generator
=============================================================================
Turns an ordinary beer pong throw into a mock "scientific experiment" clip
for LinkedIn / Instagram Reels / Shorts.

Pipeline
--------
1. POSE + HAND TRACKING  -> MediaPipe Pose (body skeleton) + Hands (release pt)
2. BALL TRACKING         -> HSV color mask + contour/blob detection
3. TRAJECTORY            -> glowing neon arc trail following the ball
4. LANDING / STATS       -> reticle, splash, and HUD stat callouts
                            (Arc angle, launch speed, "Precision", "p < 0.05")

Usage
-----
    python beerpong_science.py --input throw.mp4 --output science.mp4

    # live preview while rendering
    python beerpong_science.py --input throw.mp4 --output science.mp4 --preview

    # calibrate the ball color mask interactively, then quit
    python beerpong_science.py --input throw.mp4 --calibrate

    # MANUAL: click the ball on a few frames (most accurate)
    python beerpong_science.py --input throw.mp4 --keyframes

    # AUTOMATIC: motion + parabola physics, no clicks (see auto_track.py)
    python beerpong_science.py --input throw.mp4 --auto

Dependencies
------------
    pip install opencv-python mediapipe numpy

Notes for tuning to YOUR footage: every knob you are likely to touch lives in
the CONFIG block below. The ball color range (BALL_HSV_LOW/HIGH) is the single
most important thing to get right — use --calibrate to find it.
=============================================================================
"""

import argparse
import math
from collections import deque

import cv2
import numpy as np

# We need the LEGACY solutions API (mp.solutions.pose / .hands). Some wheels
# ship only the newer `tasks` API and omit `solutions`, so we probe for it.
# NOTE: `solutions` is exposed as a lazy ATTRIBUTE, not an importable submodule
# path, so we must touch mp.solutions.pose (not `import mediapipe.solutions`).
try:
    import mediapipe as mp
    _ = mp.solutions.pose      # probe that the attribute resolves
    _ = mp.solutions.hands     # probe that the attribute resolves
    HAS_MEDIAPIPE = True
except Exception as _mp_err:  # ImportError, AttributeError, ModuleNotFound...
    HAS_MEDIAPIPE = False
    print(f"[warn] MediaPipe 'solutions' API unavailable ({_mp_err}). "
          "Pose/hand skeletons are DISABLED; ball tracking + HUD still run.\n"
          "       To enable skeletons, install a build that includes "
          "'solutions', e.g.:  pip install \"mediapipe==0.10.14\"")


# =============================================================================
# CONFIG — tweak these for your specific video
# =============================================================================

# ---- Ball color detection (HSV) --------------------------------------------
# HSV in OpenCV: H is 0-179, S/V are 0-255. Green sits around H=35-85.
# ACTIVE PRESET: a LIGHT / PALE GREEN ball -> green hue, but LOW saturation
# (that's what makes it look washed-out) and HIGH value/brightness.
# If the trail flickers or grabs the wrong object, tighten these with
# --calibrate. Widen the S/V range if the ball is missed; narrow it if the
# background (walls, table) leaks into the mask.
BALL_HSV_LOW  = np.array([30,  25, 150])   # lower bound: greenish, pale, bright
BALL_HSV_HIGH = np.array([90, 180, 255])   # upper bound

# Preset for a standard ORANGE ball (uncomment to use):
# BALL_HSV_LOW  = np.array([5,  120, 120])
# BALL_HSV_HIGH = np.array([22, 255, 255])

# Preset for a WHITE ball (uncomment to use):
# BALL_HSV_LOW  = np.array([0,   0, 200])
# BALL_HSV_HIGH = np.array([179, 60, 255])

# Ball size gate (in pixels of contour area). Filters out noise + big blobs.
# Increase MIN if small specks get picked up; lower MAX if large objects match.
BALL_MIN_AREA = 30      # smallest blob (px^2) accepted as the ball
BALL_MAX_AREA = 4000    # largest blob accepted as the ball
BALL_MIN_CIRCULARITY = 0.55  # 1.0 == perfect circle; lower is more permissive

# ---- Motion trail ----------------------------------------------------------
TRAIL_LEN = 40          # how many past positions to keep in the glowing arc
TRAIL_COLOR = (0, 255, 255)     # BGR — neon cyan/yellow-green (0,255,255)=yellow
TRAIL_GLOW_COLOR = (255, 255, 0)  # BGR — cyan glow underlay

# ---- Landing detection -----------------------------------------------------
# The ball is considered "landed" when its vertical speed reverses / collapses
# after having been airborne. These control that heuristic.
LANDING_MIN_AIRBORNE_FRAMES = 6   # must fly this long before a landing counts
LANDING_SPEED_DROP = 0.45         # landed if speed falls below this * peak speed

# ---- Visual style ----------------------------------------------------------
HUD_ACCENT   = (0, 255, 180)    # BGR neon mint — primary HUD color
HUD_WARN     = (0, 200, 255)    # BGR amber — secondary/highlight
HUD_DIM      = (90, 90, 90)     # BGR grey — grid / inactive lines
SKELETON_COLOR = (255, 90, 200) # BGR magenta — body wireframe
RELEASE_COLOR  = (0, 255, 255)  # BGR yellow — release-point marker

# Fun fake-science stat shown on the HUD. Keep it tongue-in-cheek.
EXPERIMENT_TITLE = "PROJECTILE STUDY #42"

# ---- Social export (LinkedIn / Instagram / Shorts) -------------------------
# Output canvas sizes (w, h) per format. "reel" = 9:16 vertical (Reels/Shorts),
# "portrait" = 4:5 (IG feed), "square" = 1:1, "source" = keep original size.
SOCIAL_SIZES = {
    "reel":     (1080, 1920),
    "portrait": (1080, 1350),
    "square":   (1080, 1080),
}
# Text baked into the top/bottom bars of the social canvas. Edit freely.
SOCIAL_TITLE    = "BEER PONG // FOR SCIENCE"
SOCIAL_SUBTITLE = "a rigorous, peer-reviewed* experiment"
SOCIAL_FOOTER   = "*results may vary   |   #ForScience #ComputerVision"
SOCIAL_BG = (18, 16, 22)   # BGR dark background behind the video


# =============================================================================
# Small drawing helpers
# =============================================================================

def draw_glow_line(img, p1, p2, color, thickness, glow=9):
    """Draw a line with a soft neon glow (thick blurred underlay + crisp core)."""
    overlay = img.copy()
    cv2.line(overlay, p1, p2, color, thickness + glow, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def draw_reticle(img, center, radius, color, tick=12, thickness=2):
    """Sci-fi targeting reticle: circle + crosshair gaps + corner ticks."""
    cx, cy = int(center[0]), int(center[1])
    cv2.circle(img, (cx, cy), radius, color, thickness, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, radius // 6), color, -1, cv2.LINE_AA)
    # crosshairs with a gap in the middle
    g = radius // 3
    cv2.line(img, (cx - radius - tick, cy), (cx - g, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx + g, cy), (cx + radius + tick, cy), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, cy - radius - tick), (cx, cy - g), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx, cy + g), (cx, cy + radius + tick), color, thickness, cv2.LINE_AA)


def draw_hud_panel(img, top_left, size, alpha=0.35, color=(0, 0, 0)):
    """Semi-transparent dark panel to seat text against (keeps HUD readable)."""
    x, y = top_left
    w, h = size
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), HUD_ACCENT, 1, cv2.LINE_AA)


def put_hud_text(img, text, org, scale=0.6, color=HUD_ACCENT, thickness=1):
    """Monospace-ish HUD text with a subtle shadow for legibility.

    Shadow uses the SAME thickness (offset by 2px), not a thicker stroke: this
    OpenCV 5.0.0 build glitches thick putText strokes into ghosted trailing
    characters, which shows up over bright parts of the video.
    """
    cv2.putText(img, text, (org[0] + 2, org[1] + 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, org,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# Top-view cup rack layout. CUP_ROWS lists cup counts from the WIDEST row
# (base) to the APEX. [3,2,1] = 6-cup, [4,3,2,1] = 10-cup.
CUP_ROWS = [4, 3, 2, 1]

# Which way the triangle POINTS in the top view (where the apex / single cup
# sits): "right", "left", "up", or "down". Set this to match your footage.
CUP_ORIENTATION = "right"

# Force which cup lights up as the hit (1-based), or None to auto-pick from the
# impact position. Cups are numbered from the BASE row to the APEX, in the same
# order as CUP_ROWS. For [4,3,2,1]: #1-4 = base, #5-7 = next, #8-9 = the "2" row
# (2nd from the apex), #10 = apex. So the 2nd-from-right row here is #8 or #9.
HIT_CUP_OVERRIDE = 8


def _cup_centers(x0, y0, w, h):
    """Return cup center points for a top-down triangle in any orientation.

    Cups are first placed in a normalized (depth, lateral) frame -- depth runs
    from the base (0) to the apex (1); lateral is centered within each row and
    uses the SAME spacing across rows so narrower rows nest between the wider
    row behind them -- then rotated according to CUP_ORIENTATION.
    """
    n_rows = len(CUP_ROWS)
    max_count = max(CUP_ROWS)
    step_l = 0.66 / max(1, max_count - 1)   # lateral gap (normalized)

    norm = []  # (depth, lateral) in 0..1
    for r, count in enumerate(CUP_ROWS):
        depth = 0.15 + 0.70 * (r / max(1, n_rows - 1))   # base -> apex
        for c in range(count):
            lateral = 0.5 + (c - (count - 1) / 2.0) * step_l
            norm.append((depth, lateral))

    centers = []
    for d, l in norm:
        if CUP_ORIENTATION == "right":     # apex on the right, base on the left
            cx, cy = x0 + d * w, y0 + l * h
        elif CUP_ORIENTATION == "left":    # apex on the left
            cx, cy = x0 + (1 - d) * w, y0 + l * h
        elif CUP_ORIENTATION == "up":      # apex at the top
            cx, cy = x0 + l * w, y0 + (1 - d) * h
        else:                               # "down": apex at the bottom
            cx, cy = x0 + l * w, y0 + d * h
        centers.append((cx, cy))
    return centers


def draw_top_view_panel(frame, tracker, w, h):
    """Bottom-left mini-map: a top-down cup rack, lighting the cup that was hit.

    The hit cup is chosen by mapping the impact's HORIZONTAL screen position
    onto the rack (a single side camera has no true depth). Tune HIT_X_LO/HI
    below to line the map up with where your cups actually sit in the frame.
    """
    # Panel geometry (bottom-left).
    pw, ph = 190, 190
    px, py = 20, h - ph - 20
    draw_hud_panel(frame, (px, py), (pw, ph))
    put_hud_text(frame, "TOP VIEW - IMPACT", (px + 12, py + 22),
                 scale=0.5, color=HUD_WARN)

    # Rack drawing area inside the panel.
    rx, ry = px + 20, py + 34
    rw, rh = pw - 40, ph - 70
    # size cups from how many sit along each panel axis (depends on orientation)
    n_rows, max_count = len(CUP_ROWS), max(CUP_ROWS)
    horizontal = CUP_ORIENTATION in ("right", "left")
    cols = n_rows if horizontal else max_count       # cups spanning the width
    rows = max_count if horizontal else n_rows        # cups spanning the height
    cup_r = max(3, int(min(rw / cols, rh / rows) * 0.42))
    centers = _cup_centers(rx, ry, rw, rh)

    # Decide which cup was hit.
    hit_idx = None
    if tracker.landing_point is not None:
        if HIT_CUP_OVERRIDE is not None:
            # explicit: you told us exactly which cup it hit
            hit_idx = min(len(centers) - 1, max(0, HIT_CUP_OVERRIDE - 1))
        else:
            # auto: map impact's horizontal position onto the rack.
            # HIT_X_LO/HI: the fraction of frame WIDTH your cup cluster spans.
            HIT_X_LO, HIT_X_HI = 0.30, 0.70
            norm = (tracker.landing_point[0] / max(1, w) - HIT_X_LO) / (HIT_X_HI - HIT_X_LO)
            norm = min(1.0, max(0.0, norm))
            target_x = rx + norm * rw
            hit_idx = min(range(len(centers)),
                          key=lambda i: abs(centers[i][0] - target_x))

    # Draw the cups.
    for i, (cx, cy) in enumerate(centers):
        cx, cy = int(cx), int(cy)
        if i == hit_idx:
            # pulsing filled "hit" cup
            pulse = 0.5 + 0.5 * math.sin(len(tracker.all_points) * 0.4)
            cv2.circle(frame, (cx, cy), cup_r, HUD_WARN, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), int(cup_r + 4 + 4 * pulse),
                       HUD_WARN, 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), max(2, cup_r // 3), (0, 0, 0), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), cup_r, HUD_ACCENT, 1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), max(2, cup_r // 3), HUD_DIM, 1, cv2.LINE_AA)

    # Verdict line under the rack.
    if hit_idx is not None:
        put_hud_text(frame, f"CUP #{hit_idx + 1}  >> SPLASH!",
                     (px + 12, py + ph - 14), scale=0.5, color=HUD_WARN)
    else:
        put_hud_text(frame, "AWAITING IMPACT...",
                     (px + 12, py + ph - 14), scale=0.45, color=HUD_DIM)


def compose_social(frame, canvas_w, canvas_h, fit="fill"):
    """Place the rendered frame on a social canvas (default 9:16).

    fit="fill" : video COVERS the whole canvas (best for vertical footage);
                 title/caption overlay on translucent bands ON the video.
    fit="fit"  : video is letterboxed whole into the canvas with solid title
                 and caption bars (best for horizontal footage).
    """
    fh, fw = frame.shape[:2]

    if fit == "fill":
        # scale to COVER the canvas, then center-crop the overflow
        scale = max(canvas_w / fw, canvas_h / fh)
        new_w, new_h = int(math.ceil(fw * scale)), int(math.ceil(fh * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        x0 = (new_w - canvas_w) // 2
        y0 = (new_h - canvas_h) // 2
        canvas = resized[y0:y0 + canvas_h, x0:x0 + canvas_w].copy()

        band_top = int(canvas_h * 0.12)
        band_bot = int(canvas_h * 0.10)
        # translucent dark bands so text stays readable over the video
        ov = canvas.copy()
        cv2.rectangle(ov, (0, 0), (canvas_w, band_top), (0, 0, 0), -1)
        cv2.rectangle(ov, (0, canvas_h - band_bot), (canvas_w, canvas_h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.45, canvas, 0.55, 0, canvas)
        # neon separators
        cv2.line(canvas, (0, band_top), (canvas_w, band_top), HUD_ACCENT, 2, cv2.LINE_AA)
        cv2.line(canvas, (0, canvas_h - band_bot), (canvas_w, canvas_h - band_bot),
                 HUD_ACCENT, 2, cv2.LINE_AA)

        _put_centered(canvas, SOCIAL_TITLE, int(band_top * 0.45), canvas_w,
                      scale=1.0, color=HUD_ACCENT, thickness=2)
        _put_centered(canvas, SOCIAL_SUBTITLE, int(band_top * 0.78), canvas_w,
                      scale=0.55, color=HUD_WARN, thickness=1)
        _put_centered(canvas, SOCIAL_FOOTER, canvas_h - int(band_bot * 0.40),
                      canvas_w, scale=0.55, color=(210, 210, 210), thickness=1)
        return canvas

    # ---- fit == "fit": letterbox whole video with solid bars ----
    canvas = np.full((canvas_h, canvas_w, 3), SOCIAL_BG, dtype=np.uint8)
    for gx in range(0, canvas_w, 60):
        cv2.line(canvas, (gx, 0), (gx, canvas_h), (28, 26, 34), 1)
    for gy in range(0, canvas_h, 60):
        cv2.line(canvas, (0, gy), (canvas_w, gy), (28, 26, 34), 1)

    header_h = int(canvas_h * 0.11)
    footer_h = int(canvas_h * 0.10)
    avail_h = canvas_h - header_h - footer_h
    scale = min(canvas_w / fw, avail_h / fh)
    new_w, new_h = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    ox = (canvas_w - new_w) // 2
    oy = header_h + (avail_h - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    cv2.rectangle(canvas, (ox - 2, oy - 2), (ox + new_w + 2, oy + new_h + 2),
                  HUD_ACCENT, 2, cv2.LINE_AA)
    _put_centered(canvas, SOCIAL_TITLE, int(header_h * 0.48), canvas_w,
                  scale=1.0, color=HUD_ACCENT, thickness=2)
    _put_centered(canvas, SOCIAL_SUBTITLE, int(header_h * 0.80), canvas_w,
                  scale=0.55, color=HUD_WARN, thickness=1)
    _put_centered(canvas, SOCIAL_FOOTER, canvas_h - int(footer_h * 0.45),
                  canvas_w, scale=0.55, color=(200, 200, 200), thickness=1)
    return canvas


def _put_centered(img, text, y, canvas_w, scale, color, thickness):
    """Draw horizontally-centered text with a clean black outline at pos y.

    The outline is built from thin copies offset in 8 directions rather than a
    single thick stroke: this OpenCV 5.0.0 build glitches thick putText strokes
    (ghosted trailing characters), so every stroke here stays at `thickness`.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, (canvas_w - tw) // 2)
    for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2),
                   (-2, -2), (2, 2), (-2, 2), (2, -2)):
        cv2.putText(img, text, (x + dx, y + dy), font, scale,
                    (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_corner_brackets(img, color, margin=24, length=48, thickness=2):
    """Draws HUD corner brackets around the whole frame — instant sci-fi vibe."""
    h, w = img.shape[:2]
    m, L = margin, length
    corners = [
        ((m, m), (m + L, m), (m, m + L)),                 # top-left
        ((w - m, m), (w - m - L, m), (w - m, m + L)),     # top-right
        ((m, h - m), (m + L, h - m), (m, h - m - L)),     # bottom-left
        ((w - m, h - m), (w - m - L, h - m), (w - m, h - m - L)),  # bottom-right
    ]
    for c, a, b in corners:
        cv2.line(img, c, a, color, thickness, cv2.LINE_AA)
        cv2.line(img, c, b, color, thickness, cv2.LINE_AA)


# =============================================================================
# Ball detector (HSV color + circular-blob filtering)
# =============================================================================

class BallDetector:
    def __init__(self, hsv_low, hsv_high, min_area, max_area, min_circularity):
        self.low = hsv_low
        self.high = hsv_high
        self.min_area = min_area
        self.max_area = max_area
        self.min_circ = min_circularity
        # Morphological kernel to clean the mask (remove specks, close holes).
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame_bgr):
        """Return (center_xy, radius) of the best ball candidate, or None."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.low, self.high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_score = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            perim = cv2.arcLength(c, True)
            if perim == 0:
                continue
            # circularity = 4*pi*area / perimeter^2  -> 1.0 for a perfect circle
            circ = 4 * math.pi * area / (perim * perim)
            if circ < self.min_circ:
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            # Prefer the roundest + largest candidate.
            score = circ * area
            if score > best_score:
                best_score = score
                best = ((int(x), int(y)), int(radius))
        return best, mask


# =============================================================================
# Trajectory + stats bookkeeping
# =============================================================================

class TrajectoryTracker:
    def __init__(self, fps, px_per_meter=None):
        self.fps = max(1.0, fps)
        self.px_per_meter = px_per_meter  # set this to report real m/s (optional)
        self.points = deque(maxlen=TRAIL_LEN)   # recent (x, y) for the trail
        self.all_points = []                    # full history (for arc fit)
        self.speeds = []                         # px/frame speed history
        self.airborne_frames = 0
        self.release_point = None
        self.landing_point = None
        self.auto_landing = True   # set False when landing is set manually
        self.peak_speed = 0.0
        self.launch_angle_deg = None

    def update(self, center):
        """Feed a new ball position (or None if not detected this frame)."""
        if center is None:
            return
        self.points.append(center)
        self.all_points.append(center)

        if len(self.all_points) >= 2:
            (x0, y0), (x1, y1) = self.all_points[-2], self.all_points[-1]
            speed = math.hypot(x1 - x0, y1 - y0)  # px per frame
            self.speeds.append(speed)
            self.peak_speed = max(self.peak_speed, speed)

            # Release point = first frame the ball is clearly moving fast.
            if self.release_point is None and speed > 6:
                self.release_point = self.all_points[-2]
                self._estimate_launch_angle()

            if self.release_point is not None and self.landing_point is None:
                self.airborne_frames += 1
                self._check_landing(speed)

    def _estimate_launch_angle(self):
        """Launch angle from the first few post-release displacement vectors."""
        pts = self.all_points[-1:] + self.all_points[-2:-1]
        if len(self.all_points) < 3:
            return
        (x0, y0) = self.all_points[-3]
        (x1, y1) = self.all_points[-1]
        dx = x1 - x0
        dy = (y0 - y1)  # invert: image y grows downward, we want "up" positive
        if dx == 0 and dy == 0:
            return
        self.launch_angle_deg = abs(math.degrees(math.atan2(dy, dx)))
        # Fold obtuse angles back so it reads as an elevation, not direction.
        if self.launch_angle_deg > 90:
            self.launch_angle_deg = 180 - self.launch_angle_deg

    def _check_landing(self, speed):
        """Heuristic landing: after enough airborne frames, speed collapses."""
        if not self.auto_landing:
            return  # landing is provided manually (last keyframe = the cup)
        if self.airborne_frames < LANDING_MIN_AIRBORNE_FRAMES:
            return
        if self.peak_speed > 0 and speed < LANDING_SPEED_DROP * self.peak_speed:
            self.landing_point = self.all_points[-1]

    def launch_speed_kmh(self):
        """Report launch speed. Uses px/m if calibrated, else a playful proxy."""
        if not self.speeds:
            return 0.0
        # Use the peak (release) speed as the headline number.
        px_per_frame = self.peak_speed
        if self.px_per_meter:
            m_per_s = px_per_frame * self.fps / self.px_per_meter
            return m_per_s * 3.6
        # Uncalibrated: scale px/frame into a believable-looking km/h number.
        return px_per_frame * self.fps * 0.03


# =============================================================================
# HUD compositor — draws every overlay on top of a frame
# =============================================================================

def render_hud(frame, tracker, ball, frame_idx, total_frames, mask=None):
    h, w = frame.shape[:2]

    # --- global sci-fi frame furniture ---
    draw_corner_brackets(frame, HUD_ACCENT)

    # --- glowing neon motion trail ---
    pts = list(tracker.points)
    for i in range(1, len(pts)):
        # Fade + thin the trail as it gets older for a comet-tail look.
        frac = i / len(pts)
        thickness = max(1, int(6 * frac))
        draw_glow_line(frame, pts[i - 1], pts[i], TRAIL_COLOR, thickness)

    # --- current ball: bounding box + live tracker readout ---
    if ball is not None:
        (cx, cy), r = ball
        r = max(r, 6)
        cv2.rectangle(frame, (cx - r - 4, cy - r - 4), (cx + r + 4, cy + r + 4),
                      HUD_WARN, 1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 3, HUD_WARN, -1, cv2.LINE_AA)
        put_hud_text(frame, f"BALL x:{cx} y:{cy}", (cx + r + 8, cy - r),
                     scale=0.45, color=HUD_WARN)
        put_hud_text(frame, "TRACKING", (cx + r + 8, cy - r + 16),
                     scale=0.45, color=HUD_ACCENT)

    # --- landing visualization: reticle + expanding splash ---
    if tracker.landing_point is not None:
        lp = tracker.landing_point
        draw_reticle(frame, lp, 30, HUD_WARN, thickness=2)
        # Expanding splash ring, animated relative to when landing was detected.
        pulse = (frame_idx % 20) / 20.0
        cv2.circle(frame, lp, int(30 + pulse * 40), HUD_WARN,
                   max(1, int(3 * (1 - pulse))), cv2.LINE_AA)
        put_hud_text(frame, "IMPACT!", (lp[0] - 24, lp[1] + 55),
                     scale=0.7, color=HUD_WARN, thickness=2)

    # --- top-left experiment header ---
    draw_hud_panel(frame, (20, 20), (330, 66))
    put_hud_text(frame, EXPERIMENT_TITLE, (32, 46), scale=0.6, color=HUD_ACCENT)
    put_hud_text(frame, "SUBJECT: HOMO SAPIENS (THIRSTY)", (32, 70),
                 scale=0.42, color=HUD_WARN)

    # --- bottom-left: TOP VIEW impact map (where the ball hit the rack) ---
    draw_top_view_panel(frame, tracker, w, h)

    # --- progress / scanline bar along the top ---
    prog = frame_idx / max(1, total_frames)
    cv2.line(frame, (0, 6), (int(w * prog), 6), HUD_ACCENT, 3, cv2.LINE_AA)

    # --- optional: picture-in-picture of the color mask (debug/aesthetic) ---
    if mask is not None:
        pip = cv2.cvtColor(cv2.resize(mask, (160, 90)), cv2.COLOR_GRAY2BGR)
        pip[:, :, 0] = 0  # tint the mask green -> "sensor view"
        fh, fw = pip.shape[:2]
        frame[h - fh - 20:h - 20, w - fw - 20:w - 20] = pip
        cv2.rectangle(frame, (w - fw - 20, h - fh - 20), (w - 20, h - 20),
                      HUD_ACCENT, 1)
        put_hud_text(frame, "BALL SENSOR", (w - fw - 20, h - fh - 26),
                     scale=0.4, color=HUD_ACCENT)

    return frame


# =============================================================================
# Interactive HSV calibration tool  (python beerpong_science.py --calibrate ...)
# =============================================================================

def run_calibration(input_path):
    """Trackbar UI to find BALL_HSV_LOW / HIGH for your footage."""
    cap = cv2.VideoCapture(input_path)
    ok, frame = cap.read()
    if not ok:
        print("[error] could not read the video.")
        return
    cap.release()

    win = "Calibrate ball HSV (adjust, then press Q)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    names = ["H low", "S low", "V low", "H high", "S high", "V high"]
    defaults = [BALL_HSV_LOW[0], BALL_HSV_LOW[1], BALL_HSV_LOW[2],
                BALL_HSV_HIGH[0], BALL_HSV_HIGH[1], BALL_HSV_HIGH[2]]
    maxes = [179, 255, 255, 179, 255, 255]
    for n, d, m in zip(names, defaults, maxes):
        cv2.createTrackbar(n, win, int(d), m, lambda x: None)

    print("[info] Drag sliders until ONLY the ball is white in the mask.")
    print("[info] Copy the printed values into BALL_HSV_LOW / BALL_HSV_HIGH.")
    while True:
        vals = [cv2.getTrackbarPos(n, win) for n in names]
        low = np.array(vals[:3])
        high = np.array(vals[3:])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, low, high)
        preview = cv2.bitwise_and(frame, frame, mask=mask)
        combo = np.hstack([preview, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        cv2.imshow(win, combo)
        if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
            print(f"\nBALL_HSV_LOW  = np.array({vals[:3]})")
            print(f"BALL_HSV_HIGH = np.array({vals[3:]})\n")
            break
    cv2.destroyAllWindows()


# =============================================================================
# Click-to-track mode (CSRT) — no color needed, great for a pale ball
# =============================================================================

class TemplateFlowTracker:
    """Lightweight tracker for a fast, small object (the ball).

    Strategy: keep a small image template of the ball, and each frame predict
    where it moved using its recent velocity, then run normalized template
    matching in a search window around that prediction. This beats MIL on fast
    motion because the search follows the ball instead of sitting still.

    Same interface as an OpenCV tracker: init(frame, bbox) and
    update(frame) -> (found: bool, (x, y, w, h)).
    """

    # --- tuning knobs ---
    MATCH_THRESHOLD = 0.30   # min match score (0-1) to accept a detection.
                             #   lower = more forgiving (blur), higher = stricter
    SEARCH_MARGIN = 34       # base half-size (px) of the search window padding
    SPEED_GAIN = 1.6         # how much the search window grows with ball speed
    TEMPLATE_BLEND = 0.12    # 0 = fixed template; higher adapts to blur/rotation
    MAX_COAST_FRAMES = 6     # keep predicting this many frames after losing lock

    def __init__(self):
        self.template = None          # grayscale patch of the ball
        self.tw = self.th = 0         # template width/height
        self.center = None            # last known (x, y) center
        self.velocity = (0.0, 0.0)    # last (dx, dy) per frame
        self.coast = 0                # consecutive predicted-only frames

    def init(self, frame, bbox):
        x, y, w, h = (int(v) for v in bbox)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x = max(0, x); y = max(0, y)
        self.template = gray[y:y + h, x:x + w].copy()
        self.th, self.tw = self.template.shape[:2]
        self.center = (x + w / 2.0, y + h / 2.0)
        self.velocity = (0.0, 0.0)
        self.coast = 0
        return True

    def update(self, frame):
        if self.template is None or self.tw == 0 or self.th == 0:
            return False, (0, 0, 0, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]

        # 1) predict center from velocity
        px = self.center[0] + self.velocity[0]
        py = self.center[1] + self.velocity[1]

        # 2) search window grows with speed (so a fast ball stays inside it)
        speed = math.hypot(*self.velocity)
        margin = int(self.SEARCH_MARGIN + self.SPEED_GAIN * speed)
        sx0 = int(max(0, px - self.tw / 2 - margin))
        sy0 = int(max(0, py - self.th / 2 - margin))
        sx1 = int(min(W, px + self.tw / 2 + margin))
        sy1 = int(min(H, py + self.th / 2 + margin))
        if sx1 - sx0 < self.tw or sy1 - sy0 < self.th:
            return self._coast()

        search = gray[sy0:sy1, sx0:sx1]
        res = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < self.MATCH_THRESHOLD:
            return self._coast()

        # 3) matched -> compute new center in full-frame coordinates
        top_left = (sx0 + max_loc[0], sy0 + max_loc[1])
        new_center = (top_left[0] + self.tw / 2.0, top_left[1] + self.th / 2.0)

        # 4) update velocity (smoothed) and template (lightly, to adapt to blur)
        self.velocity = (0.6 * self.velocity[0] + 0.4 * (new_center[0] - self.center[0]),
                         0.6 * self.velocity[1] + 0.4 * (new_center[1] - self.center[1]))
        self.center = new_center
        self.coast = 0
        if self.TEMPLATE_BLEND > 0:
            patch = gray[top_left[1]:top_left[1] + self.th,
                         top_left[0]:top_left[0] + self.tw]
            if patch.shape[:2] == self.template.shape[:2]:
                self.template = cv2.addWeighted(
                    self.template, 1 - self.TEMPLATE_BLEND,
                    patch, self.TEMPLATE_BLEND, 0)
        return True, self._box()

    def _coast(self):
        """Lost the match: keep gliding on last velocity for a few frames."""
        self.coast += 1
        if self.coast > self.MAX_COAST_FRAMES:
            return False, self._box()
        self.center = (self.center[0] + self.velocity[0],
                       self.center[1] + self.velocity[1])
        return True, self._box()

    def _box(self):
        return (int(self.center[0] - self.tw / 2), int(self.center[1] - self.th / 2),
                int(self.tw), int(self.th))


def make_csrt_tracker():
    """Return the best available single-object tracker, or None.

    Preference order: CSRT (most accurate) -> KCF -> MIL. Which ones exist
    depends on how OpenCV was built. OpenCV 5 minimal builds often ship only
    TrackerMIL, which needs no model download and is fine for a ball.
    """
    factories = [
        ("CSRT", lambda: cv2.TrackerCSRT_create()),
        ("CSRT", lambda: cv2.legacy.TrackerCSRT_create()),
        ("CSRT", lambda: cv2.TrackerCSRT.create()),
        ("KCF",  lambda: cv2.TrackerKCF_create()),
        ("KCF",  lambda: cv2.TrackerKCF.create()),
        ("MIL",  lambda: cv2.TrackerMIL_create()),
        ("MIL",  lambda: cv2.TrackerMIL.create()),
    ]
    for name, factory in factories:
        try:
            t = factory()
            print(f"[info] using {name} object tracker")
            return t
        except Exception:
            continue
    return None


# Half-size (px) of the selection box placed around your click. The ball just
# needs to sit comfortably inside it. Adjust live with '+' / '-' in the UI.
CLICK_BOX_HALF = 22


def seek_and_select(input_path):
    """Scrub the video, click the ball, confirm. Returns (seed_idx, bbox).

    Controls:
      d / a : next / previous frame       (find a frame where the ball is clear)
      D / A : jump 10 frames
      click : place the selection box on the ball
      + / - : grow / shrink the box
      ENTER / SPACE : confirm      |      q / ESC : cancel
    """
    cap = cv2.VideoCapture(input_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    win = "Click the BALL  |  d/a=frame  +/-=size  ENTER=ok  q=cancel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    state = {"click": None, "half": CLICK_BOX_HALF}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)
    cv2.setMouseCallback(win, on_mouse)

    idx = 0

    def read_at(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        return fr if ok else None

    frame = read_at(idx)
    result = None
    while frame is not None:
        disp = frame.copy()
        put_hud_text(disp, f"FRAME {idx}/{total}", (20, 30), 0.6, HUD_ACCENT)
        put_hud_text(disp, "click ball | d/a frame | +/- size | ENTER ok | q cancel",
                     (20, disp.shape[0] - 20), 0.5, HUD_WARN)
        if state["click"] is not None:
            cx, cy = state["click"]
            hf = state["half"]
            cv2.rectangle(disp, (cx - hf, cy - hf), (cx + hf, cy + hf),
                          RELEASE_COLOR, 2)
            draw_reticle(disp, (cx, cy), hf, RELEASE_COLOR, thickness=1)
        cv2.imshow(win, disp)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):                    # cancel
            break
        elif key in (13, 32):                        # ENTER / SPACE = confirm
            if state["click"] is not None:
                cx, cy = state["click"]
                hf = state["half"]
                bbox = (cx - hf, cy - hf, 2 * hf, 2 * hf)  # x, y, w, h
                result = (idx, bbox)
                break
        elif key == ord('d'):
            idx = min(total - 1, idx + 1); frame = read_at(idx)
        elif key == ord('a'):
            idx = max(0, idx - 1); frame = read_at(idx)
        elif key == ord('D'):
            idx = min(total - 1, idx + 10); frame = read_at(idx)
        elif key == ord('A'):
            idx = max(0, idx - 10); frame = read_at(idx)
        elif key in (ord('+'), ord('=')):
            state["half"] = min(120, state["half"] + 3)
        elif key in (ord('-'), ord('_')):
            state["half"] = max(6, state["half"] - 3)

    cap.release()
    cv2.destroyAllWindows()
    return result


# =============================================================================
# Manual keyframe mode — click the ball on several frames for max accuracy
# =============================================================================

def annotate_keyframes(input_path):
    """Scrub the video and click the ball on multiple frames.

    Returns (keyframes, half) where keyframes is a sorted list of
    (frame_idx, (x, y)) and half is the box half-size, or None if cancelled.

    Controls:
      d / a : next / previous frame          D / A : jump 10 frames
      click : set/replace the ball point on the CURRENT frame
      x     : delete the point on the current frame
      + / - : grow / shrink the marker size
      ENTER : finish (needs >= 2 points)      q / ESC : cancel
    """
    cap = cv2.VideoCapture(input_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    win = "Click ball on several frames  |  d/a=frame  x=del  ENTER=done  q=cancel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    keyframes = {}   # frame_idx -> (x, y)
    state = {"click": None, "half": CLICK_BOX_HALF}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)
    cv2.setMouseCallback(win, on_mouse)

    idx = 0

    def read_at(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        return fr if ok else None

    frame = read_at(idx)
    confirmed = False
    while frame is not None:
        # apply a fresh click to the CURRENT frame
        if state["click"] is not None:
            keyframes[idx] = state["click"]
            state["click"] = None

        disp = frame.copy()
        hf = state["half"]
        # draw every keyframe; highlight the one on the current frame
        for fidx, (kx, ky) in keyframes.items():
            col = HUD_ACCENT if fidx == idx else HUD_DIM
            cv2.circle(disp, (kx, ky), 4, col, -1, cv2.LINE_AA)
            if fidx == idx:
                draw_reticle(disp, (kx, ky), hf, RELEASE_COLOR, thickness=1)
        put_hud_text(disp, f"FRAME {idx}/{total}   POINTS: {len(keyframes)}",
                     (20, 30), 0.6, HUD_ACCENT)
        put_hud_text(disp, "click ball | d/a frame | x del | +/- size | ENTER done | q cancel",
                     (20, disp.shape[0] - 20), 0.5, HUD_WARN)
        cv2.imshow(win, disp)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == 13:                              # ENTER = finish
            if len(keyframes) >= 2:
                confirmed = True
                break
            print("[info] place at least 2 points before finishing.")
        elif key == ord('x'):
            keyframes.pop(idx, None)
        elif key == ord('d'):
            idx = min(total - 1, idx + 1); frame = read_at(idx)
        elif key == ord('a'):
            idx = max(0, idx - 1); frame = read_at(idx)
        elif key == ord('D'):
            idx = min(total - 1, idx + 10); frame = read_at(idx)
        elif key == ord('A'):
            idx = max(0, idx - 10); frame = read_at(idx)
        elif key in (ord('+'), ord('=')):
            state["half"] = min(120, state["half"] + 3)
        elif key in (ord('-'), ord('_')):
            state["half"] = max(6, state["half"] - 3)

    cap.release()
    cv2.destroyAllWindows()
    if not confirmed:
        return None
    ordered = sorted(keyframes.items())
    return ordered, state["half"]


def _catmull_rom(p0, p1, p2, p3, t):
    """Smooth interpolation passing through p1 and p2 (t in 0..1)."""
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t
                  + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def build_manual_positions(keyframes, half):
    """Interpolate a smooth per-frame position map from clicked keyframes.

    keyframes: sorted list of (frame_idx, (x, y)).
    Returns (positions dict {frame_idx: (x, y)}, radius). Positions are filled
    for every integer frame between the first and last clicked frame using a
    Catmull-Rom spline (a clean curved arc through your points).
    """
    positions = {}
    n = len(keyframes)
    frames = [f for f, _ in keyframes]
    pts = [p for _, p in keyframes]

    for i in range(n - 1):
        f_a, f_b = frames[i], frames[i + 1]
        p1 = pts[i]
        p2 = pts[i + 1]
        p0 = pts[i - 1] if i - 1 >= 0 else p1          # clamp ends
        p3 = pts[i + 2] if i + 2 < n else p2
        span = max(1, f_b - f_a)
        for f in range(f_a, f_b + 1):
            t = (f - f_a) / span
            x = _catmull_rom(p0[0], p1[0], p2[0], p3[0], t)
            y = _catmull_rom(p0[1], p1[1], p2[1], p3[1], t)
            positions[f] = (int(round(x)), int(round(y)))
    return positions, int(half)


# =============================================================================
# Main render loop
# =============================================================================

def find_ffmpeg():
    """Locate an ffmpeg binary: PATH first, else the imageio-ffmpeg bundle."""
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def reencode_h264(ffmpeg, src, dst, crf=23, fps=None):
    """Re-encode src -> dst as H.264/yuv420p with faststart (small + universal).

    crf: quality/size knob. Lower = better quality, bigger file.
         ~18 near-lossless, 23 default, 28 small, 30+ tiny. Sweet spot 23-28.
    """
    import subprocess
    cmd = [ffmpeg, "-y", "-i", src,
           "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
           # yuv420p + even dimensions = plays everywhere (phones, browsers)
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an"]
    if fps:
        cmd += ["-r", f"{fps:.3f}"]
    cmd += [dst]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"[warn] ffmpeg re-encode failed ({e}); keeping the raw file.")
        return False


def process_video(input_path, output_path, preview=False, px_per_meter=None,
                  click_seed=None, tracker_kind="template",
                  manual_positions=None, manual_radius=None, social="reel",
                  crf=23, fit="fill"):
    # click_seed is (seed_frame_idx, bbox) when using --click-track, else None.
    # tracker_kind selects the click-track engine: "template" or "opencv".
    # manual_positions is a {frame_idx: (x, y)} map from clicked keyframes; when
    # present it overrides all automatic tracking for maximum accuracy.
    # social selects the export canvas: "reel"/"portrait"/"square"/"source".
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[error] cannot open {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    # Output canvas: social format or original size.
    if social in SOCIAL_SIZES:
        out_w, out_h = SOCIAL_SIZES[social]
    else:
        out_w, out_h = w, h

    # If ffmpeg is available we write frames to a temp file first, then
    # re-encode to a small, universally-compatible H.264 file. Otherwise we
    # write the final file directly (larger, but still valid).
    ffmpeg = find_ffmpeg()
    import os
    if ffmpeg:
        base, _ = os.path.splitext(output_path)
        write_path = base + "_raw.avi"   # temp; deleted after re-encode
    else:
        write_path = output_path
        print("[warn] ffmpeg not found -> writing directly (file will be larger).")

    writer = None
    for codec in ("avc1", "H264", "mp4v", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(write_path, fourcc, fps, (out_w, out_h))
        if writer.isOpened():
            print(f"[info] rendering with '{codec}' at {out_w}x{out_h} ({social})")
            break
        writer.release()
    if writer is None or not writer.isOpened():
        print("[error] could not open a video writer.")
        cap.release()
        return

    detector = BallDetector(BALL_HSV_LOW, BALL_HSV_HIGH,
                            BALL_MIN_AREA, BALL_MAX_AREA, BALL_MIN_CIRCULARITY)
    tracker = TrajectoryTracker(fps, px_per_meter=px_per_meter)

    # In manual mode the LAST clicked keyframe is the true impact (the cup),
    # so disable the mid-air heuristic and pin the impact there.
    manual_last = None
    if manual_positions is not None:
        tracker.auto_landing = False
        lf = max(manual_positions)
        manual_last = (lf, manual_positions[lf])

    # --- CSRT click-track setup (used instead of color when click_seed given) ---
    csrt = None
    csrt_active = False
    if click_seed is not None:
        if tracker_kind == "opencv":
            csrt = make_csrt_tracker()
        else:
            csrt = TemplateFlowTracker()
            print("[info] using template + velocity tracker")
        if csrt is None:
            print("[warn] no object tracker available in this OpenCV build -> "
                  "falling back to color detection.")
            click_seed = None

    # --- MediaPipe setup ---
    pose = hands = None
    mp_draw = mp_pose = mp_hands = None
    if HAS_MEDIAPIPE:
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)
        hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)

    frame_idx = 0
    print(f"[info] processing {total} frames @ {fps:.1f} fps ...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # ---- Pose + hand skeletons (drawn first, under the HUD) ----
        if HAS_MEDIAPIPE:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_res = pose.process(rgb)
            hands_res = hands.process(rgb)
            rgb.flags.writeable = True

            if pose_res.pose_landmarks:
                mp_draw.draw_landmarks(
                    frame, pose_res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_draw.DrawingSpec(color=SKELETON_COLOR, thickness=2,
                                        circle_radius=2),
                    mp_draw.DrawingSpec(color=HUD_ACCENT, thickness=2))
            if hands_res.multi_hand_landmarks:
                for hlm in hands_res.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame, hlm, mp_hands.HAND_CONNECTIONS,
                        mp_draw.DrawingSpec(color=RELEASE_COLOR, thickness=2,
                                            circle_radius=2),
                        mp_draw.DrawingSpec(color=HUD_WARN, thickness=1))

        # ---- Ball detection + trajectory update ----
        mask = None
        ball = None
        if manual_positions is not None:
            # Highest-accuracy path: use the interpolated clicked keyframes.
            # Loop frame_idx is 1-based; keyframes are 0-based frame indices.
            pos = manual_positions.get(frame_idx - 1)
            if pos is not None:
                ball = (pos, manual_radius or 12)
            # pin the impact reticle to the final clicked point (the cup)
            if manual_last is not None and frame_idx - 1 >= manual_last[0]:
                tracker.landing_point = manual_last[1]
        elif click_seed is not None:
            seed_idx, bbox = click_seed
            if not csrt_active and frame_idx - 1 >= seed_idx:
                # Seed the tracker on (or just after) the frame you clicked.
                csrt.init(frame, tuple(int(v) for v in bbox))
                csrt_active = True
            if csrt_active:
                ok_box, box = csrt.update(frame)
                if ok_box:
                    x, y, bw, bh = box
                    center = (int(x + bw / 2), int(y + bh / 2))
                    radius = int(max(bw, bh) / 2)
                    ball = (center, radius)
        else:
            ball, mask = detector.detect(frame)

        tracker.update(ball[0] if ball else None)

        # ---- HUD overlay ----
        frame = render_hud(frame, tracker, ball, frame_idx, total, mask=mask)

        # ---- wrap onto the social canvas (title bar + caption footer) ----
        out_frame = frame
        if social in SOCIAL_SIZES:
            out_frame = compose_social(frame, out_w, out_h, fit=fit)

        writer.write(out_frame)
        if preview:
            cv2.imshow("Beer Pong FOR SCIENCE (press Q to stop)", out_frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

        if frame_idx % 30 == 0:
            print(f"  ... {frame_idx}/{total} frames")

    cap.release()
    writer.release()
    if preview:
        cv2.destroyAllWindows()
    if pose:
        pose.close()
    if hands:
        hands.close()

    # ---- re-encode to small, browser/social-friendly H.264 ----
    if ffmpeg and write_path != output_path:
        print(f"[info] re-encoding to H.264 (crf={crf}) ...")
        if reencode_h264(ffmpeg, write_path, output_path, crf=crf, fps=fps):
            raw_mb = os.path.getsize(write_path) / 1e6
            out_mb = os.path.getsize(output_path) / 1e6
            print(f"[done] wrote {output_path}  ({out_mb:.2f} MB, "
                  f"was {raw_mb:.2f} MB raw)")
            try:
                os.remove(write_path)      # delete the temp raw file
            except OSError:
                pass
        else:
            # ffmpeg failed: fall back to the raw file as the output
            try:
                os.replace(write_path, output_path)
            except OSError:
                pass
            print(f"[done] wrote {output_path} (raw, un-compressed)")
    else:
        print(f"[done] wrote {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Beer Pong 'For Science' HUD overlay")
    ap.add_argument("--input", "-i", required=True, help="input video path")
    ap.add_argument("--output", "-o", default="beerpong_science.mp4",
                    help="output video path")
    ap.add_argument("--preview", action="store_true",
                    help="show a live window while rendering")
    ap.add_argument("--calibrate", action="store_true",
                    help="open HSV calibration UI for the ball color, then exit")
    ap.add_argument("--click-track", action="store_true",
                    help="scrub the video and CLICK the ball to track it "
                         "(no color needed) — best for a pale/faint ball")
    ap.add_argument("--tracker", choices=["template", "opencv"],
                    default="template",
                    help="click-track engine: 'template' (custom, robust on a "
                         "fast small ball, works in any build) or 'opencv' "
                         "(CSRT/KCF/MIL if your build has them). Default template.")
    ap.add_argument("--keyframes", action="store_true",
                    help="MANUAL mode: click the ball on several frames; the "
                         "arc is interpolated through your points. Most accurate.")
    ap.add_argument("--auto", action="store_true",
                    help="AUTOMATIC mode: find the throw by motion + parabola "
                         "physics (no clicking, no color tuning). See "
                         "auto_track.py.")
    ap.add_argument("--auto-yolo", action="store_true",
                    help="with --auto: also use YOLO 'sports ball' detections")
    ap.add_argument("--auto-smooth", action="store_true",
                    help="with --auto: draw the fitted parabola instead of the "
                         "measured points (smoother arc)")
    ap.add_argument("--format", choices=["reel", "portrait", "square", "source"],
                    default="reel",
                    help="output canvas: 'reel' 9:16 for Reels/Shorts (default), "
                         "'portrait' 4:5 IG feed, 'square' 1:1, 'source' as-is.")
    ap.add_argument("--crf", type=int, default=23,
                    help="H.264 quality/size: lower=better+bigger, higher=smaller. "
                         "18 near-lossless, 23 default, 28 small, 30+ tiny.")
    ap.add_argument("--fit", choices=["fill", "fit"], default="fill",
                    help="'fill' full-bleed (best for vertical footage, default), "
                         "'fit' letterbox whole clip in bars (for horizontal).")
    ap.add_argument("--px-per-meter", type=float, default=None,
                    help="pixels per meter, for real km/h (optional). "
                         "Measure a known length in-frame to find it.")
    args = ap.parse_args()

    if args.calibrate:
        run_calibration(args.input)
        return

    manual_positions = manual_radius = None
    if args.auto:
        # Same hand-off as --keyframes: a {frame: (x, y)} map whose last
        # frame is the impact — only found by physics instead of by clicks.
        try:
            from auto_track import auto_track
        except ImportError:
            from src.auto_track import auto_track
        res = auto_track(args.input, use_yolo=args.auto_yolo,
                         mode="fit" if args.auto_smooth else "hybrid")
        if res is None:
            print("[warn] no throw found automatically. Try --auto-yolo, or "
                  "the manual --keyframes mode.")
            return
        process_video(args.input, args.output, preview=args.preview,
                      px_per_meter=args.px_per_meter,
                      manual_positions=res["positions"],
                      manual_radius=res["radius"], social=args.format,
                      crf=args.crf, fit=args.fit)
        return

    if args.keyframes:
        print("[info] Click the ball on several frames (>=2). More points = "
              "more accurate arc. Press ENTER when done.")
        res = annotate_keyframes(args.input)
        if res is None:
            print("[info] keyframing cancelled.")
            return
        kfs, half = res
        manual_positions, manual_radius = build_manual_positions(kfs, half)
        print(f"[info] {len(kfs)} keyframes -> {len(manual_positions)} "
              f"interpolated frames.")
        process_video(args.input, args.output, preview=args.preview,
                      px_per_meter=args.px_per_meter,
                      manual_positions=manual_positions,
                      manual_radius=manual_radius, social=args.format,
                      crf=args.crf, fit=args.fit)
        return

    click_seed = None
    if args.click_track:
        print("[info] Scrub to a frame where the ball is clearly visible, "
              "click it, then press ENTER.")
        click_seed = seek_and_select(args.input)
        if click_seed is None:
            print("[info] no ball selected -> cancelled.")
            return
        print(f"[info] tracking from frame {click_seed[0]}, box={click_seed[1]}")

    process_video(args.input, args.output, preview=args.preview,
                  px_per_meter=args.px_per_meter, click_seed=click_seed,
                  tracker_kind=args.tracker, social=args.format, crf=args.crf,
                  fit=args.fit)


if __name__ == "__main__":
    main()
