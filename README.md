<div align="center">

# 🏓 Beer Pong Analytics

### Computer vision meets the world's most scientific party game

**Ball tracking · Trajectory physics · Automatic hit/miss detection**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/Ultralytics-YOLO-00FFFF.svg)](https://docs.ultralytics.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*If you can track a 40 mm ball flying into a cup, you can track almost anything.*

</div>

---

## ✨ Overview

Beer pong is a surprisingly good computer-vision benchmark: a **small, fast,
often motion-blurred ball**, a cluttered background, and a binary outcome that
everyone argues about. This project turns a plain phone video into
sports-analytics data.

| Stage | What happens | Where |
|-------|--------------|-------|
| **1. Ball detection** | Ultralytics YOLO (COCO *sports ball*) on every frame, with an HSV colour-mask fallback for frames where the net misses the ball | `src/track_ball.py` |
| **2. Tracking** | Constant-velocity gating keeps the track locked on *one* ball and ignores other round blobs | `src/track_ball.py` |
| **3. Trajectory analysis** | Shot segmentation, smoothing, velocity, ballistic parabola fit, launch angle, apex, flight time | `src/trajectory.py` |
| **4. Hit / miss detection** | Landing point (bounce or track end after the apex) tested against the cup circles | `src/trajectory.py` |
| **5. Visualisation** | Annotated video with trail + live **HIT / MISS** banner, CSV exports, analysis notebook | `src/`, `notebook/` |
| **Automatic tracking** | No clicks, no colour tuning: motion blobs + RANSAC on a parabola find the throw by physics | `src/auto_track.py` |
| **Bonus: "For Science" HUD** | Sci-fi overlay with MediaPipe pose skeleton, neon arc and social-media export (Reels / Shorts) | `src/beerpong_science.py` |

## 🎬 Demo

> **Add your own demo here.** Videos are git-ignored by default, so a demo has
> to be added on purpose.

**Option A: GIF in the repo (renders everywhere)**

```bash
# convert a short clip (≈5 s) with ffmpeg; keep it under ~10 MB
ffmpeg -i data/throw_tracked.mp4 -t 5 -vf "fps=15,scale=640:-1:flags=lanczos" assets/demo.gif
```

Then reference it:

```markdown
![Beer pong tracking demo](assets/demo.gif)
```

**Option B: MP4 hosted by GitHub (better quality, best for longer clips)**

1. Open any issue or PR comment box on GitHub, or edit this README in the web editor
2. Drag & drop your `.mp4` (≤ 10 MB on free plans) into it
3. GitHub uploads it and inserts a `https://github.com/user-attachments/assets/...` URL
4. Paste that URL on its own line in this README; GitHub renders it as an inline video player

<!-- DEMO: replace this comment with ![demo](assets/demo.gif) or the user-attachments URL -->

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/<your-username>/beer-pong-analytics.git
cd beer-pong-analytics
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

> **GPU (optional):** install the CUDA build of PyTorch from
> [pytorch.org](https://pytorch.org/get-started/locally/) *before*
> `requirements.txt` for much faster YOLO inference. The CPU build works fine for short clips.

### 2. Add a video

Put your clip in `data/`, e.g. `data/throw.mp4`. See [`data/README.md`](data/README.md)
for recording tips (static camera, side-on view, 60 fps+).

### 3. Mark the cups (once per camera setup)

```bash
python -m src.track_ball -i data/throw.mp4 --select-cups data/cups.json
```

Click each cup centre, use `+` / `-` to adjust the radius, then press `ENTER` to save.

### 4. Run inference

```bash
python -m src.track_ball -i data/throw.mp4 -o data/throw_tracked.mp4 --cups data/cups.json --csv data/trajectory.csv
```

The YOLO weights (`yolov8n.pt`) are downloaded automatically on first run. Example console output (illustrative):

```
 shot  n_points  flight_time_s  launch_angle_deg  apex_y  outcome  cup
    0        48           0.80             38.20  331.40      hit  7.0
    1        51           0.85             33.90  352.10     miss  NaN
[result] 1/2 hits (50%)
```

Useful flags:

| Flag | Purpose |
|------|---------|
| `--detector yolo\|hsv\|both` | detection back end (default `both` = YOLO + colour fallback) |
| `--model yolov8s.pt` | larger model, better on tiny balls |
| `--imgsz 1280` | higher inference resolution for far-away balls |
| `--conf 0.1` | lower threshold if the ball is often missed |
| `--px-per-meter 950` | real-world scale → launch speed in km/h |
| `--preview` | live window while processing |

### 5. Explore the data

```bash
jupyter notebook notebook/beer_pong_analysis.ipynb
```

The notebook uses `data/trajectory.csv` if it exists, otherwise **synthetic
throws**, so it runs before you have recorded anything.

### Bonus: the "For Science" HUD clip

The HUD renderer has two ways to find the ball:

| | **Manual** `--keyframes` | **Automatic** `--auto` |
|---|---|---|
| How | You click the ball on a few frames; a spline joins them | Motion detection + parabola physics find the throw |
| Setup | ~30 s of clicking per clip | none: no clicks, no colour tuning |
| Accuracy | best (you are the ground truth) | very good on a static camera |
| Best for | the one hero clip you post | batches of clips, whole sessions |

```bash
# manual: click the ball on several frames, ENTER when done
python src/beerpong_science.py --input data/throw.mp4 --output data/science.mp4 --keyframes

# automatic
python src/beerpong_science.py --input data/throw.mp4 --output data/science.mp4 --auto
```

`--auto-smooth` draws the fitted parabola instead of the measured points,
`--auto-yolo` adds YOLO detections as extra candidates. Other options:
`--calibrate` opens an HSV picker for your ball colour, `--click-track` tracks a
ball you click on, `--format reel|portrait|square|source` sets the export canvas.

## 🤖 Automatic tracking (motion + physics)

[`src/auto_track.py`](src/auto_track.py) relies on one physical fact: **a thrown
ball is the only small thing in the scene that moves along a parabola.**

1. **Motion candidates.** Background subtraction (MOG2) is combined with frame
   differencing to find small, roughly round moving blobs. Frames with camera
   shake, where the whole image moves, are skipped.
2. **Tracklets.** Candidates are linked frame to frame with a constant-velocity prediction.
3. **RANSAC on physics.** Short, clean tracklet windows are used as seeds. Each
   seed is grown by pulling in candidates from *any* frame that lies on its fitted
   `y = a·t² + b·t + c`. This recovers the ball where tracking broke
   (motion blur, the hand covering it).
4. **Scoring.** A flight wins on inliers × path length × coverage, divided by fit error.
   It must curve the way gravity does and still be **falling** at the end, which
   rejects wind-ups and arm swings.
5. **Positions.** The output is `{frame: (x, y)}`, the same format as manual
   keyframes. The flight ends where one parabola stops explaining the data,
   which is the first bounce or the cup.

Standalone, with a debug video (grey = candidates, blue = tracklets,
yellow = chosen throw, red = fit) and a CSV that `src/trajectory.py` and the notebook read:

```bash
python -m src.auto_track -i data/throw.mp4 --csv data/auto_trajectory.csv --debug data/auto_debug.mp4
```

`--max-throws N` extracts several throws from a longer video (default 20 standalone, 1 in the HUD).

**Limitations:** it needs a mostly static camera; a phone on a tripod or leaning on
something is ideal. If no throw is found, try `--yolo` or fall back to `--keyframes`.

## 📁 Project structure

```
beer-pong-analytics/
├── src/
│   ├── track_ball.py          # YOLO + HSV detection, tracking, hit/miss overlay, CSV export
│   ├── auto_track.py          # fully automatic tracking: motion + parabola RANSAC
│   ├── trajectory.py          # shot segmentation, physics metrics, hit/miss classification
│   └── beerpong_science.py    # sci-fi HUD renderer (MediaPipe pose, social export)
├── notebook/
│   └── beer_pong_analysis.ipynb
├── data/                      # your videos + outputs (git-ignored)
├── assets/                    # demo GIFs / images for this README
├── requirements.txt
└── LICENSE
```

## 🔬 How hit/miss detection works

1. **Segment** the detections into shots: a gap of more than `--max-gap` frames ends a flight.
2. **Smooth** positions and differentiate to get velocity.
3. **Find the apex**, the highest point (smallest image *y*).
4. **Landing** = first frame after the apex where the ball stops falling (a bounce),
   or the last tracked point (it dropped into a cup or left the frame).
5. **Hit** if the landing point lies within `tolerance × radius` of a cup.

Known limitations: one camera means no true depth, so a ball passing *in
front of* a cup can look like a hit. A side-on camera roughly level with the rim
minimises this. Stereo or a second top-down camera is the natural next step.

## 🎓 Research context

This project grew out of a sports data science research context in NRW
(North Rhine-Westphalia, Germany): using an everyday game as an accessible
testbed for the methods behind performance analysis, such as object tracking,
trajectory modelling and outcome classification.

➡️ **Read more:** [NRW Sports Data Science research](https://example.com/TODO-add-link)

## 🤝 Contributing

Ideas welcome: Kalman filtering, a fine-tuned ball detector, multi-camera 3D
reconstruction, per-player statistics. Open an issue or a PR.

## 📄 License

[MIT](LICENSE). Drink responsibly; track irresponsibly. 🍺
