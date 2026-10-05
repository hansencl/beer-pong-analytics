# data/

Everything in this folder except this file and `.gitkeep` is **git-ignored** —
videos and tracker outputs stay on your machine.

## Adding a sample video

1. Record a throw (or a whole game). Tips for good tracking:
   - **Static camera** (tripod / phone leaning on something), side-on to the table
   - **60 fps or higher** — a ping-pong ball moves ~10 px/frame at 30 fps
   - Good lighting and a ball colour that contrasts with the background
   - Keep the whole cup rack in frame
2. Copy it here, e.g. `data/throw.mp4` (`.mp4`, `.mov`, `.avi` all work).
3. Mark the cups once:
   ```bash
   python -m src.track_ball -i data/throw.mp4 --select-cups data/cups.json
   ```

## Files the pipeline writes here

| File | Content |
|------|---------|
| `cups.json` | cup centres + radii in pixels |
| `trajectory.csv` | one row per detected ball position (`frame, t, x, y, r, conf, source, shot`) |
| `trajectory_shots.csv` | one row per shot: launch angle, apex, flight time, `outcome` (hit/miss), `cup` |
| `*_tracked.mp4` | annotated video with trail, cups and HIT/MISS banner |
