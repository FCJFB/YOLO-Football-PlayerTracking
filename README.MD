# Football Player Team Segmentation

A small work-in-progress experiment that detects and tracks players in a football
broadcast, classifies them into teams by kit colour, and projects their positions
onto a top-down pitch radar.

## What it does

- Tracks players (YOLO + ByteTrack) in a single broadcast clip.
- Classifies each tracked player into one of two teams using a simple HSV
  heuristic on the bounding-box crop.
- Maps player feet onto a top-down pitch using a homography, shown as a live
  side-by-side radar view.

## What it does *not* do (yet)

- It is **not** a general-purpose system. The team classifier hardcodes the two
  kits in the demo clip (`data/barca-betis.mp4`) and the colour thresholds were
  tuned for that footage. Other matches will need re-tuning.
- The homography falls back to green-field auto-detection, which only works on
  wide shots where the whole pitch is visible and is not reliable otherwise.

## Files

| File | Purpose |
| --- | --- |
| `player-team-segmentation.py` | Main analysis/visualisation loop. |
| `train.py` | Trains a YOLO model on the Roboflow football-players dataset. |
| `custom_bytetrack.yaml` | ByteTrack tracker config used by the main script. |
| `pitch_calibration.json` | (Optional, not committed) manual pitch calibration. |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install ultralytics opencv-python numpy
```

## Usage

```bash
python player-team-segmentation.py
```

Press `q` to quit. Drag the trackbar to scrub through the video.

## Status / next steps

- WIP and heavily overfitted to the demo clip.
- Next: landmark marking (manual corner/landmark annotation) so the pitch
  homography is accurate instead of relying on the fragile green-field detection.

## Data

Training data comes from the
[football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc)
dataset on Roboflow (CC BY 4.0).
