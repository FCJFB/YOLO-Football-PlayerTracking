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
| `requirements.txt` | Python dependencies (`ultralytics`, `opencv-python`, `numpy`). |
| `pitch_calibration.json` | (Optional, not committed) manual pitch calibration. |

## Setup

Python 3.12 is expected (see `.python-version`). Create a virtual environment and
install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`ultralytics` pulls in PyTorch automatically. If you need GPU acceleration, install
a CUDA-enabled PyTorch build first (matching your driver/CUDA version), then run the
requirements install so the rest resolve against it:

> This project was developed and tested on an NVIDIA GTX 1080 Ti (11 GB VRAM).
> `train.py` sets `batch=4` and `amp=False` to stay within its VRAM limits and avoid
> cuDNN init issues; adjust these if you run on different hardware.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
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
