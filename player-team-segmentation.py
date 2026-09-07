import cv2
import json
import os
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------
SOURCE = "data/barca-betis.mp4"
CALIBRATION_FILE = "pitch_calibration.json"

# Overlay the auto-detected pitch quadrilateral onto the broadcast feed.
SHOW_FIELD_DETECTION = True

model = YOLO("yolo26l.pt")

cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    print(f"Error opening source: {SOURCE}")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
WINDOW = "Football Analytics"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

is_user_seeking = False


def on_trackbar(val):
    global is_user_seeking
    is_user_seeking = True
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)


cv2.createTrackbar("Position", WINDOW, 0, total_frames, on_trackbar)


# ---------------------------------------------------------
# PITCH RADAR TEMPLATE
# ---------------------------------------------------------
PITCH_W, PITCH_H = 1050, 680

# Top-down pitch rectangle corners: TL, TR, BR, BL.
PITCH_CORNERS = np.float32([
    [20, 20],
    [PITCH_W - 20, 20],
    [PITCH_W - 20, PITCH_H - 20],
    [20, PITCH_H - 20],
])


def create_pitch_template():
    """Standardized top-down tactical pitch (105m x 68m, scaled x10)."""
    pitch = np.zeros((PITCH_H, PITCH_W, 3), dtype=np.uint8)
    pitch[:] = (34, 139, 34)

    cv2.rectangle(pitch, (20, 20), (PITCH_W - 20, PITCH_H - 20), (255, 255, 255), 2)
    cv2.line(pitch, (PITCH_W // 2, 20), (PITCH_W // 2, PITCH_H - 20), (255, 255, 255), 2)
    cv2.circle(pitch, (PITCH_W // 2, PITCH_H // 2), 91, (255, 255, 255), 2)

    cv2.rectangle(pitch, (20, 136), (185, 544), (255, 255, 255), 2)
    cv2.rectangle(pitch, (PITCH_W - 185, 136), (PITCH_W - 20, 544), (255, 255, 255), 2)
    return pitch


# ---------------------------------------------------------
# HOMOGRAPHY
# ---------------------------------------------------------
# Reliable option: a one-time manual calibration saved to disk (run calibrate.py
# once). If no calibration exists, fall back to green-field auto-detection,
# which only works on wide shots where the whole pitch is visible.
def load_calibration():
    if not os.path.exists(CALIBRATION_FILE):
        return None
    with open(CALIBRATION_FILE) as f:
        pts = json.load(f)
    if len(pts) != 4:
        return None
    src = np.float32(pts)  # image points: TL, TR, BR, BL
    return cv2.getPerspectiveTransform(src, PITCH_CORNERS)


def _green_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = ((h >= 35) & (h <= 85) & (s > 40) & (v > 40)).astype(np.uint8)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    return green


def _field_corners(mask):
    """Four corners (TL, TR, BR, BL) of the dominant green region, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 0.25 * mask.shape[0] * mask.shape[1]:
        return None

    hull = cv2.convexHull(largest).reshape(-1, 2).astype(np.float32)
    if len(hull) < 4:
        return None

    tl = hull[np.argmin(hull[:, 0] + hull[:, 1])]
    br = hull[np.argmax(hull[:, 0] + hull[:, 1])]
    tr = hull[np.argmax(hull[:, 0] - hull[:, 1])]
    bl = hull[np.argmin(hull[:, 0] - hull[:, 1])]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_field_homography(frame):
    """image->pitch homography from the pitch boundary, or None if unclear."""
    mask = _green_mask(frame)
    corners = _field_corners(mask)
    if corners is None:
        return None
    return cv2.getPerspectiveTransform(corners, PITCH_CORNERS)


def transform_point(point, H):
    """Map an image point (player feet) to top-down radar coordinates.

    H maps image -> pitch.
    """
    p = H @ np.array([point[0], point[1], 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-9:
        return None
    return int(round(p[0] / p[2])), int(round(p[1] / p[2]))


# ---------------------------------------------------------
# TEAM CLASSIFICATION (HSV kit color)
# ---------------------------------------------------------
TEAM_COLORS = {
    0: (0, 0, 255),           # Barcelona -> red
    1: (255, 100, 0),         # Real Betis -> blue
    "neutral": (120, 120, 120),
}
TEAM_NAMES = {0: "Barca", 1: "Betis", "neutral": "?"}

track_votes = {}


def classify_team(player_crop):
    h, w = player_crop.shape[:2]
    if h < 4 or w < 4:
        return "neutral"

    if h < 25:
        chest = player_crop[0:int(h * 0.6), :]
    else:
        chest = player_crop[int(h * 0.15):int(h * 0.55), int(w * 0.15):int(w * 0.85)]

    if chest.size == 0:
        return "neutral"

    hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
    mh = float(np.median(hsv[:, :, 0]))
    ms = float(np.median(hsv[:, :, 1]))
    mv = float(np.median(hsv[:, :, 2]))

    if (mh < 25 or mh > 150) and ms > 20:
        return 0
    if (40 <= mh <= 95) or (ms < 35 and mv > 80):
        return 1

    return "neutral"


def draw_label(img, text, pos, bg):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), bg, -1)
    cv2.putText(img, text, (x + 2, y - 2), font, scale, (255, 255, 255), thickness)


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
fixed_H = load_calibration()
if fixed_H is not None:
    print(f"Loaded calibration from {CALIBRATION_FILE}.")
else:
    print(f"No {CALIBRATION_FILE} found; using green-field auto-detection "
          "(run calibrate.py once for reliable positioning).")

H_active = fixed_H

while cap.isOpened():
    if not is_user_seeking:
        cv2.setTrackbarPos("Position", WINDOW, int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    is_user_seeking = False

    success, frame = cap.read()
    if not success:
        break

    # --- Homography ---
    if fixed_H is None:
        H = detect_field_homography(frame)
        if H is not None:
            if H_active is None:
                H_active = H
            else:
                H_active = 0.8 * H_active + 0.2 * H

        if SHOW_FIELD_DETECTION and H is not None:
            corners = _field_corners(_green_mask(frame))
            if corners is not None:
                cv2.polylines(frame, [corners.astype(np.int32)], True, (0, 255, 0), 2)

    pitch_radar = create_pitch_template()

    # --- Player detection + team classification ---
    results = model.track(
        frame,
        persist=True,
        tracker="custom_bytetrack.yaml",
        classes=[0],
        conf=0.15,
        imgsz=1280,
        vid_stride=2,
        verbose=False,
    )

    boxes = results[0].boxes
    if boxes is not None and boxes.id is not None:
        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.int().cpu().numpy()

        for box, track_id in zip(xyxy, ids):
            x1, y1, x2, y2 = map(int, box)

            crop = frame[y1:y2, x1:x2]
            team = classify_team(crop)

            votes = track_votes.setdefault(int(track_id), {})
            votes[team] = votes.get(team, 0) + 1
            stable_team = max(votes, key=votes.get)

            color = TEAM_COLORS.get(stable_team, (120, 120, 120))
            name = TEAM_NAMES.get(stable_team, "?")
            label = f"ID:{track_id} {name}"

            feet = (int((x1 + x2) / 2), y2)

            cv2.circle(frame, feet, 3, (0, 255, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, label, (x1, max(y1 - 5, 15)), color)

            # --- Project to radar ---
            if H_active is not None:
                radar_pt = transform_point(feet, H_active)
                if radar_pt is not None:
                    rx, ry = radar_pt
                    if 0 <= rx < PITCH_W and 0 <= ry < PITCH_H:
                        cv2.circle(pitch_radar, (rx, ry), 7, color, -1)
                        cv2.circle(pitch_radar, (rx, ry), 9, (255, 255, 255), 1)

    # --- Side-by-side display ---
    target_h = frame.shape[0]
    target_w = int(target_h * (PITCH_W / PITCH_H))
    radar_resized = cv2.resize(pitch_radar, (target_w, target_h))
    combined = np.hstack((frame, radar_resized))

    cv2.imshow(WINDOW, combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
