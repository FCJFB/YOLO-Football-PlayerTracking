from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")

    results = model.train(
        data='football-players-detection-20/data.yaml',
        epochs=50,
        imgsz=1280,
        batch=4,           # Reduced from 8 to 4 to accommodate fp32 VRAM usage
        amp=False,         # Disables FP16 mixed precision to fix cuDNN initialization
        device=0,
        workers=2,         # Lower worker threads to prevent RAM bottlenecks
        name="football_v1"
    )

if __name__ == "__main__":
    main()