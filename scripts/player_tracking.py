import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


VIDEO_PATH = PROJECT_ROOT / 'data' / 'video' / "Unity VS SJ Titans.mp4"
OUTPUT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'player_positions.csv'
YOLO_MODEL_PATH = PROJECT_ROOT / 'models' / 'yolov8n.pt'




def load_yolo_model(model_path):
    try:
        model = YOLO(str(model_path))
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        exit()

def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Invalide FPS retrived from video")
        exit()
    print(f"Video loaded successfully. FPS: {fps}")
    return cap, fps

def detect_objects(model, frame):
    results = model.predict(frame, verbose=True)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return []
    return results


def extract_positions(detections, timestamp):
    objects = []
    for box in detections.boxes:
        cls = int(box.cls[0])
        if cls not in [0, 32]: # assumes 0 = person , 32=basketball
            continue
        obj_type = 'player' if cls == 0 else 'ball'
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        obj_data = {
            'timestamp': timestamp,
            'type': obj_type,
            'x1': float(x1),
            'y1': float(y1),
            'x2': float(x2),
            'y2': float(y2),
            'center_x': float((x1 + x2)/ 2),
            'center_y': float((y1 + y2) / 2),
            'confidence': float(box.conf[0])
        }
        objects.append(obj_data)

    return objects

def process_video(model, video_path, output_csv):
    cap, fps = open_video(video_path)
    frame_data = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        timestamp = frame_count / fps

        detections = detect_objects(model, frame)
        if detections:
            positions = extract_positions(detections, timestamp)
            frame_data.extend(positions)

        frame_count += 1
    cap.release()
    return frame_data

def save_to_csv(frame_data, output_csv_path):
    df = pd.DataFrame(frame_data)
    df.to_csv(output_csv_path, index=False)
    print(f"Data successfully saved to {output_csv_path}")


def main():
    model = load_yolo_model(YOLO_MODEL_PATH)
    data = process_video(model, VIDEO_PATH, OUTPUT_CSV)
    save_to_csv(data, OUTPUT_CSV)

if __name__ == "__main__":
    main()

print(f"Project root: {PROJECT_ROOT}")
print(VIDEO_PATH.exists())
print(YOLO_MODEL_PATH.exists())
