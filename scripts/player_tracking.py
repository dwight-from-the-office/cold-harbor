import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np


'''
VIDEO_PATH = '../data/video/game.mp4'
OUTPUt_CSV = '../data/processed/player_positions.csv'
YOLO_MODEL_PATH = 'yolov8.pt'
'''



def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
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
        x1, y1, x2, y2 = box.xyxy[0].cpu.tolist()
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