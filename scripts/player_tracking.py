import cv2
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from ultralytics import YOLO
from boxmot import StrongSort

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = PROJECT_ROOT / 'data' / 'video' / 'Training_video.mp4'

OUTPUT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'player_positions.csv'

OUTPUT_VIDEO = PROJECT_ROOT / 'data' / 'processed' / 'tracked_video.mp4'

YOLO_MODEL_PATH = PROJECT_ROOT / 'models' / 'yolov8n.pt'

previous_ball_positions = {} # shot detection



def load_yolo_model(model_path):
    try:
        model = YOLO(str(model_path))
        tracker = StrongSort(
            model_weights="osnet_x0_25_msmt17.pt", # found pretrained model Identify and remember objects using a CVV
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print("Model loaded successfully.")
        return model, tracker
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
    
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls not in [0, 32]:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        confidence = float(box.conf[0])
        detections.append(([x1, y1, x2, y2], confidence, cls))
    
    return detections

def detect_shot(ball_positions):
    # detects when a shot is taken mostlily needs to be tuned 
    # cause its a check if the ball follows an upward path and then falls down

    if len(ball_positions) < 3:
        return False # not enough frams to get trajectory
    
    prev_frames = list(ball_positions.values())[-3:]
    y_vals = [pos[1] for pos in prev_frames] # get y-cords of last 3 frames

    # shot path logic
    if y_vals[0] > y_vals[1] > y_vals[2]: # ball going up
        return "shooting"
    elif y_vals[0] < y_vals[1] < y_vals[2]: # ball coming down
        return "descending"
    return None


'''
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
'''
# draw tracking method




def process_video(model, tracker, video_path, output_csv, output_video):
    cap, fps = open_video(video_path)
    if not cap.isOpened():
        print("Error: Canot open Video file.")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
    
    frame_data = []
    frame_count = 0
    last_positions = {} # stores previous player positions to aviod duplicates

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        timestamp = frame_count / fps

        detections = detect_objects(model, frame)
        
        # DeepSORT
        tracked_objects = tracker.update(detections, frame)

        # draw_tracking_info
        # tracked object positions
        for track in tracked_objects:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = track.to_ltwh()
            obj_type = 'player' if track.det_class == 0 else 'ball'

            # detect shots based on ball movement
            if obj_type == 'ball':
                previous_ball_positions[track_id] = ((x1 + x2) / 2, (y1 - y2) / 2)
                shot_status = detect_shot(previous_ball_positions)
            else:
                shot_status = None
            

            obj_data = {
                            'timestamp': timestamp,
            'type': obj_type,
            'x1': float(x1),
            'y1': float(y1),
            'x2': float(x2),
            'y2': float(y2),
            'center_x': float((x1 + x2)/ 2),
            'center_y': float((y1 + y2) / 2),
            'confidence': track.det_conf,
            'shot_status': shot_status if obj_type == "ball" else None
            }
            frame_data.append(obj_data)




def save_to_csv(frame_data, output_csv_path):
    df = pd.DataFrame(frame_data)
    df.to_csv(output_csv_path, index=False)
    print(f"Data successfully saved to {output_csv_path}")


def main():
    model, tracker = load_yolo_model(YOLO_MODEL_PATH)
    data = process_video(model, tracker, VIDEO_PATH, OUTPUT_CSV, OUTPUT_VIDEO)
    save_to_csv(data, OUTPUT_CSV)

if __name__ == "__main__":
    main()


