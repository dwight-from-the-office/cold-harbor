import cv2
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from ultralytics import YOLO
from boxmot import StrongSort

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = PROJECT_ROOT / 'data' / 'video' / 'Training_video_full_game.mp4'

OUTPUT_CSV = PROJECT_ROOT / 'data' / 'processed' / 'player_positions.csv'

OUTPUT_VIDEO = PROJECT_ROOT / 'data' / 'processed' / 'tracked_video_full_game.mp4'

YOLO_MODEL_PATH = PROJECT_ROOT / 'models' / 'yolov8n.pt'

previous_ball_positions = {} # shot detection



def load_yolo_model(model_path):
    try:
        model = YOLO(str(model_path))
        tracker = StrongSort(
            reid_weights=PROJECT_ROOT / "external" / "osnet_x0_25_msmt17.pt", # trained model Identify and remember objects using a CVV
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            max_iou_dist=0.7, # increase if the same player id changes
            max_age=50, # player disappears from frame
            half=False
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
    results = model.predict(frame, verbose=False)[0]
    if not hasattr(results, 'boxes') or len(results.boxes) == 0:
        return np.empty((0,6), dtype=np.float32)
    
    detections = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls not in [0, 32]:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        confidence = float(box.conf[0])
        detections.append(([x1, y1, x2, y2, confidence, cls]))
    
    return np.array(detections, dtype=np.float32)

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



# draw tracking 

def draw_tracking(frame, tracked_objects):

    for track in tracked_objects:
        x1, y1, x2, y2 = track[:4]
        track_id = int(track[4])

        obj_type = 'player' if int(track[5]) == 0 else 'ball'


        color = (0, 255, 0) if obj_type == 'player' else (255, 0, 0)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        cv2.putText(frame, f'ID {track_id}', (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if obj_type == "ball" and detect_shot(previous_ball_positions) == "shooting":
            cv2.putText(frame, "SHOT ATTEMPT", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


def process_video(model, tracker, video_path, output_csv, output_video):
    cap, fps = open_video(video_path)
    if not cap.isOpened():
        print("Error: Canot open Video file.")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    
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
        print(f"Detections shape: {detections.shape}")
        print(f"Processing frame: {frame_count}")
        
        
        # DeepSORT
        try:
            tracked_objects = tracker.update(detections, frame)
        except cv2.error as e:
            print(f"Warning: Tracking update failed due to ECC error: {e}")
        # draw_tracking_info
        draw_tracking(frame, tracked_objects)

        # save processed frame
        out.write(frame)




        # tracked object positions
        for track in tracked_objects:

            track_id = int(track[4])
            x1, y1, x2, y2 = track[:4]
            obj_type = 'player' if int(track[5]) == 0 else 'ball'
            confidance = float(track[6])

            # detect shots based on ball movement
            if obj_type == 'ball':
                previous_ball_positions[track_id] = ((x1 + x2) / 2, (y1 + y2) / 2)
                shot_status = detect_shot(previous_ball_positions)
            else:
                shot_status = None

            if track_id in last_positions:
                last_x, last_y = last_positions[track_id]
                if abs(last_x - (x1 + x2) / 2) < 5 and abs(last_y - (y1 + y2) / 2) < 5:
                    continue
            
            last_positions[track_id] = ((x1 + x2) / 2, (y1 + y2) / 2)

            obj_data = {
            'timestamp': timestamp,
            'type': obj_type,
            'x1': float(x1),
            'y1': float(y1),
            'x2': float(x2),
            'y2': float(y2),
            'center_x': float((x1 + x2)/ 2),
            'center_y': float((y1 + y2) / 2),
            'confidence': confidance,
            'shot_status': shot_status if obj_type == "ball" else None
            }
            frame_data.append(obj_data)
        
        frame_count += 1
    
    cap.release()
    out.release()
    return frame_data



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


