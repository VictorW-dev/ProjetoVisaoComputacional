# Roda YOLO Pose e salva keypoints

import os
import cv2
import json
import argparse
from pathlib import Path
from ultralytics import YOLO

def run_pose_estimation(model_path, input_dir, output_dir):
    model = YOLO(model_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(input_dir.glob("*.jpg"))

    for frame_path in frames:
        img = cv2.imread(str(frame_path))
        results = model(img, verbose=False)[0]

        frame_keypoints = []
        for person in results.keypoints.xy:
            keypoints = person.cpu().numpy().tolist()
            frame_keypoints.append(keypoints)

        # Salvar keypoints como JSON por frame
        output_path = output_dir / f"{frame_path.stem}.json"
        with open(output_path, 'w') as f:
            json.dump(frame_keypoints, f)

        print(f"🔍 Keypoints salvos: {output_path.name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roda YOLO Pose sobre frames e salva keypoints.")
    parser.add_argument("--model_path", type=str, default="yolov8n-pose.pt", help="Caminho para o modelo YOLO Pose")
    parser.add_argument("--input_dir", type=str, required=True, help="Pasta com os frames .jpg")
    parser.add_argument("--output_dir", type=str, required=True, help="Pasta para salvar os keypoints")

    args = parser.parse_args()
    run_pose_estimation(args.model_path, args.input_dir, args.output_dir)
