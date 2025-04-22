import os
import cv2
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, fps):
    video_name = Path(video_path).stem
    output_path = Path(output_dir) / video_name / f"fps_{fps}"
    output_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Erro ao abrir o vídeo: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(original_fps / fps))

    total = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            frame_filename = output_path / f"{video_name}_frame_{total:04d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            total += 1

        frame_id += 1

    cap.release()
    print(f"[{video_name}] {total} frames salvos em {output_path}")

def process_all_videos(input_dir, output_dir, fps):
    for file in Path(input_dir).glob("*.mp4"):
        extract_frames(file, output_dir, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrai frames de vídeos em .mp4")
    parser.add_argument("--input_video", type=str, help="Caminho de um único vídeo .mp4")
    parser.add_argument("--input_dir", type=str, help="Pasta com vídeos .mp4")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Pasta de saída")
    parser.add_argument("--fps", type=int, default=10, help="Frames por segundo")

    args = parser.parse_args()

    if args.input_video:
        extract_frames(args.input_video, args.output_dir, args.fps)
    elif args.input_dir:
        process_all_videos(args.input_dir, args.output_dir, args.fps)
    else:
        print("❗ É necessário fornecer --input_video ou --input_dir")