# Extrai frames dos vídeos com diferentes FPS

# Exemplo de como usar no terminal: python src/1_extract_frames.py --fps 10
# Você pode mudar o FPS para testar diferentes taxas como 5, 10, 15...

import cv2
import os
import argparse
from pathlib import Path

def extract_frames(video_path, output_dir, fps):
    cap = cv2.VideoCapture(str(video_path))
    video_name = video_path.stem

    os.makedirs(output_dir, exist_ok=True)
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = round(original_fps / fps)

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"{video_name}_frame_{saved_count:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"[{video_name}] {saved_count} frames salvos em {output_dir}")

def process_all_videos(input_dir, output_base_dir, fps):
    input_dir = Path(input_dir)
    videos = list(input_dir.glob("*.mp4"))

    if not videos:
        print(f"Nenhum vídeo .mp4 encontrado em {input_dir}")
        return

    for video_path in videos:
        video_name = video_path.stem
        output_dir = Path(output_base_dir) / video_name / f"fps_{fps}"
        extract_frames(video_path, output_dir, fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrai frames dos vídeos com FPS especificado.")
    parser.add_argument("--input_dir", type=str, default="data/raw", help="Pasta com os vídeos originais.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Pasta para salvar os frames.")
    parser.add_argument("--fps", type=int, default=10, help="Taxa de quadros desejada (FPS).")

    args = parser.parse_args()
    process_all_videos(args.input_dir, args.output_dir, args.fps)
