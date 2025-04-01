import cv2
import numpy as np
import os

def generate_dummy_video(output_path, width=640, height=480, duration=5, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    num_frames = duration * fps

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Criando um quadrado que se move horizontalmente
        top_left_x = (i * 5) % (width - 100)
        cv2.rectangle(frame, (top_left_x, 100), (top_left_x + 100, 200), (0, 255, 0), -1)

        out.write(frame)

    out.release()
    print(f"Vídeo gerado: {output_path}")

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    generate_dummy_video("data/raw/video_teste.mp4")
