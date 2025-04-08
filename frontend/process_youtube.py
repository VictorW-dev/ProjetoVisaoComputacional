import os
import subprocess
import uuid
from pathlib import Path

def process_video_from_youtube(link, model='lstm', fps=10, max_people=3):
    # Gera um nome único para o vídeo
    video_name = f"ytvideo_{uuid.uuid4().hex[:8]}"
    raw_path = Path("data/raw") / f"{video_name}.mp4"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Baixar vídeo
    print("⬇️ Baixando vídeo do YouTube...")
    subprocess.run(f"python frontend/downloadVideo.py \"{link}\" \"{raw_path}\"", shell=True)

    # 2. Executar pipeline com vídeo único
    print("🚀 Executando pipeline no vídeo baixado...")
    pipeline_cmd = f"python src/run_pipeline.py --model {model} --fps {fps} --max_people {max_people} --train_violence {video_name} --train_nonviolence {video_name}"
    os.system(pipeline_cmd)

    return video_name
