import os
import subprocess
import uuid
from pathlib import Path
import pandas as pd

def process_video_from_youtube(source, is_local=False, model="lstm", fps=10, max_people=5, hidden_size=64):
    if is_local:
        video_name = Path(source).stem
        raw_path = f"data/raw/{video_name}.mp4"
    else:
        print("⬇️ Baixando vídeo do YouTube...")
        video_name = f"ytvideo_{uuid.uuid4().hex[:8]}"
        raw_path = f"data/raw/{video_name}.mp4"
        download_cmd = f"yt-dlp --cookies cookies.txt -f best -o {raw_path} {source}"
        subprocess.run(download_cmd, shell=True)

        if not Path(raw_path).exists():
            print(f"❌ Erro: vídeo não foi baixado corretamente em {raw_path}")
            return video_name, None

    print("🎞️ Extraindo frames...")
    subprocess.run(
        f"python src/1_extract_frames.py --input_video {raw_path} --output_dir data/processed --fps {fps}",
        shell=True
    )

    input_dir = f"data/processed/{video_name}/fps_{fps}"
    output_dir = f"data/keypoints/{video_name}/fps_{fps}"
    feature_path = f"{output_dir}/features.csv"

    print("📌 Rodando YOLO Pose...")
    subprocess.run(f"python src/2_yolo_pose_inference.py --input_dir {input_dir} --output_dir {output_dir}", shell=True)

    print("📈 Extraindo vetores de características...")
    subprocess.run(f"python src/3_extract_features.py --input_dir {output_dir} --output_path {feature_path} --max_people {max_people}", shell=True)

    if not Path(feature_path).exists():
        print("❌ Nenhum vetor foi extraído. Abordagem encerrada.")
        return video_name, None

    print("📊 Avaliando com modelo treinado...")
    model_path = f"models/{model}_violence.pth"
    subprocess.run(
        f"python src/5_evaluate_model.py --model {model} --model_path {model_path} --violence {feature_path} "
        f"--nonviolence {feature_path} --max_people {max_people} --video_name {video_name} --hidden_size {hidden_size}",
        shell=True
    )

    # Interpreta o resultado
    predictions_file = f"report/{video_name}_predictions.csv"
    if not Path(predictions_file).exists():
        print("❌ Arquivo de predições não encontrado.")
        return video_name, None

    try:
        df = pd.read_csv(predictions_file)
        predicted_class = int(df['Previsto'].mean() > 0.5)
        return video_name, predicted_class
    except Exception as e:
        print(f"❌ Erro ao ler predições: {e}")
        return video_name, None