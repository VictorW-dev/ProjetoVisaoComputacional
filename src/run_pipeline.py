import os
import subprocess
import argparse
from pathlib import Path

def process_video(video_name, fps, max_people):
    # 1. Extrair frames
    print(f"🎞️ Extraindo frames de {video_name}.mp4...")
    subprocess.run(f"python src/1_extract_frames.py --fps {fps}", shell=True)

    # 2. Rodar YOLO Pose
    input_dir = f"data/processed/{video_name}/fps_{fps}"
    output_dir = f"data/keypoints/{video_name}/fps_{fps}"
    print(f"📌 Detectando poses em {video_name}...")
    subprocess.run(f"python src/2_yolo_pose_inference.py --input_dir {input_dir} --output_dir {output_dir}", shell=True)

    # 3. Extrair vetores com número configurável de pessoas
    print(f"📈 Extraindo vetores com até {max_people} pessoas por frame...")
    subprocess.run(
        f"python src/3_extract_features.py --input_dir {output_dir} --output_path {output_dir}/features.csv --max_people {max_people}",
        shell=True
    )

    return f"{output_dir}/features.csv"

def run_pipeline(train_violence, train_nonviolence, test_violence=None, test_nonviolence=None, model="lstm", fps=10, max_people=3):
    print("🚀 Rodando pipeline completo...")

    # Processa vídeos de treino
    train_v = [process_video(v, fps, max_people) for v in train_violence]
    train_nv = [process_video(v, fps, max_people) for v in train_nonviolence]

    # Treina modelo
    model_path = f"models/{model}_violence.pth"
    train_cmd = f"python src/4_train_model.py --model {model} --violence {' '.join(train_v)} --nonviolence {' '.join(train_nv)} --max_people {max_people}"
    print("🤖 Treinando modelo...")
    subprocess.run(train_cmd, shell=True)

    # Avaliação (se vídeos de teste forem fornecidos)
    if test_violence and test_nonviolence:
        test_v = [process_video(v, fps, max_people) for v in test_violence]
        test_nv = [process_video(v, fps, max_people) for v in test_nonviolence]

        eval_cmd = f"python src/5_evaluate_model.py --model {model} --model_path {model_path} --violence {' '.join(test_v)} --nonviolence {' '.join(test_nv)}"
        print("📊 Avaliando modelo...")
        subprocess.run(eval_cmd, shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa o pipeline completo para treino e teste.")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru", "rnn"])
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--max_people", type=int, default=3, help="Máximo de pessoas por frame")
    parser.add_argument("--train_violence", nargs="+", required=True)
    parser.add_argument("--train_nonviolence", nargs="+", required=True)
    parser.add_argument("--test_violence", nargs="+", help="Vídeos de teste com violência")
    parser.add_argument("--test_nonviolence", nargs="+", help="Vídeos de teste sem violência")
    args = parser.parse_args()

    run_pipeline(
        train_violence=args.train_violence,
        train_nonviolence=args.train_nonviolence,
        test_violence=args.test_violence,
        test_nonviolence=args.test_nonviolence,
        model=args.model,
        fps=args.fps,
        max_people=args.max_people
    )
