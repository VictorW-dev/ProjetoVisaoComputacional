# Constrói vetores de características (relativas/absolutas)

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def extract_features_from_json(json_path, max_people):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data or len(data) == 0:
        return None

    # Pega até `max_people` pessoas
    people = data[:max_people]

    flat_keypoints = []
    for person in people:
        for (x, y) in person:
            flat_keypoints.extend([x, y])  # vetor: [x1, y1, ..., x17, y17] por pessoa

    # Se houver menos que max_people no frame, preenche com zeros
    total_keypoints = max_people * 17 * 2  # 17 keypoints com (x, y)
    flat_keypoints += [0] * (total_keypoints - len(flat_keypoints))

    return flat_keypoints

def process_keypoints_dir(input_dir, output_path, max_people):
    input_dir = Path(input_dir)
    json_files = sorted(input_dir.glob("*.json"))

    all_features = []

    for json_file in json_files:
        features = extract_features_from_json(json_file, max_people)
        if features:
            all_features.append(features)
        else:
            print(f"⚠️ Sem keypoints em: {json_file.name}")

    if all_features:
        df = pd.DataFrame(all_features)
        df.to_csv(output_path, index=False)
        print(f"✅ Vetores salvos em: {output_path}")
    else:
        print("❌ Nenhum vetor foi extraído.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrai vetores de características dos keypoints YOLO Pose.")
    parser.add_argument("--input_dir", type=str, required=True, help="Pasta com os .json de keypoints")
    parser.add_argument("--output_path", type=str, required=True, help="Caminho para salvar o .csv de vetores")
    parser.add_argument("--max_people", type=int, default=1, help="Máximo de pessoas por frame")

    args = parser.parse_args()
    process_keypoints_dir(args.input_dir, args.output_path, args.max_people)
