# Constrói vetores de características (relativas/absolutas)

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def extract_features_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data or len(data) == 0:
        return None

    # Usar apenas a primeira pessoa detectada no frame (poderia estender p/ várias)
    person = data[0]  # lista de pares [x, y]
    
    flat_keypoints = []
    for (x, y) in person:
        flat_keypoints.extend([x, y])  # vetor: [x1, y1, x2, y2, ..., x17, y17]

    return flat_keypoints

def process_keypoints_dir(input_dir, output_path):
    input_dir = Path(input_dir)
    json_files = sorted(input_dir.glob("*.json"))

    all_features = []

    for json_file in json_files:
        features = extract_features_from_json(json_file)
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

    args = parser.parse_args()
    process_keypoints_dir(args.input_dir, args.output_path)
