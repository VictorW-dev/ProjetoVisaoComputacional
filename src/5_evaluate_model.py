
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==== Parâmetros base ====
KEYPOINTS_PER_PERSON = 34
SEQ_LEN = 30
BATCH_SIZE = 16

# ==== Dataset ====
class ViolenceDataset(Dataset):
    def __init__(self, data_paths, labels, seq_len):
        self.samples = []
        for path, label in zip(data_paths, labels):
            data = pd.read_csv(path).values
            for i in range(0, len(data) - seq_len + 1):
                seq = data[i:i + seq_len]
                self.samples.append((seq, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ==== Modelos ====
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, h_n = self.rnn(x)
        return self.fc(h_n[-1])

# ==== Avaliação ====
def evaluate_model(model_type, model_path, violence_csvs, nonviolence_csvs, max_people=3, video_name="ytvideo", hidden_size=64):
    print(f"📊 Avaliando modelo: {model_type.upper()}")

    input_size = max_people * KEYPOINTS_PER_PERSON
    data_paths = violence_csvs + nonviolence_csvs
    labels = [1] * len(violence_csvs) + [0] * len(nonviolence_csvs)

    dataset = ViolenceDataset(data_paths, labels, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if model_type == "lstm":
        model = LSTMClassifier(input_size, hidden_size)
    elif model_type == "gru":
        model = GRUClassifier(input_size, hidden_size)
    elif model_type == "rnn":
        model = RNNClassifier(input_size, hidden_size)
    else:
        raise ValueError("Modelo inválido.")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.numpy())

    report = classification_report(y_true, y_pred, target_names=["Não violência", "Violência"])
    print("=== Relatório de Classificação ===")
    print(report)

    os.makedirs("report", exist_ok=True)

    # Salva o relatório
    with open(f"report/{video_name}_metrics.txt", "w") as f:
        f.write(report)

    # Matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Não violência", "Violência"],
                yticklabels=["Não violência", "Violência"])
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(f"report/{video_name}_confusion.png")
    plt.close()

    # Salvar CSV com previsões
    df_preds = pd.DataFrame({"Real": y_true, "Previsto": y_pred})
    df_preds.to_csv(f"report/{video_name}_predictions.csv", index=False)

# ==== Terminal ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia um modelo treinado (LSTM/GRU/RNN) com novos dados.")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru", "rnn"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--violence", nargs="+", required=True)
    parser.add_argument("--nonviolence", nargs="+", required=True)
    parser.add_argument("--max_people", type=int, default=3)
    parser.add_argument("--video_name", type=str, default="ytvideo")
    parser.add_argument("--hidden_size", type=int, default=64)

    args = parser.parse_args()
    evaluate_model(args.model, args.model_path, args.violence, args.nonviolence, args.max_people, args.video_name, args.hidden_size)
