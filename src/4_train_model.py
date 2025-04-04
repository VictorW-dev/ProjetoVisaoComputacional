import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import argparse

# ==== Parâmetros padrão ====
SEQ_LEN = 30
HIDDEN_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.001
KEYPOINTS_PER_PERSON = 34  # 17 keypoints (x, y)

# ==== Dataset Customizado ====
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

# ==== Treinamento ====
def train_model(model_type, violence_csvs, nonviolence_csvs, max_people):
    print(f"📦 Carregando dados para modelo: {model_type.upper()}")
    input_size = max_people * KEYPOINTS_PER_PERSON

    data_paths = violence_csvs + nonviolence_csvs
    labels = [1] * len(violence_csvs) + [0] * len(nonviolence_csvs)

    dataset = ViolenceDataset(data_paths, labels, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    if model_type == "lstm":
        model = LSTMClassifier(input_size, HIDDEN_SIZE)
    elif model_type == "gru":
        model = GRUClassifier(input_size, HIDDEN_SIZE)
    elif model_type == "rnn":
        model = RNNClassifier(input_size, HIDDEN_SIZE)
    else:
        raise ValueError("Modelo inválido. Escolha entre: lstm, gru, rnn")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("🚀 Iniciando treinamento...")
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == y_batch).sum().item()

        acc = correct / len(dataset)
        print(f"📊 Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f} - Acc: {acc:.4f}")

    model_path = f"models/{model_type}_violence.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Modelo salvo em: {model_path}")

# ==== Terminal ====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina um modelo RNN/GRU/LSTM para detecção de violência.")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "gru", "rnn"], help="Tipo de modelo")
    parser.add_argument("--violence", nargs="+", required=True, help="CSVs com violência")
    parser.add_argument("--nonviolence", nargs="+", required=True, help="CSVs sem violência")
    parser.add_argument("--max_people", type=int, default=3, help="Número máximo de pessoas por frame")

    args = parser.parse_args()
    train_model(args.model, args.violence, args.nonviolence, args.max_people)
