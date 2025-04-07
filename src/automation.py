import os
import itertools
import subprocess

# Parâmetros a serem testados (apenas fps10_t30_h128 e fps20_t30_h128)
fps_values = [10, 20]  # Alterando para 10 e 20 FPS
timesteps_values = [30]  # Mantendo apenas 30 timesteps
hidden_size_values = [128]  # Mantendo apenas hidden_size 128

# Caminho do diretório onde o código está sendo executado
current_dir = os.path.dirname(os.path.abspath(__file__))

# Caminhos completos para os vídeos
train_violence = r"Fight1"
train_nonviolence = r"Normal1"
test_violence = r"Fight2"
test_nonviolence = r"Normal2"

# Modelo base
model_type = "lstm"

# Diretório para salvar logs dos experimentos (opcional)
os.makedirs("experiments_logs", exist_ok=True)

# Definir as classes (violência e não violência) para balanceamento
class_1 = "violence"
class_2 = "nonviolence"
max_people = 5  # Máximo de pessoas por frame

# Loop apenas pelas combinações de fps10_t30_h128 e fps20_t30_h128
for fps, t, h in itertools.product(fps_values, timesteps_values, hidden_size_values):
    print(f"\n🔁 Rodando experimento: FPS={fps}, Timesteps={t}, HiddenSize={h}")

    # Criação do comando para rodar o pipeline
    cmd = [
        "python", "src\\run_pipeline.py",  # Comando para rodar o script
        "--model", model_type,
        "--fps", str(fps),  # Convertendo 'fps' para string
        "--train_violence", train_violence,
        "--train_nonviolence", train_nonviolence,
        "--test_violence", test_violence,
        "--test_nonviolence", test_nonviolence,
        "--max_people", str(max_people)
    ]

    # Nome do log
    log_file = f"experiments_logs/fps{fps}_t{t}_h{h}.log"

    # Executa e salva log
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"✅ Experimento salvo em {log_file}")
