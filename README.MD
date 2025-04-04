# Sistema de Detecção de Violência Física em Vídeos de Câmeras de Segurança

Este projeto tem como objetivo desenvolver um sistema automatizado para detectar e classificar eventos de **violência física** em vídeos de vigilância, utilizando **estimativa de poses humanas com YOLO Pose** e **Redes Neurais Recorrentes (RNN, GRU, LSTM)**.

---

## 📂 Estrutura do Projeto

```
projeto-violencia-fisica/
├── data/                # Dados brutos, frames, keypoints e vetores
├── models/              # Checkpoints de modelos treinados (.pth)
├── report/              # Relatório final e análises
├── src/                 # Scripts principais do pipeline
├── requirements.txt     # Dependências do projeto
└── README.md
```

---

## 🧪 Etapas do Pipeline

1. **Extração de frames** dos vídeos (`1_extract_frames.py`)
2. **Estimativa de poses humanas** com YOLOv8 Pose (`2_yolo_pose_inference.py`)
3. **Construção de vetores de características** dos keypoints (`3_extract_features.py`)
   - Suporta múltiplas pessoas por frame com o argumento `--max_people`
4. **Classificação temporal** com modelos RNN, GRU ou LSTM (`4_train_model.py`)
5. **Avaliação do modelo** com métricas e matriz de confusão (`5_evaluate_model.py`)
6. (Opcional) **Interface Web** com Streamlit (em desenvolvimento)
7. **Execução completa automatizada** com `run_pipeline.py`

---

## 🚀 Como rodar localmente

### 1. Crie um ambiente virtual e ative:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# ou
source venv/bin/activate  # Linux/Mac
```

### 2. Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## 📼 Exemplo de uso com `run_pipeline.py`

### Pré-requisitos:
- Coloque os vídeos em `data/raw/` com nomes como:
  - `Fight1.mp4`, `Normal1.mp4`, `Fight2.mp4`, `Normal2.mp4`

### Comando único para processar tudo:

```bash
python src/run_pipeline.py ^
  --model lstm ^
  --fps 10 ^
  --max_people 5 ^
  --train_violence Fight1 ^
  --train_nonviolence Normal1 ^
  --test_violence Fight2 ^
  --test_nonviolence Normal2
```

---

## ⚙️ Parâmetros úteis

- `--model`: tipo de modelo (`lstm`, `gru` ou `rnn`)
- `--fps`: taxa de quadros para extração de frames
- `--max_people`: número máximo de pessoas consideradas por frame (default = 3)

---

## 📌 Tema

**Violência Física**

Este projeto utiliza vídeos do **Real-World Violence Dataset** e/ou outros contextos de vigilância para detectar comportamentos agressivos de forma automatizada.

---

## 📊 Métricas de exemplo (teste real)

- **Acurácia geral:** 92.8%
- **Recall (violência):** 100%
- **Falsos positivos:** 9
- **Falsos negativos:** 0

> O sistema é altamente sensível a eventos de violência, priorizando a não perda de detecções relevantes.

---

## ⚠️ Observações

- O vetor por frame é composto pelos keypoints de até `N` pessoas (`N × 34` valores).
- Se houver menos pessoas no frame, o vetor é preenchido com zeros.
- Se houver mais, apenas as primeiras `N` pessoas detectadas são consideradas.
- Isso permite entrada fixa para as redes neurais recorrentes.

---

## 👨‍💻 Desenvolvido por

[Victor Winicius Barros Santos](https://github.com/VictorW-dev)
