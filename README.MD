# Sistema de Detecção de Violência Física em Vídeos de Câmeras de Segurança

Este projeto tem como objetivo desenvolver um sistema automatizado para detectar e classificar eventos de **violência física** em vídeos de vigilância, utilizando **estimativa de poses humanas com YOLO Pose** e **Redes Neurais Recorrentes (RNN, GRU, LSTM)**.

---

## 📂 Estrutura do Projeto

```
projeto-violencia-fisica/
├── data/                # Dados brutos e processados
├── models/              # Checkpoints e resultados
├── report/              # Relatório final
├── src/                 # Scripts principais do pipeline
└── README.md
```

---

## 🧪 Etapas do Pipeline

1. **Extração de frames** dos vídeos (`1_extract_frames.py`)
2. **Estimativa de poses** com YOLOv8 Pose (`2_yolo_pose_inference.py`)
3. **Construção de vetores de características** dos keypoints
4. **Classificação** usando modelos RNN/GRU/LSTM
5. (Opcional) **Interface Web** com Streamlit

---

## 🚀 Como rodar localmente

### 1. Crie um ambiente virtual e ative:

```bash
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate no Windows
```

### 2. Instale as dependências:

```bash
pip install -r requirements.txt
```

### 3. Gere um vídeo de teste:

```bash
python src/generate_dummy_video.py
```

### 4. Extraia os frames:

```bash
python src/1_extract_frames.py --fps 10
```

### 5. Rode YOLO Pose:

```bash
python src/2_yolo_pose_inference.py --input_dir ... --output_dir ...
```

## Frontend:

### 1. Install

`pip install streamlit==1.44.1 pytubefix==8.12.3`

### 2. Player

`streamlit run front.py`

---

## 📌 Tema

Violência Física

Este projeto utiliza o Real-World Violence Dataset para detectar comportamentos agressivos em ambientes monitorados por vídeo.




---

## 👨‍💻 Desenvolvido por

[Victor Winicius Barros Santos
](https://github.com/VictorW-dev)

[
](https://github.com/VictorW-dev)

[
](https://github.com/VictorW-dev)
