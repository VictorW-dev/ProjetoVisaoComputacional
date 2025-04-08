import streamlit as st
from process_youtube import process_video_from_youtube
import pandas as pd
import os
from PIL import Image

st.title("Detecção de Violência Física em Vídeos")

video_url = st.text_input("Cole o link do vídeo do YouTube")

if st.button("Analisar vídeo"):
    if video_url:
        st.info("🔄 Processando vídeo, isso pode levar alguns minutos...")
        video_name = process_video_from_youtube(video_url)
        st.success(f"✅ Vídeo processado com nome: {video_name}")

        # Caminhos esperados de saída
        metrics_path = f"report/{video_name}_metrics.txt"
        predictions_path = f"report/{video_name}_predictions.csv"
        confusion_path = f"report/{video_name}_confusion.png"

        if os.path.exists(metrics_path):
            st.subheader("📊 Métricas de Avaliação")
            with open(metrics_path, "r") as f:
                st.text(f.read())

        if os.path.exists(confusion_path):
            st.subheader("🧩 Matriz de Confusão")
            st.image(Image.open(confusion_path), caption="Matriz de Confusão", use_column_width=True)

        if os.path.exists(predictions_path):
            st.subheader("📈 Classes Previstas por Frame")
            df = pd.read_csv(predictions_path)
            st.dataframe(df)
    else:
        st.warning("⚠️ Por favor, cole um link válido.")
