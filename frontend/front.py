import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import os
import uuid
import re
from frontend.process_youtube import process_video_from_youtube

st.set_page_config(page_title="Violência Física", layout="centered")

st.markdown("## 🛡️ Detecção de Violência Física em Vídeos")
st.markdown("### 🎥 Opções de vídeo")

video_source = st.radio("Escolha a fonte do vídeo", ["YouTube", "Vídeo Local"])

if video_source == "YouTube":
    youtube_url = st.text_input("Cole o link do vídeo do YouTube")

    if st.button("🔍 Analisar vídeo"):
        if not youtube_url:
            st.warning("Por favor, insira uma URL do YouTube.")
        else:
            with st.spinner("📥 Baixando e processando vídeo..."):
                try:
                    result = process_video_from_youtube(youtube_url, is_local=False)
                    st.success("✅ Análise concluída!")
                    st.info(f"📊 Resultado: {'Violência detectada' if result == 1 else 'Sem violência detectada'}")
                except Exception as e:
                    st.error("❌ Ocorreu um erro durante o processamento.")
                    st.error(str(e))

elif video_source == "Vídeo Local":
    uploaded_file = st.file_uploader("📁 Faça upload de um vídeo local (.mp4)", type=["mp4"])
    if uploaded_file is not None:
        # Limpa nome do vídeo
        raw_name = os.path.splitext(uploaded_file.name)[0]
        video_name = re.sub(r"[^\w\-]", "", raw_name)  # Remove parênteses, espaços etc
        clean_filename = f"{video_name}.mp4"

        # Salva na pasta data/raw com nome limpo
        save_path = os.path.join("data", "raw", clean_filename)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.read())

        if st.button("🔍 Analisar vídeo"):
            with st.spinner("📥 Processando vídeo local..."):
                try:
                    result = process_video_from_youtube(video_name, is_local=True)
                    st.success(f"✅ Vídeo processado: {video_name}")
                    st.info(f"📊 Resultado: {'Violência detectada' if result == 1 else 'Sem violência detectada'}")
                except Exception as e:
                    st.error("❌ Ocorreu um erro durante o processamento.")
                    st.error(str(e))
