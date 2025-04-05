import streamlit as st
import os
from frontend.downloadVideo import dv

st.title("Identificador de Violência Física em Vídeos")
video_url = st.text_input("Insira o link do vídeo do YouTube:")

def on_progress(d, chunk, file_handle, bytes_remaining):
    if d['status'] == 'downloading':
        percent = (d['downloaded_bytes'] / d['total_bytes']) * 100
        print(f"\rBaixando: {percent:.2f}%", end='')
    elif d['status'] == 'finished':
        print(f"\nDownload concluído! Salvando em {d['filename']}")

if st.button("Baixar Vídeo"):
    print(video_url)
    if video_url:
        download_result = dv.download_video(video_url)
        if os.path.exists(download_result):
            st.session_state['video_path'] = download_result
            st.success(f"Download concluído! Vídeo salvo em: {download_result}")
        else:
            st.error(f"Erro ao baixar o vídeo: {download_result}")
    else:
        st.warning("Por favor, insira um link válido.")

# Simulação do processamento do vídeo
def processar_video(video_path):
    return 0  # ou 1, dependendo do resultado do processamento

if 'video_path' in st.session_state:
    resultado = processar_video(st.session_state['video_path'])

    if resultado == 0:
        st.markdown("<style>.stApp { background-color: green; }</style>", unsafe_allow_html=True)
    elif resultado == 1:
        st.markdown("<style>.stApp { background-color: red; }</style>", unsafe_allow_html=True)

    # Exibir vídeo na tela
    with open(st.session_state['video_path'], 'rb') as video_file:
        st.video(video_file.read())
