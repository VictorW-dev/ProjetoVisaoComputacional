from pytubefix import YouTube
import os

class dv:
    def download_video(url):
        # Criação do objeto YouTube
        yt = YouTube(url, use_oauth=True)

        # Seleção do stream de maior resolução
        stream = yt.streams.get_highest_resolution()

        # Caminho para o diretório de saída
        output_path = 'frontend'  # Caminho para a pasta onde o vídeo será salvo
        
        # Verifica se a pasta 'frontend' existe, senão cria
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Download do vídeo
        stream.download(output_path=output_path)
        
        # Caminho completo para o vídeo baixado
        download_path = os.path.join(output_path, f"{yt.title}.mp4")
        return download_path
