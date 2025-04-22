import os
from pathlib import Path

# Caminho onde estão os vídeos (ajuste se necessário)
pasta_videos = Path("data/raw")

# Contadores
contador = {
    "Normal": 1,
    "Violence": 1
}

# Processa os arquivos
for arquivo in pasta_videos.glob("*.mp4"):
    nome = arquivo.stem.lower()

    if "normal" in nome:
        novo_nome = f"Normal{contador['Normal']}.mp4"
        contador["Normal"] += 1
    elif "fight" in nome or "violence" in nome:
        novo_nome = f"Violence{contador['Violence']}.mp4"
        contador["Violence"] += 1
    else:
        print(f"Ignorado: {arquivo.name}")
        continue

    novo_caminho = pasta_videos / novo_nome
    print(f"Renomeando: {arquivo.name} → {novo_nome}")
    os.rename(arquivo, novo_caminho)

print("✅ Renomeação concluída.")