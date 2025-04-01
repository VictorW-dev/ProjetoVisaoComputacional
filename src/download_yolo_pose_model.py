# Modelo pré-treinado YOLO Pose. O yolov8n-pose.pt é leve e funciona bem pra começar

from ultralytics import YOLO

# Baixa e prepara o modelo YOLOv8n Pose
YOLO("yolov8n-pose.pt")

print("✅ Modelo YOLOv8n Pose baixado com sucesso.")
