import os
import zipfile
import gdown

def download_and_extract():
    url = "https://drive.google.com/uc?id=1jJz-VnGRRz87ZhD0JNu45naLZfUjWEzF"  # ID da pasta/arquivo do dataset no GDrive
    output = "data/raw/RealWorldViolenceDataset.zip"

    os.makedirs("data/raw", exist_ok=True)

    print("🔽 Baixando Real-World Violence Dataset...")
    gdown.download(url, output, quiet=False)

    print("📦 Extraindo...")
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall("data/raw")

    print("✅ Dataset pronto em: data/raw")

if __name__ == "__main__":
    download_and_extract()
