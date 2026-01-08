import os
import requests
from dotenv import load_dotenv
from huggingface_hub import login, hf_hub_download

load_dotenv()
token = os.getenv("HF_TOKEN")

# Authenticate with Hugging Face
if token:
    login(token=token)
else:
    print("⚠️ Warning: HF_TOKEN not found in .env file")

# Setup Inpainting Model paths
inpainting_dir = "models/inpainting"
os.makedirs(inpainting_dir, exist_ok=True)
inpainting_model_path = os.path.join(inpainting_dir, "anime-manga-big-lama.pt")

INPAINTING_MODEL_URL = "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt"

# Download Inpainting Model if it doesn't exist
if not os.path.exists(inpainting_model_path):
    print("🚀 Downloading LaMa Anime model...")
    try:
        r = requests.get(INPAINTING_MODEL_URL, allow_redirects=True, stream=True)
        r.raise_for_status()
        with open(inpainting_model_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(" Inpainting model downloaded successfully.")
    except Exception as e:
        print(f" Error downloading Inpainting model: {e}")
else:
    print(f" Inpainting model already exists at: {inpainting_model_path}")

# Setup Detection Model paths
detection_dir = "models/detection"
os.makedirs(detection_dir, exist_ok=True)
final_onnx_path = os.path.join(detection_dir, "model.onnx")

# Download Detection Model from Hugging Face if it doesn't exist
if os.path.exists(final_onnx_path):
    print(f" Detection model already exists at: {final_onnx_path}")
else:
    print(f"🚀 Downloading Detection model from Hugging Face...")
    try:
        downloaded_path = hf_hub_download(
            repo_id="deepghs/AnimeText_yolo",
            filename="yolo12l_animetext/model.onnx",
            local_dir=detection_dir,
            local_dir_use_symlinks=False,
        )

        # Move the file to the root of the detection directory for cleaner access
        downloaded_expected_path = os.path.join(
            detection_dir, "yolo12l_animetext/model.onnx"
        )
        if os.path.exists(downloaded_expected_path):
            os.rename(downloaded_expected_path, final_onnx_path)
            # Clean up the extra subdirectory created by hf_hub_download
            os.rmdir(os.path.join(detection_dir, "yolo12l_animetext"))

        print(f" Detection model downloaded and configured: {final_onnx_path}")
    except Exception as e:
        print(f" Error downloading from Hugging Face: {e}")
