import os
from dotenv import load_dotenv

# Load variables from a .env file if it exists
load_dotenv()


class Config:
    """Base config."""

    SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")
    DATABASE_URL = os.getenv("DATABASE_URL")
    DEBUG = False
    TESTING = False


class ModelWeightsConfig:
    """Links and paths for AI model weights."""

    INPAINTING_MODEL_URL = "https://github.com/daominhwysi/manga-translator/releases/download/weights/anime-manga-big-lama.pt"
    TEXT_SEGMENTATION_URL = "https://github.com/daominhwysi/manga-translator/releases/download/weights/text_seg_unet_b1.pth"
    TEXT_DETECTION_URL = "https://github.com/daominhwysi/manga-translator/releases/download/weights/text_det_yolo.onnx"
