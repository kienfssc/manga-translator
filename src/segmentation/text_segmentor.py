import cv2
import numpy as np
from typing import List, Dict, Any
from src.segmentation.segmentor import BaseSegmentor


class TextSegmentor_B1(BaseSegmentor):
    def __init__(self, model_path: str, conf_threshold: float = 0.3):
        pass

    def load_model(self, model_path: str):
        pass

    def detect(self, image: np.ndarray):
        pass


if __name__ == "__main__":
    pass
