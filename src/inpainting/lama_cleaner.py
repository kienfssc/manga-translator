from abc import ABC, abstractmethod
import os
import torch
import cv2
import numpy as np
from PIL import Image
import requests

MODEL_URL = "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt"

class LamaCleaner:
    """
    LaMa-based inpainting model for image restoration.
    """

    def __init__(self, model_path: str):
        if not self.download_model(model_path):
            raise RuntimeError("Failed to download the model.")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location="cpu").to(self.device)
        self.model.eval()

    def download_model(self, model_path: str):
        try:
            if not os.path.exists(model_path):
                r = requests.get(MODEL_URL, allow_redirects=True)
                with open(model_path, 'wb') as f:
                    f.write(r.content)

            return True
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False

    def convert_cv2_mask(self, mask_buffer):
        mask = cv2.cvtColor(mask_buffer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def get_bounding_box(self, mask, margin=128):
        coords = cv2.findNonZero(mask)
        x, y, w, h = cv2.boundingRect(coords)

        img_h, img_w = mask.shape

        x1 = max(x - margin, 0)
        y1 = max(y - margin, 0)
        x2 = min(x + w + margin, img_w)
        y2 = min(y + h + margin, img_h)

        crop_w = x2 - x1
        crop_h = y2 - y1
        new_w = (crop_w // 8) * 8
        new_h = (crop_h // 8) * 8
        x2, y2 = x1 + new_w, y1 + new_h

        return x1, y1, x2, y2

    def inpaint_with_crop(self, image_buffer, mask_buffer, margin=128):
        img_bgr = cv2.cvtColor(image_buffer, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = self.convert_cv2_mask(mask_buffer)

        x1, y1, x2, y2 = self.get_bounding_box(mask, margin)

        img_crop = img_rgb[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        img_t = torch.from_numpy(img_crop).permute(2,0,1).unsqueeze(0).float().to(self.device) / 255.0
        mask_t = torch.from_numpy(mask_crop).unsqueeze(0).unsqueeze(0).float().to(self.device) / 255.0

        with torch.no_grad():
            res_t = self.model(img_t, mask_t)

        res_crop = res_t[0].permute(1, 2, 0).cpu().numpy()
        res_crop = np.clip(res_crop * 255, 0, 255).astype(np.uint8)

        img_final = img_rgb.copy()
        mask_indices = mask_crop > 0

        crop_final = img_final[y1:y2, x1:x2]
        crop_final[mask_indices] = res_crop[mask_indices]
        img_final[y1:y2, x1:x2] = crop_final

        return Image.fromarray(img_final)

# result = inpaint_with_crop("anh_goc.jpg", "mask.png", margin=128)
# result.save("output_crop.png")
