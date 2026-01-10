import cv2
import numpy as np
import os
from tqdm import tqdm

# Imports based on your project structure
from src.detection.manga_detector import TextDetector_YOLO
from src.segmentation.text_segmentor import TextSegmentor_B1
from src.inpainting.lama_cleaner import LamaCleaner


def create_overlay(image, mask, color=(0, 0, 255), alpha=0.5):
    """Overlays a binary mask onto an image with a specific color."""
    overlay = image.copy()
    overlay[mask > 0] = color
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return output


def run_dev_pipeline(image_path, model_dir="checkpoints", dev_out="output_dev"):
    # 0. Setup Output Directories
    det_out_dir = os.path.join(dev_out, "text_detection")
    seg_out_dir = os.path.join(dev_out, "text_segmentation")
    inp_out_dir = os.path.join(dev_out, "inpainting_results")

    os.makedirs(det_out_dir, exist_ok=True)
    os.makedirs(seg_out_dir, exist_ok=True)
    os.makedirs(inp_out_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    print(f"\n--- Processing: {filename} ---")

    # 1. Initialize Models
    print("Initializing models...")
    text_detector = TextDetector_YOLO(os.path.join(model_dir, "text_det_yolo.onnx"))
    text_segmenter = TextSegmentor_B1(os.path.join(model_dir, "text-segmentation.pth"))
    inpainter = LamaCleaner(os.path.join(model_dir, "anime-manga-big-lama.pt"))

    # 2. Load Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Error: Could not load image at {image_path}")
        return

    h, w, _ = img_bgr.shape
    inpainted_res = img_bgr.copy()

    # 3. Detection Phase
    print("Step 1: Detecting text regions...")
    detections = text_detector.detect(img_bgr)
    print(f"Found {len(detections)} text regions.")

    # 4. Segmentation & Inpainting Phase
    print("Step 2 & 3: Segmenting, Dilating, and Inpainting...")
    global_mask = np.zeros((h, w), dtype=np.uint8)

    # Define dilation kernel (3x3 or 5x5 usually works best for text)
    # A 3x3 kernel with 2 iterations is generally safer for manga
    kernel = np.ones((3, 3), np.uint8)

    for det in tqdm(detections, desc="Cleaning regions", unit="roi"):
        x1, y1, x2, y2 = map(int, det["box"])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        roi_img = img_bgr[y1:y2, x1:x2]
        if roi_img.size == 0:
            continue

        # A. Segment ROI
        roi_mask = text_segmenter.segment(roi_img)

        # B. Dilate Mask (Expand mask to capture text edges/anti-aliasing)
        # iterations=2 expands the mask by roughly 2-3 pixels in all directions
        roi_mask_dilated = cv2.dilate(roi_mask, kernel, iterations=2)

        # Merge dilated mask into global mask for visualization
        global_mask[y1:y2, x1:x2] = cv2.bitwise_or(
            global_mask[y1:y2, x1:x2], roi_mask_dilated
        )

        # C. Inpaint ROI
        roi_mask_3ch = cv2.cvtColor(roi_mask_dilated, cv2.COLOR_GRAY2BGR)
        try:
            # We inpaint the ROI using the dilated mask
            inpainted_roi = inpainter.inpaint_with_crop(
                roi_img, roi_mask_3ch, margin=32
            )
            inpainted_res[y1:y2, x1:x2] = inpainted_roi
        except Exception:
            continue

    # 5. Save Results
    # 5.1 Save Detection Visualization
    det_viz = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        cv2.rectangle(det_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(det_out_dir, f"det_{filename}"), det_viz)

    # 5.2 Save Segmentation Visualization (with dilation)
    seg_overlay = create_overlay(img_bgr, global_mask, color=(0, 0, 255), alpha=0.4)
    cv2.imwrite(os.path.join(seg_out_dir, f"mask_{filename}"), global_mask)
    cv2.imwrite(os.path.join(seg_out_dir, f"overlay_{filename}"), seg_overlay)

    # 5.3 Save Final Cleaned image
    final_out_path = os.path.join(inp_out_dir, f"clean_{filename}")
    cv2.imwrite(final_out_path, inpainted_res)

    print(f"✅ Finished! Result saved to: {final_out_path}")


if __name__ == "__main__":
    IMAGE_TO_TEST = "sample/3ff69466-329e-4fd6-b307-e3e0237e320c.png"
    run_dev_pipeline(IMAGE_TO_TEST)
