import cv2
import numpy as np
import os

# Imports based on your project structure
from src.detection.manga_detector import TextDetector_YOLO
from src.segmentation.text_segmentor import TextSegmentor_B1


def create_overlay(image, mask, color=(0, 0, 255), alpha=0.5):
    """
    Overlays a binary mask onto an image with a specific color.
    Args:
        image: Original BGR image
        mask: Binary mask (0 or 255)
        color: BGR color for the overlay (default: Red)
        alpha: Transparency (0 to 1)
    """
    overlay = image.copy()
    # Apply the color to the areas where the mask is active
    overlay[mask > 0] = color
    # Blend the overlay with the original image
    output = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return output


def run_dev_pipeline(image_path, model_dir="checkpoints", dev_out="output_dev"):
    # 0. Setup Output Directories
    det_out_dir = os.path.join(dev_out, "text_detection")
    seg_out_dir = os.path.join(dev_out, "text_segmentation")
    os.makedirs(det_out_dir, exist_ok=True)
    os.makedirs(seg_out_dir, exist_ok=True)

    filename = os.path.basename(image_path)
    print(f"--- Processing: {filename} ---")

    # 1. Initialize Models
    print("Initializing models...")
    text_detector = TextDetector_YOLO(os.path.join(model_dir, "text_det_yolo.onnx"))
    text_segmenter = TextSegmentor_B1(os.path.join(model_dir, "text-segmentation.pth"))

    # 2. Load Image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Error: Could not load image at {image_path}")
        return

    h, w, _ = img_bgr.shape

    # 3. Detection Phase
    print("Step 1: Detecting text regions...")
    detections = text_detector.detect(img_bgr)

    # Visualize Bounding Boxes
    det_viz = img_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        # Draw box
        cv2.rectangle(det_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label/conf
        label = f"{det['conf']:.2f}"
        cv2.putText(
            det_viz, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

    cv2.imwrite(os.path.join(det_out_dir, f"det_{filename}"), det_viz)
    print(f"✅ Detection result saved to: {det_out_dir}")

    # 4. Segmentation Phase
    print("Step 2: Segmenting regions and creating overlay...")
    global_mask = np.zeros((h, w), dtype=np.uint8)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        roi = img_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Run segmentation on the detected box
        roi_mask = text_segmenter.segment(roi)

        # Merge ROI mask into global mask
        global_mask[y1:y2, x1:x2] = cv2.bitwise_or(global_mask[y1:y2, x1:x2], roi_mask)

    # Create the Visual Overlay (Red tint on text)
    seg_overlay = create_overlay(img_bgr, global_mask, color=(0, 0, 255), alpha=0.4)
    """
    print("Step 3: Inpainting (Skipped due to hardware)...")
    mask_buffer = cv2.cvtColor(global_mask, cv2.COLOR_GRAY2BGR)
    result_pil = inpainter.inpaint_with_crop(img_bgr, mask_buffer, margin=128)
    result_pil.save(os.path.join(dev_out, f"final_{filename}"))
    """

    # Save both the raw mask and the overlay for comparison
    cv2.imwrite(os.path.join(seg_out_dir, f"mask_{filename}"), global_mask)
    cv2.imwrite(os.path.join(seg_out_dir, f"overlay_{filename}"), seg_overlay)

    print(f"✅ Segmentation overlay saved to: {seg_out_dir}")
    print(f"--- Finished {filename} ---\n")


if __name__ == "__main__":
    # You can point this to a specific image or a folder
    IMAGE_TO_TEST = "sample/3ff69466-329e-4fd6-b307-e3e0237e320c.png"
    run_dev_pipeline(IMAGE_TO_TEST)
