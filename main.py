from src.inpainting.lama_cleaner import LamaCleaner
import cv2

inpainting = LamaCleaner(model_path="anime-manga-big-lama.pt")

image_buffer = cv2.imread("./sample/3ff69466-329e-4fd6-b307-e3e0237e320c.png", cv2.IMREAD_UNCHANGED)
print(image_buffer.shape)
mask_buffer = cv2.imread("./sample/binary_mask_1.png", cv2.IMREAD_UNCHANGED)
inpainted = inpainting.inpaint_with_crop(
    image_buffer,
    mask_buffer,
    margin=128
)

print(inpainted)

inpainted.save("./sample/output.png")