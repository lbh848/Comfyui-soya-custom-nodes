import torch
import numpy as np
from PIL import Image


class SoyaDetectAndCrop_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox_detector": ("BBOX_DETECTOR",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "crop_factor_top": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "crop_factor_bottom": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "detect_and_crop"
    CATEGORY = "Soya/Image"

    def detect_and_crop(self, image, bbox_detector, threshold, crop_factor_top, crop_factor_bottom):
        yolo = bbox_detector.bbox_model
        side_factor = (crop_factor_top + crop_factor_bottom) / 2

        results = []
        for img_tensor in image:
            img_np = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            W, H = pil_img.size

            detections = yolo(img_np, verbose=False)
            for result in detections:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < threshold:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    bw = x2 - x1
                    bh = y2 - y1
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    nx1 = max(0, int(cx - bw * side_factor / 2))
                    ny1 = max(0, int(cy - bh * crop_factor_top / 2))
                    nx2 = min(W, int(cx + bw * side_factor / 2))
                    ny2 = min(H, int(cy + bh * crop_factor_bottom / 2))

                    cropped = pil_img.crop((nx1, ny1, nx2, ny2))
                    cropped_np = np.array(cropped).astype(np.float32) / 255.0
                    results.append(torch.from_numpy(cropped_np).unsqueeze(0))

        return (results,)
