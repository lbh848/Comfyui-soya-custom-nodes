import re
import json
import torch
import numpy as np

try:
    from impact.core import SEG
except ImportError:
    SEG = None


class DetailerDistributorPipe_mdsoya:
    """
    디테일러 디스트리뷰터 파이프

    1. 마스크 기반 합성으로 디테일 결과를 붙여넣기 (bbox 겹침 방지)
    2. 업데이트된 이미지에서 다음 영역을 크롭+업스케일
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "wildcard": ("STRING", {"multiline": True, "forceInput": True}),
                "original_image": ("IMAGE",),
                "detailed_image": ("IMAGE",),
                "modified_segs": ("SEGS",),
                "bbox_list": ("STRING", {"forceInput": True}),
                "crop_params": ("STRING", {"forceInput": True}),
                "upscale_model": ("UPSCALE_MODEL",),
                "paste_mask": ("MASK",),
                "node_id": ("INT", {"default": 1, "min": 0, "max": 999}),
            }
        }

    RETURN_TYPES = ("STRING", "SEGS", "IMAGE", "SEGS", "STRING", "STRING", "STRING", "IMAGE", "MASK")
    RETURN_NAMES = (
        "wildcard", "remaining_segs", "single_image", "single_segs",
        "modified_wildcard", "bbox_list", "crop_params", "updated_image", "paste_mask",
    )
    FUNCTION = "doit"
    CATEGORY = "Soya/Detailer"
    OUTPUT_NODE = True

    def doit(self, wildcard, original_image, detailed_image, modified_segs, bbox_list, crop_params, upscale_model, paste_mask, node_id):
        segs_list = modified_segs[1]
        crop_size = modified_segs[0][0]

        params = self._parse_crop_params(crop_params)
        bboxes = self._parse_bbox_list(bbox_list)

        # --- Step 1: Mask-based paste-back ---
        if bboxes:
            bbox = bboxes[0]
            updated_image = self._paste_back_masked(original_image, detailed_image, paste_mask, bbox)
            remaining_bboxes = bboxes[1:]
            print(f"[DetailerPipe #{node_id}] Pasted back at bbox={bbox} (masked)")
        else:
            updated_image = original_image.clone()
            remaining_bboxes = []

        # --- Step 2: Crop next from UPDATED image ---
        if segs_list and params:
            first_seg = segs_list[0]
            remaining_segs = (modified_segs[0], segs_list[1:])

            single_image = self._crop_and_upscale(updated_image, params[0], upscale_model, crop_size)

            # 모든 레벨: 업스케일 비율에 맞게 SEGS 좌표 조정
            orig_h = params[0]["cy2"] - params[0]["cy1"]
            orig_w = params[0]["cx2"] - params[0]["cx1"]
            scale_x = single_image.shape[2] / orig_w if orig_w > 0 else 1.0
            scale_y = single_image.shape[1] / orig_h if orig_h > 0 else 1.0
            first_seg = self._scale_seg_coords(first_seg, scale_x, scale_y)

            # SEGS 튜플 사이즈를 실제 이미지 크기에 맞춤
            actual_segs_size = (single_image.shape[2], single_image.shape[1])
            single_segs = (actual_segs_size, [first_seg])
            mask_h = single_image.shape[1]
            mask_w = single_image.shape[2]
            next_paste_mask = self._create_full_mask(first_seg, mask_h, mask_w)

            modified_wildcard = self._filter_wildcard_by_label(wildcard, first_seg.label)
            remaining_params = params[1:]

            print(f"[DetailerPipe #{node_id}] label={first_seg.label} "
                  f"level={params[0].get('level', '?')} remaining: {len(segs_list) - 1}")
        else:
            C = original_image.shape[3]
            single_image = torch.zeros((1, crop_size, crop_size, C), dtype=original_image.dtype)
            remaining_segs = ((crop_size, crop_size), [])
            single_segs = ((crop_size, crop_size), [])
            next_paste_mask = torch.ones((1, crop_size, crop_size), dtype=torch.float32)
            modified_wildcard = wildcard
            remaining_params = []

        bbox_str = ";".join(f"{x1},{y1},{x2},{y2}" for x1, y1, x2, y2 in remaining_bboxes)
        remaining_params_str = json.dumps(remaining_params)

        return (wildcard, remaining_segs, single_image, single_segs,
                modified_wildcard, bbox_str, remaining_params_str, updated_image, next_paste_mask)

    @staticmethod
    def _parse_bbox_list(bbox_str):
        if not bbox_str or not bbox_str.strip():
            return []
        result = []
        for entry in bbox_str.split(';'):
            parts = entry.split(',')
            if len(parts) == 4:
                try:
                    result.append(tuple(int(p) for p in parts))
                except ValueError:
                    continue
        return result

    @staticmethod
    def _parse_crop_params(crop_params_str):
        try:
            return json.loads(crop_params_str)
        except (json.JSONDecodeError, TypeError):
            return []

    @staticmethod
    def _paste_back_masked(original_image, detailed_image, paste_mask, bbox):
        """Mask-based paste-back: only overwrite pixels within the mask.

        detailed_image: [1, crop_size, crop_size, 3]
        paste_mask: [1, crop_size, crop_size]
        bbox: (x1, y1, x2, y2) in original image
        """
        x1, y1, x2, y2 = bbox
        bbox_h = y2 - y1
        bbox_w = x2 - x1

        if bbox_h <= 0 or bbox_w <= 0:
            return original_image.clone()

        result = original_image.clone()

        # Resize detailed image and mask to bbox size
        resized_detailed = torch.nn.functional.interpolate(
            detailed_image.permute(0, 3, 1, 2),
            size=(bbox_h, bbox_w), mode='bilinear', align_corners=False,
        ).permute(0, 2, 3, 1)

        # Resize mask to bbox size
        mask_4d = paste_mask.unsqueeze(1) if paste_mask.dim() == 3 else paste_mask.unsqueeze(0).unsqueeze(0)
        resized_mask = torch.nn.functional.interpolate(
            mask_4d,
            size=(bbox_h, bbox_w), mode='bilinear', align_corners=False,
        ).squeeze(0).squeeze(0)  # [bbox_h, bbox_w]
        mask_3 = resized_mask.unsqueeze(2)  # [bbox_h, bbox_w, 1] for broadcast

        # Mask-based blending: result = mask * detailed + (1-mask) * original
        original_region = result[:, y1:y2, x1:x2, :]
        blended = mask_3 * resized_detailed + (1 - mask_3) * original_region
        result[:, y1:y2, x1:x2, :] = blended

        return result

    @staticmethod
    def _crop_and_upscale(image, params, upscale_model, crop_size):
        level = params.get("level", 0)
        cx1, cy1 = params["cx1"], params["cy1"]
        cx2, cy2 = params["cx2"], params["cy2"]

        cropped = image[0:1, cy1:cy2, cx1:cx2, :]

        upscaled = _upscale_with_model(upscale_model, cropped)
        if level == 2:
            upscaled = _upscale_with_model(upscale_model, upscaled)

        return upscaled

    @staticmethod
    def _create_full_mask(seg, output_h, output_w):
        full_mask = torch.zeros((1, output_h, output_w), dtype=torch.float32)

        cropped_mask = seg.cropped_mask
        if cropped_mask is None:
            return full_mask

        if isinstance(cropped_mask, np.ndarray):
            cropped_mask = torch.from_numpy(cropped_mask).float()

        if cropped_mask.dim() == 3:
            cropped_mask = cropped_mask[0]

        cr_x1, cr_y1, cr_x2, cr_y2 = seg.crop_region
        cr_h = cr_y2 - cr_y1
        cr_w = cr_x2 - cr_x1

        if cr_h > 0 and cr_w > 0 and cropped_mask.numel() > 0:
            resized = torch.nn.functional.interpolate(
                cropped_mask.unsqueeze(0).unsqueeze(0),
                size=(cr_h, cr_w), mode='bilinear', align_corners=False
            ).squeeze(0).squeeze(0)
            full_mask[0, cr_y1:cr_y2, cr_x1:cr_x2] = resized

        return full_mask

    @staticmethod
    def _scale_seg_coords(seg, scale_x, scale_y):
        """업스케일 비율에 맞게 SEGS 좌표 조정"""
        cr = seg.crop_region
        bb = seg.bbox

        new_crop_region = (
            max(0, int(cr[0] * scale_x)),
            max(0, int(cr[1] * scale_y)),
            int(cr[2] * scale_x),
            int(cr[3] * scale_y),
        )
        new_bbox = (
            max(0, int(bb[0] * scale_x)),
            max(0, int(bb[1] * scale_y)),
            int(bb[2] * scale_x),
            int(bb[3] * scale_y),
        )
        return SEG(
            cropped_image=None,
            cropped_mask=seg.cropped_mask,
            confidence=seg.confidence,
            crop_region=new_crop_region,
            bbox=new_bbox,
            label=seg.label,
            control_net_wrapper=seg.control_net_wrapper,
        )

    @staticmethod
    def _filter_wildcard_by_label(wildcard, label):
        if not wildcard or not wildcard.strip():
            return wildcard
        lines = wildcard.strip().split('\n')
        result_lines = []
        found = False

        for line in lines:
            stripped = line.strip()
            match = re.match(r'\[(\d+)\]\s*', stripped)
            if match:
                if match.group(1) == str(label):
                    result_lines.append(line)
                    found = True
            else:
                result_lines.append(line)

        if not found:
            return wildcard

        return '\n'.join(result_lines)


def _upscale_with_model(upscale_model, image):
    import comfy.utils
    import comfy.model_management

    device = comfy.model_management.get_torch_device()
    upscale_model.to(device)
    in_img = image.movedim(-1, -3).to(device)

    tile = 512
    overlap = 32
    oom = True
    try:
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(
                    in_img, lambda a: upscale_model(a),
                    tile_x=tile, tile_y=tile, overlap=overlap,
                    upscale_amount=upscale_model.scale, pbar=pbar,
                )
                oom = False
            except comfy.model_management.OOM_EXCEPTION:
                tile //= 2
                if tile < 128:
                    raise
    finally:
        upscale_model.to("cpu")

    return torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
