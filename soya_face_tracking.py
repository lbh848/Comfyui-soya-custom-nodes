"""CPU face observations from Anima's linear x0 preview, with no extra steps."""
import itertools
import math
import os
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .soya_spatial_face_context import expand_face_box


TRACKED_REGION_MASKS = "soya_tracked_region_masks"
_detector_lock = threading.RLock()
_detector_cache = None


def _detect(image):
    """Reuse only the CPU detector. Never download a model during sampling."""
    global _detector_cache
    import folder_paths

    model_path = Path(folder_paths.models_dir) / "ultralytics" / "bbox" / "face_yolov8n.pt"
    if not model_path.is_file():
        for root in folder_paths.get_folder_paths("ultralytics"):
            candidate = Path(root) / "bbox" / "face_yolov8n.pt"
            if candidate.is_file():
                model_path = candidate
                break
    stamp = (str(model_path), model_path.stat().st_mtime_ns)
    with _detector_lock:
        if _detector_cache is None or _detector_cache[0] != stamp:
            config_dir = Path(folder_paths.get_temp_directory()) / "soya_face_detector"
            config_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
            from ultralytics import YOLO
            detector = YOLO(str(model_path)).to("cpu")
            _detector_cache = stamp, detector
            print(f"[FACE_TRACK] CPU detector loaded: {model_path}")
        prediction = _detector_cache[1].predict(
            image, device="cpu", imgsz=640, conf=0.25, verbose=False,
        )[0]
        return prediction.boxes.xyxy.cpu().numpy().tolist()


def associate(boxes, regions, previous, width, height):
    """Minimize source-mask distance first; use temporal overlap only to break ties."""
    count = min(len(boxes), len(regions))
    if not count:
        print("[FACE_TRACK] No detected face; retaining preceding geometry")
        return {}

    def cost(box, subject):
        x, y = (box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height)
        rx, ry, rw, rh = regions[subject]
        geometric = (max(rx - x, 0, x - rx - rw) / rw
                     + max(ry - y, 0, y - ry - rh) / rh)
        if subject not in previous:
            return geometric, 1
        a, b = box, previous[subject]
        intersection = (max(0, min(a[2], b[2]) - max(a[0], b[0]))
                        * max(0, min(a[3], b[3]) - max(a[1], b[1])))
        union = ((a[2] - a[0]) * (a[3] - a[1])
                 + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
        return geometric, 1 - intersection / max(union, 1)

    candidates = (
        ([cost(boxes[b], s) for b, s in zip(chosen, subjects)],
         {s: boxes[b] for b, s in zip(chosen, subjects)})
        for chosen in itertools.combinations(range(len(boxes)), count)
        for subjects in itertools.permutations(range(len(regions)), count)
    )
    return min(candidates, key=lambda item: (
        sum(score[0] for score in item[0]),
        sum(score[1] for score in item[0]),
    ))[1]


def observation_steps(total):
    # Four proven checkpoints, scaled to the existing schedule. Leave a next
    # step in which the observation can actually affect the model.
    if total < 2:
        return set()
    return {min(total - 2, max(0, round(total * fraction) - 1))
            for fraction in (5 / 30, 12 / 30, 17 / 30, 28 / 30)}


class FaceTracker:
    def __init__(self, masks, latent_format, detector=None):
        self.detector = detector or _detect
        self.factors = np.asarray(latent_format.latent_rgb_factors, dtype=np.float32)
        bias = getattr(latent_format, "latent_rgb_factors_bias", None)
        self.bias = np.asarray(bias if bias is not None else [0, 0, 0], dtype=np.float32)
        self.scale = getattr(latent_format, "spacial_downscale_ratio", 8)
        self.regions = []
        for mask in masks:
            array = mask.detach().cpu().numpy().reshape(mask.shape[-2:])
            ys, xs = np.nonzero(array > 1e-6)
            if not len(xs):
                raise ValueError("Face tracking requires nonempty source masks")
            h, w = array.shape
            self.regions.append((xs.min() / w, ys.min() / h,
                                 (xs.max() - xs.min() + 1) / w,
                                 (ys.max() - ys.min() + 1) / h))
        self.faces = {}
        self.records = []
        self.epoch = 0
        self.cache_key = None
        self.cached = None
        self.last_weights = None
        self.width = self.height = None
        self.disabled = False

    def clear_device_cache(self):
        self.cache_key = self.cached = None

    def observe(self, step, x0):
        if self.disabled:
            return
        started = time.perf_counter()
        try:
            if x0.ndim != 5 or x0.shape[0] != 1 or x0.shape[2] != 1:
                self.disabled = True
                print(f"[FACE_TRACK] Unsupported batch/video shape={tuple(x0.shape)}; using source masks")
                return
            x = x0.detach().to(device="cpu", dtype=torch.float32).numpy()
            self.height, self.width = x.shape[-2] * self.scale, x.shape[-1] * self.scale
            rgb = x[0, :, 0].transpose(1, 2, 0) @ self.factors + self.bias
            image = Image.fromarray(np.clip((rgb + 1) * 127.5, 0, 255).astype(np.uint8)).resize(
                (self.width, self.height), Image.Resampling.BILINEAR,
            )
            boxes = self.detector(image)
            assigned = associate(boxes, self.regions, self.faces, self.width, self.height)
            if assigned:
                self.faces.update(assigned)
                self.epoch += 1
                self.clear_device_cache()
            row = {"step": step, "boxes": boxes, "assigned": assigned,
                   "seconds": time.perf_counter() - started}
            self.records.append(row)
            print(f"[FACE_TRACK] {row}")
        except Exception as exc:
            self.disabled = True
            print(f"[FACE_TRACK] Observation disabled: step={step}, shape={tuple(x0.shape)}, error={exc}")
            traceback.print_exc()

    def layout(self, original, lengths, shape, dit, device, dtype, build_bias):
        if shape[0] != 1 or not self.faces:
            return None
        height = math.ceil(shape[-2] / dit.patch_spatial)
        width = math.ceil(shape[-1] / dit.patch_spatial)
        key = (self.epoch, height, width, tuple(lengths), device, dtype)
        if key == self.cache_key:
            return self.cached
        weights = original.clone()
        cell = dit.patch_spatial * self.scale
        px = ((torch.arange(width, device=device, dtype=torch.float32) + .5) * cell).repeat(height)
        py = ((torch.arange(height, device=device, dtype=torch.float32) + .5) * cell).repeat_interleave(width)
        coverage = torch.zeros_like(original[1:], dtype=torch.bool)
        cores = torch.zeros_like(coverage)
        scores = torch.full(coverage.shape, float("inf"), device=device, dtype=torch.float32)
        face_scores = torch.full_like(scores, float("inf"))
        centerlines = {}
        for subject, box in sorted(self.faces.items()):
            x0, y0, x1, y1 = box
            crop_left, crop_top, crop_right, crop_bottom = expand_face_box(
                box, self.width, self.height, crop_top=2.5, crop_bottom=1.0,
            )
            rx, ry, rw, rh = self.regions[subject]
            left = max(0, math.floor(min(rx * self.width, crop_left) / cell))
            right = min(width, math.ceil(max((rx + rw) * self.width, crop_right) / cell))
            top = max(0, math.floor(min(ry * self.height, crop_top) / cell))
            bottom = min(height, math.ceil(max((ry + rh) * self.height, crop_bottom) / cell))
            coverage[subject].view(height, width)[top:bottom, left:right] = True

            # Distances use image pixels, including the actual padded token grid.
            fx, fy = (x0 + x1) / 2, (y0 + y1) / 2
            bx, by = (rx + rw / 2) * self.width, (ry + rh / 2) * self.height
            vx, vy = bx - fx, by - fy
            projection = (((px - fx) * vx + (py - fy) * vy) / max(vx * vx + vy * vy, 1)).clamp(0, 1)
            distance = (px - fx - projection * vx) ** 2 + (py - fy - projection * vy) ** 2
            scores[subject] = torch.where(coverage[subject], distance, float("inf"))
            centerlines[subject] = ((fx, fy), (bx, by))

            guard = torch.zeros((height, width), device=device, dtype=torch.bool)
            guard[crop_top // cell:min(height, math.ceil(crop_bottom / cell)),
                  crop_left // cell:min(width, math.ceil(crop_right / cell))] = True
            face_scores[subject] = torch.where(guard.flatten(), (px - fx) ** 2 + (py - fy) ** 2, float("inf"))
            cores[subject].view(height, width)[
                max(0, math.floor(y0 / cell)):min(height, math.ceil(y1 / cell)),
                max(0, math.floor(x0 / cell)):min(width, math.ceil(x1 / cell))] = True

        minimum, owners = scores.min(0)
        source_owner = original[1:].argmax(0)
        source_score = scores.gather(0, source_owner.unsqueeze(0)).squeeze(0)
        owners = torch.where(source_score == minimum, source_owner, owners)
        # An observed face outranks another face's padding and every body box.
        face_scores.masked_fill_(cores.any(0).unsqueeze(0) & ~cores, float("inf"))
        face_minimum, face_owners = face_scores.min(0)
        guarded = torch.isfinite(face_minimum)
        owners = torch.where(guarded, face_owners, owners)
        protected = ((original[1:] > 1e-6) & ~coverage).any(0)
        take = torch.isfinite(minimum) & (~protected | guarded)
        weights[:, take] = 0
        changed = {}
        for subject in self.faces:
            weights[subject + 1, take & (owners == subject)] = 1
            changed[subject] = int((weights[subject + 1] != original[subject + 1]).sum().item())
        bias = build_bias(weights > 1e-6, lengths, "uncovered_only", device, dtype)
        self.cache_key, self.cached = key, (weights, bias)
        self.last_weights = weights.detach().float().cpu().numpy().reshape(-1, height, width)
        print(f"[FACE_TRACK_MASK] epoch={self.epoch}, centerlines={centerlines}, changed_tokens={changed}")
        return self.cached

    def render_masks(self, size):
        if self.last_weights is None:
            return ()
        # Expand the actual patch cells, then crop padding added by the sampler.
        weights = self.last_weights[1:]
        cell_w = math.ceil(self.width / weights.shape[-1])
        cell_h = math.ceil(self.height / weights.shape[-2])
        pixels = weights.repeat(cell_h, axis=-2).repeat(cell_w, axis=-1)
        width, height = size
        return torch.from_numpy(pixels[:, :height, :width]).split(1)

    def export(self, masks):
        if not masks:
            print("[FACE_TRACK] No dynamic mask applied; retaining source preview")
            return None
        import folder_paths
        try:
            rgb = np.zeros((*masks[0].shape[-2:], 3), dtype=np.uint8)
            for index, mask in enumerate(masks):
                rgb[:, :, index] = np.clip(mask[0].numpy() * 255, 0, 255).astype(np.uint8)
            subfolder = "soya_dynamic_masks"
            directory = Path(folder_paths.get_temp_directory()) / subfolder
            directory.mkdir(parents=True, exist_ok=True)
            filename = f"mask_{uuid.uuid4().hex}.png"
            Image.fromarray(rgb).save(directory / filename, format="PNG")
            return {"filename": filename, "subfolder": subfolder, "type": "temp"}
        except Exception as exc:
            print(f"[FACE_TRACK] Mask preview export failed: size={masks[0].shape[-2:]}, error={exc}")
            traceback.print_exc()
            return None
