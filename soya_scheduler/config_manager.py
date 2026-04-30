import json
import os
import shutil
import tempfile
import threading
import numpy as np
from PIL import Image

_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_DIR, "node_info.json")

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}

_config_lock = threading.Lock()


def load_config():
    with _config_lock:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    return {
        "settings": {
            "base_path": "",
            "device": "cuda:1",
            "bbox_device": "cuda:1",
            "clip_device": "cuda:1",
            "bbox_model": "face_yolov8m.pt",
            "clip_vision_model": "clip_vision_vit_h.safetensors",
            "male_enhance_prompt": "mature male, (flat color:0.9)",
            "female_enhance_prompt": "drunkendream,hshi",
            "common_prompt": "",
            "unknown_gender": "boy",
            "unknown_eye_prompt": "unknown, black eyes",
            "bbox_threshold": 0.5,
            "face_crop_factor": 3.0,
            "ref_face_crop_factor": 3.0,
            "max_faces": 3,
            "tracking_face_count": 3,
            "keep_face_count": 3,
            "keep_matched_only": False,
            "target_size": 512,
            "detailer_face_crop_factor": 3.0,
            "upscale_model": "",
            "upscale_device": "cuda:1",
            "preprocess_upscale": False,
            "sam2_model": "",
            "grounding_dino_model": "",
            "segment_device": "cuda:1",
            "segment_prompt": "eyes",
            "segment_threshold": 0.3,
            "segs_min_distance": 0,
            "crop_mode": "preserve",
            "num_cpus": 1,
            "faceid_ipadapter_file": "",
            "faceid_weight": 1.0,
            "faceid_weight_faceidv2": 1.0,
            "faceid_start_at": 0.0,
            "faceid_end_at": 1.0,
            "faceid_embeds_scaling": "V only",
            "faceid_lora_strength": 1.0,
            "faceid_clip_vision_model": "",
            "faceid_mask_sigma_factor": 0.4,
            "mask_sigma_factor": 0.4,
            "faceid_crop_zoom": 1.0,
            "segment_method": "sam2",
            "eye_seg_model": "",
            "eye_seg_threshold": 0.5,
            "eyebrow_mode": "off",
            "eyebrow_threshold": 0.5,
            "eyebrow_model": "",
            "eyebrow_restore": False,
            "eyebrow_restore_mode": "hs_preserve",
            "eyebrow_blur": 0,
            "eyebrow_hs_percentile": 0.0,
            "eyebrow_v_range": 1.0,
            "eyebrow_opacity": 0.0,
            "closed_eye_threshold": 0.8,
            "save_face_data": False,
            "face_data_path": "face_data",
            "face_detailer_enabled": False,
            "face_surrounding_mode": "bbox_scale",
            "face_surrounding_pixels": 64,
            "face_bbox_scale_factor": 1.3,
            "face_upscale_model": "",
            "face_upscale_device": "cuda:1",
        },
        "last_process_result": None,
        "last_final_prompts": None,
    }


def save_config(data):
    with _config_lock:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".tmp", dir=os.path.dirname(CONFIG_PATH), prefix=".node_info_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, CONFIG_PATH)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


def _scan_model_dir(subdir, extensions=None):
    """Scan models/<subdir> for model files. Fallback when folder_paths has no entry."""
    try:
        import folder_paths
        models_dir = folder_paths.models_dir
    except Exception:
        return []
    d = os.path.join(models_dir, subdir)
    if not os.path.isdir(d):
        return []
    if extensions is None:
        extensions = {'.pt', '.pth', '.safetensors', '.bin', '.onnx'}
    return sorted(
        f for f in os.listdir(d)
        if os.path.isfile(os.path.join(d, f)) and os.path.splitext(f)[1].lower() in extensions
    )


def _folder_paths_list(folder_name):
    """Try folder_paths.get_filename_list, return [] on failure."""
    try:
        import folder_paths
        return folder_paths.get_filename_list(folder_name)
    except Exception:
        return []


def get_available_models(model_type):
    if model_type == "ultralytics_bbox":
        return _folder_paths_list("ultralytics_bbox")
    elif model_type == "clip_vision":
        return _folder_paths_list("clip_vision")
    elif model_type == "upscale_models":
        return _folder_paths_list("upscale_models")
    elif model_type == "sam2":
        result = _folder_paths_list("sam2")
        if not result:
            result = _scan_model_dir("sam2")
        return result
    elif model_type == "grounding_dino":
        result = _folder_paths_list("grounding-dino")
        if not result:
            result = _scan_model_dir("grounding-dino")
        return result
    elif model_type == "ipadapter":
        return _folder_paths_list("ipadapter")
    elif model_type == "eyebrow_seg":
        result = _scan_model_dir("soya_seg", extensions={'.ckpt', '.pth', '.safetensors'})
        if not result:
            result = _scan_model_dir("eyebrow_seg", extensions={'.ckpt', '.pth', '.safetensors'})
        return result
    elif model_type == "soya_seg":
        result = _scan_model_dir("soya_seg", extensions={'.ckpt', '.pth', '.safetensors'})
        if not result:
            result = _scan_model_dir("eyebrow_seg", extensions={'.ckpt', '.pth', '.safetensors'})
        return result
    return []


def get_available_devices():
    devices = ["cpu"]
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            encoding="utf-8", timeout=5,
        )
        for line in out.strip().split("\n"):
            idx = line.strip()
            if idx.isdigit():
                devices.append(f"cuda:{idx}")
    except Exception:
        pass
    return devices


# ── Per-character file management ───────────────────────────────

def find_faceid_embed(base_path, name):
    """Return path to {name}_faceid.safetensors if it exists, else None."""
    if not base_path or not os.path.isdir(base_path):
        return None
    p = os.path.join(base_path, f"{name}_faceid.safetensors")
    return p if os.path.isfile(p) else None


def delete_faceid_embed(base_path, name):
    """Delete faceid embed file if it exists."""
    p = find_faceid_embed(base_path, name)
    if p:
        os.remove(p)
        return True
    return False


def rename_faceid_embed(base_path, old_name, new_name):
    """Rename faceid embed file."""
    old_p = find_faceid_embed(base_path, old_name)
    if old_p:
        new_p = os.path.join(base_path, f"{new_name}_faceid.safetensors")
        shutil.move(old_p, new_p)
        return True
    return False


def find_image_file(base_path, name):
    """Find the image file for a character name in base_path."""
    if not base_path or not os.path.isdir(base_path):
        return None
    for ext in IMAGE_EXTENSIONS:
        filepath = os.path.join(base_path, f"{name}{ext}")
        if os.path.exists(filepath):
            return f"{name}{ext}"
    return None


def load_characters(base_path):
    """Load characters from base_path. Each image gets a corresponding JSON."""
    if not base_path or not os.path.isdir(base_path):
        return []

    characters = []
    seen_stems = set()

    for filename in sorted(os.listdir(base_path)):
        stem, ext = os.path.splitext(filename)
        if ext.lower() not in IMAGE_EXTENSIONS:
            continue
        if stem in seen_stems:
            continue
        seen_stems.add(stem)

        json_path = os.path.join(base_path, f"{stem}.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        else:
            info = {}

        characters.append({
            "name": info.get("name", stem),
            "gender": info.get("gender", "girl"),
            "eye_prompt": info.get("eye_prompt", ""),
            "image_file": filename,
            "has_faceid_embed": find_faceid_embed(base_path, stem) is not None,
        })

    return characters


def save_character_info(base_path, char_name, data):
    """Save character info to an individual JSON file."""
    os.makedirs(base_path, exist_ok=True)
    json_path = os.path.join(base_path, f"{char_name}.json")
    info = {
        "name": data.get("name", char_name),
        "gender": data.get("gender", "girl"),
        "eye_prompt": data.get("eye_prompt", ""),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


def delete_character(base_path, char_name):
    """Delete character image + JSON + faceid embed files."""
    if not base_path or not os.path.isdir(base_path):
        return False

    deleted = False
    json_path = os.path.join(base_path, f"{char_name}.json")
    if os.path.exists(json_path):
        os.remove(json_path)
        deleted = True

    for ext in IMAGE_EXTENSIONS:
        img_path = os.path.join(base_path, f"{char_name}{ext}")
        if os.path.exists(img_path):
            os.remove(img_path)
            deleted = True

    if delete_faceid_embed(base_path, char_name):
        deleted = True

    return deleted


def rename_character(base_path, old_name, new_name):
    """Rename character image + JSON + faceid embed files."""
    if not base_path or not os.path.isdir(base_path):
        return False

    renamed = False

    old_json = os.path.join(base_path, f"{old_name}.json")
    new_json = os.path.join(base_path, f"{new_name}.json")
    if os.path.exists(old_json):
        with open(old_json, "r", encoding="utf-8") as f:
            info = json.load(f)
        info["name"] = new_name
        with open(new_json, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        os.remove(old_json)
        renamed = True

    for ext in IMAGE_EXTENSIONS:
        old_img = os.path.join(base_path, f"{old_name}{ext}")
        if os.path.exists(old_img):
            new_img = os.path.join(base_path, f"{new_name}{ext}")
            shutil.move(old_img, new_img)
            renamed = True

    if rename_faceid_embed(base_path, old_name, new_name):
        renamed = True

    return renamed


def reload_characters(base_path):
    """Sync folder: delete orphan JSONs, create default JSONs for images without them."""
    if not base_path or not os.path.isdir(base_path):
        return []

    image_stems = set()
    for filename in os.listdir(base_path):
        stem, ext = os.path.splitext(filename)
        if ext.lower() in IMAGE_EXTENSIONS:
            image_stems.add(stem)

    for filename in list(os.listdir(base_path)):
        if filename.endswith('.json'):
            stem = filename[:-5]
            if stem not in image_stems:
                os.remove(os.path.join(base_path, filename))

    for stem in image_stems:
        json_path = os.path.join(base_path, f"{stem}.json")
        if not os.path.exists(json_path):
            save_character_info(base_path, stem, {
                "name": stem, "gender": "girl", "eye_prompt": ""
            })

    return load_characters(base_path)


def load_reference_image(base_path, image_file):
    """Load a reference image from base_path."""
    if not base_path or not image_file:
        return None
    filepath = os.path.join(base_path, image_file)
    if not os.path.exists(filepath):
        return None
    img = Image.open(filepath).convert("RGB")
    return np.array(img)
