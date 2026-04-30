"""
Model loading and caching for upscale, SAM2, and GroundingDINO models.
All models are loaded independently – no dependency on comfyui-sam2.
"""

import os
import torch
import numpy as np


def _resolve_model_path(folder_name, model_name):
    """Resolve model path via folder_paths, falling back to models/<folder_name>/."""
    import folder_paths
    path = folder_paths.get_full_path(folder_name, model_name)
    if path is not None:
        return path
    # Fallback: construct path directly from models_dir
    fallback = os.path.join(folder_paths.models_dir, folder_name, model_name)
    if os.path.isfile(fallback):
        return fallback
    return None


# ── Upscale model (spandrel) ──────────────────────────────────
_upscale_cache = {}  # key: model_path -> model


def get_upscale_model(model_name, device):
    """Load/cache an upscale model using spandrel (ComfyUI built-in pattern)."""
    model_path = _resolve_model_path("upscale_models", model_name)
    if model_path is None:
        raise FileNotFoundError(f"Upscale model not found: {model_name}")

    if model_path not in _upscale_cache:
        import comfy.utils
        from spandrel import ModelLoader, ImageModelDescriptor
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        model = ModelLoader().load_from_state_dict(sd).eval()
        if not isinstance(model, ImageModelDescriptor):
            raise Exception("Upscale model must be a single-image model.")
        _upscale_cache[model_path] = model

    model = _upscale_cache[model_path]
    model.to(device)
    return model


def upscale_image(model, image_tensor, device, tile=512, overlap=32):
    """Upscale a BHWC image tensor using tiled_scale.

    Args:
        model: spandrel ImageModelDescriptor
        image_tensor: (B, H, W, 3) float32 tensor in [0, 1]
        device: e.g. "cuda:1"
        tile: tile size for tiled_scale
        overlap: overlap for tiled_scale

    Returns:
        Upscaled (B, H', W', 3) float32 tensor in [0, 1]
    """
    import comfy.utils
    model.to(device)
    in_img = image_tensor.movedim(-1, -3).to(device)  # BHWC -> BCHW

    s = comfy.utils.tiled_scale(
        in_img,
        lambda a: model(a),
        tile_x=tile, tile_y=tile,
        overlap=overlap,
        upscale_amount=model.scale,
    )

    return torch.clamp(s.movedim(-3, -1), min=0, max=1.0).cpu()  # BCHW -> BHWC


# ── SAM2 model ────────────────────────────────────────────────
_sam2_cache = {}  # key: (model_path, device) -> sam2 model


def get_sam2_model(model_name, device):
    """Load/cache a SAM2 model using sam2 pip package."""
    model_path = _resolve_model_path("sam2", model_name)
    if model_path is None:
        raise FileNotFoundError(f"SAM2 model not found: {model_name}")

    key = (model_path, device)
    if key not in _sam2_cache:
        from sam2.build_sam import build_sam2
        basename = os.path.basename(model_path)
        name_no_ext = os.path.splitext(basename)[0]

        # Map checkpoint name to config
        _config_map = {
            "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
            "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
            "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "sam2_hiera_large": "configs/sam2/sam2_hiera_l.yaml",
            "sam2_hiera_base_plus": "configs/sam2/sam2_hiera_b+.yaml",
            "sam2_hiera_small": "configs/sam2/sam2_hiera_s.yaml",
            "sam2_hiera_tiny": "configs/sam2/sam2_hiera_t.yaml",
        }
        config = _config_map.get(name_no_ext)
        if config is None:
            # Fallback: try to guess from name patterns
            if "sam2.1" in name_no_ext:
                size_hint = "l" if "large" in name_no_ext else ("b+" if "base_plus" in name_no_ext else ("s" if "small" in name_no_ext else "t"))
                config = f"configs/sam2.1/sam2.1_hiera_{size_hint}.yaml"
            elif "sam2" in name_no_ext:
                size_hint = "l" if "large" in name_no_ext else ("b+" if "base_plus" in name_no_ext else ("s" if "small" in name_no_ext else "t"))
                config = f"configs/sam2/sam2_hiera_{size_hint}.yaml"

        sam = build_sam2(config, model_path, device=device)
        _sam2_cache[key] = sam

    return _sam2_cache[key]


def sam2_segment(sam_model, image_np, boxes):
    """Run SAM2 inference to generate masks for given boxes.

    Args:
        sam_model: SAM2 model
        image_np: (H, W, 3) uint8 numpy array
        boxes: list of [x1, y1, x2, y2] boxes

    Returns:
        list of (H, W) uint8 numpy masks, one per box
    """
    if not boxes:
        return []

    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(sam_model)
    predictor.set_image(image_np)

    masks_list = []
    for box in boxes:
        box_arr = np.array(box, dtype=np.float32)
        with torch.no_grad():
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_arr[None, :],  # (1, 4)
                multimask_output=False,
            )
        # masks: (1, H, W), take best mask
        best_mask = masks[0]  # (H, W) bool
        masks_list.append(best_mask.astype(np.uint8))

    return masks_list


# ── GroundingDINO model ───────────────────────────────────────
# Uses comfyui-sam2's vendored local_groundingdino for .pth + .cfg.py loading.
_dino_cache = {}  # key: (model_name, device) -> dino model

# Map checkpoint filename -> corresponding .cfg.py filename
_DINO_CKPT_TO_CFG = {
    "groundingdino_swinb_cogcoor.pth": "GroundingDINO_SwinB.cfg.py",
    "groundingdino_swint_ogc.pth": "GroundingDINO_SwinT_OGC.cfg.py",
}


def get_grounding_dino_model(model_name, device):
    """Load/cache a GroundingDINO model (.pth + .cfg.py format).

    Uses comfyui-sam2's vendored local_groundingdino for model construction.
    """
    key = (model_name, device)
    if key in _dino_cache:
        return _dino_cache[key]

    # Resolve model directory
    model_path = _resolve_model_path("grounding-dino", model_name)
    if model_path is None:
        raise FileNotFoundError(f"GroundingDINO model not found: {model_name}")
    model_dir = os.path.dirname(model_path)

    # Find config file
    cfg_name = _DINO_CKPT_TO_CFG.get(model_name)
    if cfg_name is None:
        # Try to guess: replace .pth with .cfg.py, or look for any .cfg.py
        cfg_name = model_name.replace(".pth", ".cfg.py")
    cfg_path = os.path.join(model_dir, cfg_name)
    if not os.path.isfile(cfg_path):
        # Search for any .cfg.py in the same directory
        cfg_files = [f for f in os.listdir(model_dir) if f.endswith(".cfg.py")]
        if cfg_files:
            cfg_path = os.path.join(model_dir, cfg_files[0])
        else:
            raise FileNotFoundError(
                f"GroundingDINO config (.cfg.py) not found in {model_dir}"
            )

    # Import from comfyui-sam2's vendored local_groundingdino
    # comfyui-sam2's node.py adds its dir to sys.path, making local_groundingdino importable
    try:
        from local_groundingdino.util.slconfig import SLConfig
        from local_groundingdino.models import build_model
        from local_groundingdino.util.utils import clean_state_dict
    except ImportError:
        raise ImportError(
            "GroundingDINO requires comfyui-sam2's local_groundingdino. "
            "Install comfyui-sam2 custom node first."
        )

    # Load config
    args = SLConfig.fromfile(cfg_path)

    # Handle BERT text encoder path
    if hasattr(args, 'text_encoder_type') and args.text_encoder_type == "bert-base-uncased":
        import folder_paths
        bert_local = os.path.join(folder_paths.models_dir, "bert-base-uncased")
        if os.path.isdir(bert_local):
            args.text_encoder_type = bert_local

    # Build model and load weights
    dino = build_model(args)
    checkpoint = torch.load(model_path, map_location="cpu")
    dino.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    dino.to(device=device)
    dino.eval()

    _dino_cache[key] = dino
    return dino


# ── InsightFace model ─────────────────────────────────────────
_insightface_cache = {}  # key: provider -> FaceAnalysis


# ── YOLO model (main process) ─────────────────────────────────
_yolo_main_cache = {}  # key: model_path -> YOLO model


def get_yolo_model(model_path, device):
    """Load/cache a YOLO model for face detection in the main process."""
    if model_path not in _yolo_main_cache:
        from ultralytics import YOLO
        _yolo_main_cache[model_path] = YOLO(model_path).to(device)
    return _yolo_main_cache[model_path]


def get_insightface_model(provider="CUDAExecutionProvider"):
    """Load/cache an InsightFace buffalo_l model."""
    key = provider
    if key not in _insightface_cache:
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface is required. Add it to requirements.txt and restart ComfyUI."
            )
        import folder_paths
        path = os.path.join(folder_paths.models_dir, "insightface")
        model = FaceAnalysis(name="buffalo_l", root=path, providers=[provider])
        model.prepare(ctx_id=0, det_size=(640, 640))
        _insightface_cache[key] = model
    return _insightface_cache[key]


# ── ISNet Segmentation (Soya Seg) ─────────────────────────────
# Both eyebrow and eye models use the same ISNetDIS architecture.
# Models are stored in models/soya_seg/.
_isnet_cache = {}  # key: (model_path, device) -> model


def _load_isnet_model(model_name, device):
    """Load/cache an ISNetDIS model from a Lightning checkpoint in models/soya_seg/.

    Args:
        model_name: filename in models/soya_seg/ (e.g. "eyebrow_seg.ckpt", "eye_seg.ckpt")
        device: torch device string
    """
    model_path = _resolve_model_path("soya_seg", model_name)
    if model_path is None:
        raise FileNotFoundError(f"ISNet model not found: {model_name}")

    key = (model_path, device)
    if key in _isnet_cache:
        return _isnet_cache[key]

    from .isnet_model import ISNetDIS

    state = torch.load(model_path, map_location="cpu")

    # Lightning checkpoint: state_dict keys have "net." prefix
    net_state = {}
    for k, v in state["state_dict"].items():
        if k.startswith("net."):
            net_state[k[4:]] = v

    model = ISNetDIS(in_ch=3, out_ch=1)
    model.load_state_dict(net_state)
    model.to(device)
    model.eval()

    _isnet_cache[key] = model
    return model


def get_eyebrow_model(model_name, device):
    """Load/cache the eyebrow ISNet model from models/soya_seg/."""
    return _load_isnet_model(model_name, device)


def get_eye_seg_model(model_name, device):
    """Load/cache the eye segmentation ISNet model from models/soya_seg/."""
    return _load_isnet_model(model_name, device)


def _isnet_segment(model, image_np, device, img_size=384):
    """Run ISNet segmentation on an image.

    Args:
        model: ISNetDIS model
        image_np: (H, W, 3) uint8 numpy array
        device: torch device string
        img_size: model input size (384 for the checkpoints)

    Returns:
        (H, W) float32 mask, values in [0, 1], 1.0 = detected region
    """
    from PIL import Image as PILImage

    h0, w0 = image_np.shape[:2]
    img = (image_np / 255).astype(np.float32)

    # Resize longest side to img_size, maintaining aspect ratio
    if h0 >= w0:
        h, w = img_size, int(img_size * w0 / h0)
    else:
        h, w = int(img_size * h0 / w0), img_size

    # Center-aligned zero-padding to (img_size, img_size)
    ph, pw = img_size - h, img_size - w
    img_resized = np.array(
        PILImage.fromarray((img * 255).astype(np.uint8)).resize(
            (w, h), PILImage.BILINEAR
        )
    ).astype(np.float32) / 255.0

    canvas = np.zeros((img_size, img_size, 3), dtype=np.float32)
    canvas[ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w] = img_resized

    # HWC -> CHW -> add batch dim
    tensor = torch.from_numpy(canvas.transpose(2, 0, 1)[np.newaxis]).float().to(device)

    is_cuda = isinstance(device, str) and device.startswith("cuda")
    with torch.no_grad():
        if is_cuda:
            with torch.autocast(device_type="cuda"):
                pred = model(tensor)[0][0].sigmoid()
            pred = pred.float().cpu()[0]  # (1, 384, 384)
        else:
            pred = model(tensor)[0][0].sigmoid().cpu()[0]  # (1, 384, 384)
    pred_np = pred.numpy().transpose(1, 2, 0)  # (img_size, img_size, 1)
    pred_np = pred_np[ph // 2 : ph // 2 + h, pw // 2 : pw // 2 + w]
    pred_resized = np.array(
        PILImage.fromarray((pred_np[:, :, 0] * 255).astype(np.uint8)).resize(
            (w0, h0), PILImage.BILINEAR
        )
    ).astype(np.float32) / 255.0

    return pred_resized


def eyebrow_segment(model, image_np, device, img_size=384):
    """Run eyebrow segmentation on an image. Returns (H, W) float32 mask in [0, 1]."""
    return _isnet_segment(model, image_np, device, img_size)


def eye_seg_segment(model, image_np, device, img_size=384):
    """Run eye segmentation on an image. Returns (H, W) float32 mask in [0, 1]."""
    return _isnet_segment(model, image_np, device, img_size)


def grounding_dino_predict(dino_model, image_pil, prompt, threshold):
    """Run GroundingDINO prediction.

    Args:
        dino_model: GroundingDINO model (from get_grounding_dino_model)
        image_pil: PIL Image
        prompt: text prompt (e.g. "eyes")
        threshold: box threshold

    Returns:
        list of [x1, y1, x2, y2] boxes in pixel coordinates
    """
    from torchvision.transforms.functional import to_tensor
    from torchvision.ops import box_convert

    W_orig, H_orig = image_pil.size

    # Resize so shorter side = 800, longer side <= 1333 (matches GroundingDINO training)
    min_side = 800
    max_side = 1333
    scale = min_side / min(W_orig, H_orig)
    if max(W_orig, H_orig) * scale > max_side:
        scale = max_side / max(W_orig, H_orig)
    new_W, new_H = int(W_orig * scale), int(H_orig * scale)
    image_resized = image_pil.resize((new_W, new_H))

    # Convert to tensor and apply ImageNet normalization
    img_tensor = to_tensor(image_resized).float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    device = next(dino_model.parameters()).device

    # Inline predict logic to avoid importing inference.py (which requires supervision)
    caption = prompt.lower().strip()
    if not caption.endswith("."):
        caption += "."

    dino_model = dino_model.to(device)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = dino_model(img_tensor[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]
    prediction_boxes = outputs["pred_boxes"].cpu()[0]

    mask = prediction_logits.max(dim=1)[0] > threshold
    boxes = prediction_boxes[mask]

    # Convert cxcywh -> xyxy pixel coords
    W, H = image_pil.size
    if boxes.numel() > 0:
        boxes = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")
        boxes[:, 0] *= W
        boxes[:, 1] *= H
        boxes[:, 2] *= W
        boxes[:, 3] *= H
        boxes_np = boxes.numpy()
        return [[int(b[0]), int(b[1]), int(b[2]), int(b[3])] for b in boxes_np]
    return []
