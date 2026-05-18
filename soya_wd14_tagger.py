import comfy.utils
import csv
import numpy as np
import os
import urllib.request
import folder_paths
from onnxruntime import InferenceSession
from PIL import Image

# Use existing WD14 tagger models from pythongosssss extension
_WD14_EXT_MODELS = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "comfyui-wd14-tagger", "models"
))

if "wd14_tagger" in folder_paths.folder_names_and_paths:
    models_dir = folder_paths.get_folder_paths("wd14_tagger")[0]
elif os.path.isdir(_WD14_EXT_MODELS):
    models_dir = _WD14_EXT_MODELS
else:
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)

# Model cache: model_name -> InferenceSession
_model_cache = {}

ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

HF_MODELS = {
    "wd-eva02-large-tagger-v3": "SmilingWolf/wd-eva02-large-tagger-v3",
    "wd-vit-tagger-v3": "SmilingWolf/wd-vit-tagger-v3",
    "wd-swinv2-tagger-v3": "SmilingWolf/wd-swinv2-tagger-v3",
    "wd-convnext-tagger-v3": "SmilingWolf/wd-convnext-tagger-v3",
    "wd-v1-4-moat-tagger-v2": "SmilingWolf/wd-v1-4-moat-tagger-v2",
    "wd-v1-4-convnextv2-tagger-v2": "SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
    "wd-v1-4-convnext-tagger-v2": "SmilingWolf/wd-v1-4-convnext-tagger-v2",
    "wd-v1-4-convnext-tagger": "SmilingWolf/wd-v1-4-convnext-tagger",
    "wd-v1-4-vit-tagger-v2": "SmilingWolf/wd-v1-4-vit-tagger-v2",
    "wd-v1-4-swinv2-tagger-v2": "SmilingWolf/wd-v1-4-swinv2-tagger-v2",
    "wd-v1-4-vit-tagger": "SmilingWolf/wd-v1-4-vit-tagger",
}


def _get_installed_models():
    if not os.path.isdir(models_dir):
        return []
    models = [m for m in os.listdir(models_dir)
              if m.endswith(".onnx")
              and os.path.exists(os.path.join(models_dir, os.path.splitext(m)[0] + ".csv"))]
    return [os.path.splitext(m)[0] for m in models]


def _download_model(model_name):
    if model_name not in HF_MODELS:
        raise ValueError(f"Unknown WD14 model: {model_name}. Available: {list(HF_MODELS.keys())}")

    hf_endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    if not hf_endpoint.startswith("https://"):
        hf_endpoint = f"https://{hf_endpoint}"

    repo = HF_MODELS[model_name]
    base_url = f"{hf_endpoint}/{repo}/resolve/main"

    onnx_path = os.path.join(models_dir, model_name + ".onnx")
    csv_path = os.path.join(models_dir, model_name + ".csv")

    if not os.path.exists(onnx_path):
        print(f"[Soya WD14] Downloading {model_name}.onnx ...")
        urllib.request.urlretrieve(f"{base_url}/model.onnx", onnx_path)
        print(f"[Soya WD14] Downloaded {model_name}.onnx")

    if not os.path.exists(csv_path):
        print(f"[Soya WD14] Downloading {model_name}.csv ...")
        urllib.request.urlretrieve(f"{base_url}/selected_tags.csv", csv_path)
        print(f"[Soya WD14] Downloaded {model_name}.csv")


def _get_model(model_name):
    if model_name not in _model_cache:
        onnx_path = os.path.join(models_dir, model_name + ".onnx")
        if not os.path.exists(onnx_path):
            _download_model(model_name)
        print(f"[Soya WD14] Loading model: {model_name}")
        _model_cache[model_name] = InferenceSession(os.path.join(models_dir, model_name + ".onnx"), providers=ORT_PROVIDERS)
        print(f"[Soya WD14] Model cached: {model_name}")
    return _model_cache[model_name]


class SoyaWD14Tagger_mdsoya:
    @classmethod
    def INPUT_TYPES(s):
        installed = _get_installed_models()
        # Show installed models + known but not yet downloaded models
        all_known = [m for m in HF_MODELS if m not in installed]
        models = installed + all_known
        default = installed[0] if installed else "wd-v1-4-moat-tagger-v2"
        return {"required": {
            "image": ("IMAGE",),
            "model": (models or [default], {"default": default}),
            "threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
            "character_threshold": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
            "replace_underscore": ("BOOLEAN", {"default": True}),
            "trailing_comma": ("BOOLEAN", {"default": False}),
            "exclude_tags": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "tag"
    OUTPUT_NODE = True
    CATEGORY = "Soya"

    def tag(self, image, model, threshold=0.35, character_threshold=0.85,
            replace_underscore=True, trailing_comma=False, exclude_tags=""):
        tensor = (image.cpu().numpy() * 255).astype(np.uint8)
        pbar = comfy.utils.ProgressBar(tensor.shape[0])
        results = []

        for i in range(tensor.shape[0]):
            pil_image = Image.fromarray(tensor[i])

            # Load model (cached after first load, downloads if missing)
            session = _get_model(model)
            input_info = session.get_inputs()[0]
            height = input_info.shape[1]

            # Aspect-ratio-preserving resize + white pad to square
            ratio = float(height) / max(pil_image.size)
            new_size = tuple(int(x * ratio) for x in pil_image.size)
            pil_image = pil_image.resize(new_size, Image.LANCZOS)
            square = Image.new("RGB", (height, height), (255, 255, 255))
            square.paste(pil_image, ((height - new_size[0]) // 2, (height - new_size[1]) // 2))

            # RGB -> BGR, add batch dim
            img = np.array(square).astype(np.float32)[:, :, ::-1]
            img = np.expand_dims(img, 0)

            # Read tags from CSV
            tags = []
            general_index = None
            character_index = None
            csv_path = os.path.join(models_dir, model + ".csv")
            with open(csv_path) as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if general_index is None and row[2] == "0":
                        general_index = reader.line_num - 2
                    elif character_index is None and row[2] == "4":
                        character_index = reader.line_num - 2
                    tags.append(row[1].replace("_", " ") if replace_underscore else row[1])

            # Run inference
            label_name = session.get_outputs()[0].name
            probs = session.run([label_name], {input_info.name: img})[0]
            result = list(zip(tags, probs[0]))

            # Filter by threshold
            general = [item for item in result[general_index:character_index] if item[1] > threshold]
            character = [item for item in result[character_index:] if item[1] > character_threshold]
            all_tags = character + general

            # Exclude tags
            remove = {s.strip().lower() for s in exclude_tags.split(",") if s.strip()}
            all_tags = [tag for tag in all_tags if tag[0].lower() not in remove]

            # Format output
            separator = "" if trailing_comma else ", "
            res = separator.join(
                item[0].replace("(", "\\(").replace(")", "\\)") + (", " if trailing_comma else "")
                for item in all_tags
            )
            results.append(res)
            pbar.update(1)

        return {"ui": {"tags": results}, "result": (results,)}
