"""
Soya Model Manager API server – aiohttp routes served by ComfyUI's PromptServer.
Supports: ControlNet, IPAdapter, Upscale, VAE, CLIP, CLIP Vision, Ultralytics, Diffusion Models
"""

import os
import re
import json
import asyncio
import hashlib
import shutil
from pathlib import Path
from aiohttp import web, ClientSession, ClientTimeout

try:
    import folder_paths
except ImportError:
    folder_paths = None

# Model categories we manage (the ones LoRA Manager doesn't cover in UI)
# folder_paths_key: key in ComfyUI's folder_paths (may differ from category name)
# folder_name: actual directory name under models/ (for fallback)
# recursive: scan subdirectories too (e.g. ultralytics/bbox/)
MODEL_CATEGORIES = {
    "controlnet": {
        "label": "ControlNet",
        "extensions": [".safetensors", ".pt", ".pth", ".bin"],
        "folder_name": "controlnet",
    },
    "ipadapter": {
        "label": "IP-Adapter",
        "extensions": [".safetensors", ".pt", ".pth", ".bin"],
        "folder_name": "ipadapter",
    },
    "upscale_models": {
        "label": "Upscale",
        "extensions": [".safetensors", ".pt", ".pth", ".bin"],
        "folder_name": "upscale_models",
    },
    "vae": {
        "label": "VAE",
        "extensions": [".safetensors", ".pt", ".pth", ".bin"],
        "folder_name": "vae",
    },
    "clip": {
        "label": "CLIP / Text Encoders",
        "extensions": [".safetensors", ".pt", ".pth", ".bin"],
        "folder_name": "clip",
        "folder_paths_key": "text_encoders",
    },
    "clip_vision": {
        "label": "CLIP Vision",
        "extensions": [".safetensors", ".pt", ".pth", ".bin"],
        "folder_name": "clip_vision",
    },
    "ultralytics": {
        "label": "Ultralytics (YOLO)",
        "extensions": [".pt", ".onnx", ".safetensors", ".ckpt"],
        "folder_name": "ultralytics",
        "recursive": True,
    },
    "diffusion_models": {
        "label": "Diffusion Models",
        "extensions": [".safetensors", ".pt", ".pth", ".bin"],
        "folder_name": "diffusion_models",
    },
}

# Active downloads tracking
_active_downloads = {}

_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(_DIR, "web")


def _get_model_dirs(category: str) -> list:
    """Get all absolute paths for a model category using folder_paths."""
    if folder_paths is None:
        return []

    cat_info = MODEL_CATEGORIES.get(category, {})
    # Use explicit folder_paths_key if defined (e.g. "text_encoders" for clip)
    fp_key = cat_info.get("folder_paths_key", category)
    try:
        paths = list(folder_paths.get_folder_paths(fp_key))
    except (KeyError, Exception):
        paths = []

    # If folder_paths has no entry, fallback to models_dir/folder_name
    if not paths:
        try:
            models_dir = folder_paths.models_dir
        except AttributeError:
            try:
                models_dir = os.path.dirname(folder_paths.get_folder_paths("checkpoints")[0])
            except (IndexError, Exception):
                models_dir = ""
        if models_dir:
            folder_name = cat_info.get("folder_name", category)
            fallback = os.path.join(models_dir, folder_name)
            if os.path.isdir(fallback):
                paths = [fallback]

    return paths


def _get_model_dir(category: str) -> str:
    """Get primary path for a model category (first match)."""
    dirs = _get_model_dirs(category)
    return dirs[0] if dirs else ""


def _scan_models(category: str) -> list:
    """Scan model directories and return list of model files."""
    cat_info = MODEL_CATEGORIES.get(category, {})
    exts = cat_info.get("extensions", [".safetensors"])
    recursive = cat_info.get("recursive", False)
    model_dirs = _get_model_dirs(category)
    models = []

    for model_dir in model_dirs:
        if not model_dir or not os.path.isdir(model_dir):
            continue

        if recursive:
            for root, dirs, files in os.walk(model_dir):
                rel_root = os.path.relpath(root, model_dir)
                for f in sorted(files):
                    if any(f.lower().endswith(ext) for ext in exts):
                        fpath = os.path.join(root, f)
                        display_name = f if rel_root == "." else os.path.join(rel_root, f).replace("\\", "/")
                        models.append({
                            "name": display_name,
                            "size": os.path.getsize(fpath),
                            "modified": os.path.getmtime(fpath),
                        })
        else:
            for f in sorted(os.listdir(model_dir)):
                fpath = os.path.join(model_dir, f)
                if os.path.isfile(fpath) and any(f.lower().endswith(ext) for ext in exts):
                    models.append({
                        "name": f,
                        "size": os.path.getsize(fpath),
                        "modified": os.path.getmtime(fpath),
                    })
    return models


def _parse_civitai_url(url: str) -> int | None:
    """Extract model ID from CivitAI URL."""
    # https://civitai.com/models/12345 or /models/12345/some-name
    m = re.search(r"civitai\.com/models/(\d+)", url)
    if m:
        return int(m.group(1))
    # https://civitai.com/api/download/models/12345 (version file ID)
    m = re.search(r"civitai\.com/api/download/models/(\d+)", url)
    if m:
        return int(m.group(1))
    # Just a numeric ID
    if url.strip().isdigit():
        return int(url.strip())
    return None


def _parse_civitai_version_url(url: str) -> int | None:
    """Extract version ID from CivitAI URL."""
    m = re.search(r"modelVersionId=(\d+)", url)
    if m:
        return int(m.group(1))
    m = re.search(r"civitai\.com/models/\d+\?modelVersionId=(\d+)", url)
    if m:
        return int(m.group(1))
    return None


def setup_routes():
    try:
        from server import PromptServer
        instance = PromptServer.instance
    except Exception:
        print("[Soya:ModelManager] PromptServer not available, skipping route setup")
        return

    routes = instance.routes

    # ── Serve SPA page ─────────────────────────────────────────
    @routes.get("/soya_model_manager")
    @routes.get("/soya_model_manager/")
    async def serve_page(request):
        html_path = os.path.join(WEB_DIR, "index.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return web.Response(text=f.read(), content_type="text/html")
        return web.Response(text="Soya Model Manager not found", status=404)

    # ── API: Categories ────────────────────────────────────────
    @routes.get("/soya_model_manager/api/categories")
    async def api_categories(request):
        cats = []
        for key, info in MODEL_CATEGORIES.items():
            model_dirs = _get_model_dirs(key)
            count = len(_scan_models(key))
            cats.append({
                "key": key,
                "label": info["label"],
                "count": count,
                "folder": ", ".join(model_dirs) if model_dirs else "",
            })
        return web.json_response(cats)

    # ── API: List models in category ───────────────────────────
    @routes.get("/soya_model_manager/api/models/{category}")
    async def api_list_models(request):
        category = request.match_info["category"]
        if category not in MODEL_CATEGORIES:
            return web.json_response({"error": f"Unknown category: {category}"}, status=400)
        models = _scan_models(category)
        return web.json_response({"category": category, "models": models})

    # ── API: Delete model ──────────────────────────────────────
    @routes.delete("/soya_model_manager/api/models/{category}/{filename}")
    async def api_delete_model(request):
        category = request.match_info["category"]
        filename = request.match_info["filename"]
        if category not in MODEL_CATEGORIES:
            return web.json_response({"error": "Unknown category"}, status=400)

        model_dir = _get_model_dir(category)
        if not model_dir:
            return web.json_response({"error": "Model directory not found"}, status=404)

        # Security: prevent path traversal
        filepath = os.path.normpath(os.path.join(model_dir, filename))
        if not filepath.startswith(os.path.normpath(model_dir)):
            return web.json_response({"error": "Invalid path"}, status=400)
        if not os.path.isfile(filepath):
            return web.json_response({"error": "File not found"}, status=404)

        os.remove(filepath)
        return web.json_response({"ok": True})

    # ── API: CivitAI - Get model info ──────────────────────────
    @routes.get("/soya_model_manager/api/civitai/model/{model_id}")
    async def api_civitai_model_info(request):
        model_id = request.match_info["model_id"]
        async with ClientSession(timeout=ClientTimeout(total=30)) as session:
            async with session.get(f"https://civitai.com/api/v1/models/{model_id}") as resp:
                if resp.status != 200:
                    return web.json_response({"error": f"CivitAI API returned {resp.status}"}, status=resp.status)
                data = await resp.json()
                return web.json_response(data)

    # ── API: CivitAI - Get version info ────────────────────────
    @routes.get("/soya_model_manager/api/civitai/version/{version_id}")
    async def api_civitai_version_info(request):
        version_id = request.match_info["version_id"]
        async with ClientSession(timeout=ClientTimeout(total=30)) as session:
            async with session.get(f"https://civitai.com/api/v1/model-versions/{version_id}") as resp:
                if resp.status != 200:
                    return web.json_response({"error": f"CivitAI API returned {resp.status}"}, status=resp.status)
                data = await resp.json()
                return web.json_response(data)

    # ── API: Download model ────────────────────────────────────
    @routes.post("/soya_model_manager/api/download")
    async def api_download_model(request):
        body = await request.json()
        url = body.get("url", "")
        category = body.get("category", "")
        filename = body.get("filename", "")

        if not url or not category or not filename:
            return web.json_response({"error": "Missing url, category, or filename"}, status=400)
        if category not in MODEL_CATEGORIES:
            return web.json_response({"error": f"Unknown category: {category}"}, status=400)

        model_dir = _get_model_dir(category)
        if not model_dir:
            return web.json_response({"error": "Model directory not found"}, status=404)

        os.makedirs(model_dir, exist_ok=True)
        filepath = os.path.join(model_dir, filename)

        # Security: prevent path traversal
        filepath = os.path.normpath(filepath)
        if not filepath.startswith(os.path.normpath(model_dir)):
            return web.json_response({"error": "Invalid path"}, status=400)

        if os.path.exists(filepath):
            return web.json_response({"error": "File already exists"}, status=409)

        download_id = hashlib.md5(f"{url}:{filepath}".encode()).hexdigest()[:12]
        _active_downloads[download_id] = {
            "url": url,
            "filepath": filepath,
            "status": "downloading",
            "progress": 0,
            "downloaded": 0,
            "total": 0,
        }

        # Start download in background
        asyncio.ensure_future(_do_download(download_id, url, filepath))

        return web.json_response({"download_id": download_id, "status": "started"})

    # ── API: Download progress ─────────────────────────────────
    @routes.get("/soya_model_manager/api/download/{download_id}")
    async def api_download_progress(request):
        download_id = request.match_info["download_id"]
        info = _active_downloads.get(download_id)
        if not info:
            return web.json_response({"error": "Unknown download"}, status=404)
        return web.json_response(info)

    # ── API: Parse CivitAI URL ─────────────────────────────────
    @routes.post("/soya_model_manager/api/parse-url")
    async def api_parse_url(request):
        body = await request.json()
        url = body.get("url", "")
        model_id = _parse_civitai_url(url)
        version_id = _parse_civitai_version_url(url)
        if not model_id:
            return web.json_response({"error": "Could not parse CivitAI URL"}, status=400)
        return web.json_response({"model_id": model_id, "version_id": version_id})

    # ── API: Direct URL download (non-CivitAI) ────────────────
    @routes.post("/soya_model_manager/api/download-direct")
    async def api_download_direct(request):
        body = await request.json()
        url = body.get("url", "")
        category = body.get("category", "")

        if not url or not category:
            return web.json_response({"error": "Missing url or category"}, status=400)
        if category not in MODEL_CATEGORIES:
            return web.json_response({"error": f"Unknown category: {category}"}, status=400)

        model_dir = _get_model_dir(category)
        if not model_dir:
            return web.json_response({"error": "Model directory not found"}, status=404)

        os.makedirs(model_dir, exist_ok=True)

        # Extract filename from URL
        filename = url.split("/")[-1].split("?")[0]
        if not filename:
            filename = "model.safetensors"

        filepath = os.path.join(model_dir, filename)
        filepath = os.path.normpath(filepath)
        if not filepath.startswith(os.path.normpath(model_dir)):
            return web.json_response({"error": "Invalid path"}, status=400)

        if os.path.exists(filepath):
            return web.json_response({"error": "File already exists"}, status=409)

        download_id = hashlib.md5(f"{url}:{filepath}".encode()).hexdigest()[:12]
        _active_downloads[download_id] = {
            "url": url,
            "filepath": filepath,
            "status": "downloading",
            "progress": 0,
            "downloaded": 0,
            "total": 0,
        }

        asyncio.ensure_future(_do_download(download_id, url, filepath))
        return web.json_response({"download_id": download_id, "status": "started"})

    print("[Soya:ModelManager] API routes registered")


async def _do_download(download_id: str, url: str, filepath: str):
    """Background download with progress tracking."""
    info = _active_downloads[download_id]
    tmp_path = filepath + ".downloading"

    try:
        # CivitAI download URLs may need API key for some models
        headers = {}
        async with ClientSession(timeout=ClientTimeout(total=3600), headers=headers) as session:
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status != 200:
                    info["status"] = "error"
                    info["error"] = f"HTTP {resp.status}"
                    return

                total = int(resp.headers.get("Content-Length", 0))
                info["total"] = total
                downloaded = 0

                with open(tmp_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(1024 * 1024):  # 1MB chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        info["downloaded"] = downloaded
                        if total > 0:
                            info["progress"] = round(downloaded / total * 100, 1)

        # Rename temp file to final
        if os.path.exists(tmp_path):
            os.rename(tmp_path, filepath)

        info["status"] = "completed"
        info["progress"] = 100

    except asyncio.CancelledError:
        info["status"] = "cancelled"
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception as e:
        info["status"] = "error"
        info["error"] = str(e)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
