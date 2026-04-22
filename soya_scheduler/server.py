"""
Soya Scheduler API server – aiohttp routes served by ComfyUI's PromptServer.
"""

import os
import json
from aiohttp import web

from .config_manager import (
    load_config,
    save_config,
    get_available_models,
    get_available_devices,
    load_characters,
    save_character_info,
    delete_character,
    rename_character,
    reload_characters,
    find_image_file,
    find_faceid_embed,
)

_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(_DIR, "web")


def setup_routes():
    try:
        from server import PromptServer
        instance = PromptServer.instance
    except Exception:
        print("[Soya:Scheduler] PromptServer not available, skipping route setup")
        return

    routes = instance.routes

    # ── Serve management page ──────────────────────────────────
    @routes.get("/soya_scheduler")
    @routes.get("/soya_scheduler/")
    async def serve_page(request):
        html_path = os.path.join(WEB_DIR, "index.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return web.Response(text=f.read(), content_type="text/html")
        return web.Response(text="Soya Scheduler page not found", status=404)

    # ── API: Config ─────────────────────────────────────────────
    @routes.get("/soya_scheduler/api/config")
    async def api_get_config(request):
        config = load_config()
        resp = web.json_response(config)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @routes.post("/soya_scheduler/api/config")
    async def api_update_config(request):
        body = await request.json()
        config = load_config()
        if "settings" in body:
            config["settings"].update(body["settings"])
        save_config(config)
        return web.json_response({"ok": True})

    # ── API: Models ─────────────────────────────────────────────
    @routes.get("/soya_scheduler/api/models/{model_type}")
    async def api_get_models(request):
        model_type = request.match_info["model_type"]
        models = get_available_models(model_type)
        return web.json_response(models)

    # ── API: Characters ─────────────────────────────────────────
    @routes.get("/soya_scheduler/api/characters")
    async def api_get_characters(request):
        config = load_config()
        settings = config.get("settings", {})
        base_path = settings.get("base_path", "")
        characters = load_characters(base_path)

        characters.append({
            "name": "unknown",
            "gender": settings.get("unknown_gender", "boy"),
            "eye_prompt": settings.get("unknown_eye_prompt", "unknown, black eyes"),
            "image_file": None,
            "is_unknown": True,
        })

        return web.json_response(characters)

    @routes.post("/soya_scheduler/api/characters/{char_name}")
    async def api_add_character(request):
        char_name = request.match_info["char_name"]
        if char_name == "unknown":
            return web.json_response({"error": "Cannot create character named 'unknown'"}, status=400)

        config = load_config()
        base_path = config.get("settings", {}).get("base_path", "")
        if not base_path:
            return web.json_response({"error": "base_path not configured"}, status=400)

        if find_image_file(base_path, char_name):
            return web.json_response({"error": "Character already exists"}, status=409)

        reader = await request.multipart()

        file_data = None
        original_filename = None
        gender = "girl"
        eye_prompt = ""

        while True:
            field = await reader.next()
            if field is None:
                break
            if field.name == "file":
                file_data = await field.read()
                original_filename = field.filename
            elif field.name == "gender":
                gender = (await field.text()).strip() or "girl"
            elif field.name == "eye_prompt":
                eye_prompt = await field.text()

        if not file_data:
            return web.json_response({"error": "No file uploaded"}, status=400)

        ext = os.path.splitext(original_filename or "image.webp")[1] or ".webp"
        os.makedirs(base_path, exist_ok=True)
        img_path = os.path.join(base_path, f"{char_name}{ext}")
        with open(img_path, "wb") as f:
            f.write(file_data)

        info = save_character_info(base_path, char_name, {
            "name": char_name,
            "gender": gender,
            "eye_prompt": eye_prompt,
        })
        info["image_file"] = f"{char_name}{ext}"

        return web.json_response(info, status=201)

    @routes.put("/soya_scheduler/api/characters/{char_name}")
    async def api_update_character(request):
        char_name = request.match_info["char_name"]
        body = await request.json()
        config = load_config()
        settings = config.get("settings", {})
        base_path = settings.get("base_path", "")

        if char_name == "unknown":
            if "gender" in body:
                settings["unknown_gender"] = body["gender"]
            if "eye_prompt" in body:
                settings["unknown_eye_prompt"] = body["eye_prompt"]
            config["settings"] = settings
            save_config(config)
            return web.json_response({
                "name": "unknown",
                "gender": settings.get("unknown_gender", "boy"),
                "eye_prompt": settings.get("unknown_eye_prompt", "unknown, black eyes"),
                "image_file": None,
                "is_unknown": True,
            })

        if not base_path:
            return web.json_response({"error": "base_path not configured"}, status=400)

        new_name = body.get("name", char_name)

        if new_name != char_name:
            rename_character(base_path, char_name, new_name)

        info = save_character_info(base_path, new_name, body)
        info["image_file"] = find_image_file(base_path, new_name)

        return web.json_response(info)

    @routes.delete("/soya_scheduler/api/characters/{char_name}")
    async def api_delete_character(request):
        char_name = request.match_info["char_name"]
        if char_name == "unknown":
            return web.json_response({"error": "Cannot delete unknown"}, status=400)

        config = load_config()
        base_path = config.get("settings", {}).get("base_path", "")

        if not delete_character(base_path, char_name):
            return web.json_response({"error": "Character not found"}, status=404)

        return web.json_response({"ok": True})

    # ── API: Reload ─────────────────────────────────────────────
    @routes.post("/soya_scheduler/api/reload")
    async def api_reload(request):
        config = load_config()
        base_path = config.get("settings", {}).get("base_path", "")
        characters = reload_characters(base_path)
        return web.json_response({"ok": True, "characters": characters})

    # ── API: Character image serve ──────────────────────────────
    @routes.get("/soya_scheduler/api/characters/{char_name}/image")
    async def api_get_image(request):
        char_name = request.match_info["char_name"]
        config = load_config()
        base_path = config.get("settings", {}).get("base_path", "")

        if not base_path:
            return web.Response(status=404)

        filename = find_image_file(base_path, char_name)
        if filename:
            filepath = os.path.join(base_path, filename)
            if os.path.exists(filepath):
                return web.FileResponse(filepath)

        return web.Response(status=404)

    # ── API: Character image upload (for existing characters) ──
    @routes.post("/soya_scheduler/api/characters/{char_name}/image")
    async def api_upload_image(request):
        char_name = request.match_info["char_name"]
        if char_name == "unknown":
            return web.json_response({"error": "Cannot upload image for unknown"}, status=400)

        config = load_config()
        base_path = config.get("settings", {}).get("base_path", "")
        if not base_path:
            return web.json_response({"error": "base_path not configured"}, status=400)

        reader = await request.multipart()
        field = await reader.next()
        if field is None:
            return web.json_response({"error": "No file uploaded"}, status=400)

        file_data = await field.read()
        original_filename = field.filename or "image.webp"
        ext = os.path.splitext(original_filename)[1] or ".webp"

        from .config_manager import IMAGE_EXTENSIONS
        for old_ext in IMAGE_EXTENSIONS:
            old_path = os.path.join(base_path, f"{char_name}{old_ext}")
            if os.path.exists(old_path):
                os.remove(old_path)

        img_path = os.path.join(base_path, f"{char_name}{ext}")
        with open(img_path, "wb") as f:
            f.write(file_data)

        return web.json_response({"ok": True, "image_file": f"{char_name}{ext}"})

    # ── API: FaceID embed generation ─────────────────────────────
    @routes.post("/soya_scheduler/api/characters/{char_name}/faceid_embed")
    async def api_generate_faceid_embed(request):
        import asyncio
        import torch
        import numpy as np
        from PIL import Image

        char_name = request.match_info["char_name"]
        if char_name == "unknown":
            return web.json_response({"error": "Cannot generate FaceID embed for unknown"}, status=400)

        config = load_config()
        base_path = config.get("settings", {}).get("base_path", "")
        if not base_path:
            return web.json_response({"error": "base_path not configured"}, status=400)

        image_file = find_image_file(base_path, char_name)
        if not image_file:
            return web.json_response({"error": "Character image not found"}, status=404)

        def _generate():
            from .model_manager import get_insightface_model
            from insightface.utils import face_align

            insightface = get_insightface_model("CUDAExecutionProvider")
            img_path = os.path.join(base_path, image_file)
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)

            insightface.det_model.input_size = (640, 640)
            for size in range(640, 256, -64):
                insightface.det_model.input_size = (size, size)
                face = insightface.get(img_np)
                if face:
                    break
            else:
                raise ValueError("No face detected in character image")

            face_embed = torch.from_numpy(face[0].normed_embedding).unsqueeze(0)  # (1, 512)
            face_crop = face_align.norm_crop(img_np, landmark=face[0].kps, image_size=256)  # (256, 256, 3)
            face_crop_t = torch.from_numpy(face_crop.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()  # (3, 256, 256)

            # Pre-compute CLIP Vision embeddings (two separate models)
            clip_embed = None       # for face matching (clip_vision_model)
            clip_embed_ipa = None   # for IPAdapter FaceID (faceid_clip_vision_model)

            # Apply crop zoom for IPAdapter CLIP encoding to exclude hair area
            crop_zoom = float(config.get("settings", {}).get("faceid_crop_zoom", 1.0))
            if crop_zoom > 1.0:
                h, w = face_crop.shape[:2]
                new_h, new_w = int(h / crop_zoom), int(w / crop_zoom)
                top = (h - new_h) // 2
                left = (w - new_w) // 2
                face_crop_zoomed = face_crop[top:top+new_h, left:left+new_w]
                face_crop_zoomed = np.array(Image.fromarray(face_crop_zoomed).resize((256, 256), Image.BILINEAR))
            else:
                face_crop_zoomed = face_crop

            face_crop_bhwc = face_crop_t.permute(1, 2, 0).unsqueeze(0)  # (1, 256, 256, 3) – full crop for matching
            face_crop_zoomed_bhwc = torch.from_numpy(face_crop_zoomed.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)  # (1, 3, 256, 256)
            face_crop_zoomed_bhwc = face_crop_zoomed_bhwc.permute(0, 2, 3, 1)  # (1, 256, 256, 3)

            def _encode_clip(model_name, input_tensor):
                if not model_name:
                    return None
                try:
                    import folder_paths
                    import comfy.clip_vision
                    import comfy.model_management
                    clip_path = folder_paths.get_full_path("clip_vision", model_name)
                    if not clip_path:
                        print(f"[Soya:FaceID] CLIP Vision model file not found: {model_name}")
                        return None
                    clip_v = comfy.clip_vision.load(clip_path)
                    if clip_v is None:
                        print(f"[Soya:FaceID] Failed to load CLIP Vision model: {model_name} (not a valid CLIP Vision model file)")
                        return None
                    comfy.model_management.load_model_gpu(clip_v.patcher)
                    with torch.no_grad():
                        clip_out = clip_v.encode_image(input_tensor, crop="center")
                    result = clip_out.penultimate_hidden_states.detach().cpu()
                    del clip_v
                    return result
                except Exception as e:
                    print(f"[Soya:FaceID] CLIP encoding with {model_name} failed: {e}")
                    return None

            clip_embed = _encode_clip(config.get("settings", {}).get("clip_vision_model", ""), face_crop_bhwc)
            clip_embed_ipa = _encode_clip(config.get("settings", {}).get("faceid_clip_vision_model", ""), face_crop_zoomed_bhwc)

            from safetensors.torch import save_file
            embed_path = os.path.join(base_path, f"{char_name}_faceid.safetensors")
            data = {
                "face_embed": face_embed,
                "face_crop": face_crop_t,
            }
            if clip_embed is not None:
                data["clip_embed"] = clip_embed
            if clip_embed_ipa is not None:
                data["clip_embed_ipa"] = clip_embed_ipa
            save_file(data, embed_path)
            return embed_path

        loop = asyncio.get_event_loop()
        try:
            embed_path = await loop.run_in_executor(None, _generate)
            return web.json_response({"ok": True, "path": embed_path})
        except ValueError as e:
            return web.json_response({"error": str(e)}, status=422)
        except Exception as e:
            return web.json_response({"error": f"Failed: {str(e)}"}, status=500)

    @routes.delete("/soya_scheduler/api/characters/{char_name}/faceid_embed")
    async def api_delete_faceid_embed(request):
        char_name = request.match_info["char_name"]
        config = load_config()
        base_path = config.get("settings", {}).get("base_path", "")
        if not base_path:
            return web.json_response({"error": "base_path not configured"}, status=400)
        from .config_manager import delete_faceid_embed
        if delete_faceid_embed(base_path, char_name):
            return web.json_response({"ok": True})
        return web.json_response({"error": "FaceID embed not found"}, status=404)

    # ── API: Devices ────────────────────────────────────────────
    @routes.get("/soya_scheduler/api/devices")
    async def api_get_devices(request):
        devices = get_available_devices()
        return web.json_response(devices)

    # ── API: VRAM usage ────────────────────────────────────────
    @routes.get("/soya_scheduler/api/vram")
    async def api_get_vram(request):
        import subprocess, shutil

        nvidia_smi = shutil.which("nvidia-smi")
        gpus = []
        procs = []
        ram = {}

        # RAM info
        try:
            import psutil
            mem = psutil.virtual_memory()
            ram = {"total_mb": mem.total / (1024*1024), "used_mb": mem.used / (1024*1024), "percent": mem.percent}
        except ImportError:
            try:
                out = subprocess.check_output(["wmic", "OS", "get", "TotalVisibleMemorySize,FreePhysicalMemory", "/format:value"],
                    encoding="utf-8", timeout=5)
                total_kb = free_kb = 0
                for line in out.strip().split("\n"):
                    if line.startswith("TotalVisibleMemorySize="):
                        total_kb = int(line.split("=")[1].strip())
                    elif line.startswith("FreePhysicalMemory="):
                        free_kb = int(line.split("=")[1].strip())
                if total_kb:
                    ram = {"total_mb": total_kb/1024, "used_mb": (total_kb-free_kb)/1024, "percent": (total_kb-free_kb)/total_kb*100}
            except Exception:
                pass

        # GPU info via nvidia-smi (no torch.cuda – avoids initializing CUDA on GPU 0)
        if nvidia_smi:
            try:
                out = subprocess.check_output(
                    [nvidia_smi, "--query-gpu=index,gpu_bus_id,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                    encoding="utf-8", timeout=5
                )
                bus_to_index = {}
                for line in out.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 5:
                        idx = int(parts[0])
                        gpus.append({
                            "index": idx,
                            "bus_id": parts[1],
                            "name": parts[2],
                            "used_mb": float(parts[3]),
                            "total_mb": float(parts[4]),
                            "gpu_util": float(parts[5]) if len(parts) > 5 else 0,
                        })
                        bus_to_index[parts[1].lower().replace(" ","")] = idx
            except Exception:
                pass

            try:
                out = subprocess.check_output(
                    [nvidia_smi, "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory", "--format=csv,noheader"],
                    encoding="utf-8", timeout=5
                )
                for line in out.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        bus = parts[0].lower().replace(" ","")
                        procs.append({
                            "gpu_index": bus_to_index.get(bus, -1),
                            "pid": int(parts[1]),
                            "name": parts[2],
                            "used_mb": float(parts[3].replace(" MiB", "").replace("Mib", "")),
                        })
            except Exception:
                pass

        return web.json_response({"gpus": gpus, "processes": procs, "ram": ram})

    # ── API: Process result ─────────────────────────────────────
    @routes.get("/soya_scheduler/api/process_result")
    async def api_get_process_result(request):
        config = load_config()
        return web.json_response(config.get("last_process_result"))

    # ── API: Final prompts ──────────────────────────────────────
    @routes.get("/soya_scheduler/api/final_prompts")
    async def api_get_final_prompts(request):
        config = load_config()
        return web.json_response({"prompts": config.get("last_final_prompts")})

    print("[Soya:Scheduler] API routes registered")
