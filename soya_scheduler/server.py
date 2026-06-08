"""
Soya Scheduler API server – aiohttp routes served by ComfyUI's PromptServer.
"""

import os
import json
from aiohttp import web

from .config_manager import (
    load_config,
    get_available_models,
    get_available_devices,
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
        return web.json_response({"ok": True})

    # ── API: Models ─────────────────────────────────────────────
    @routes.get("/soya_scheduler/api/models/{model_type}")
    async def api_get_models(request):
        model_type = request.match_info["model_type"]
        models = get_available_models(model_type)
        return web.json_response(models)

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

        # GPU info via nvidia-smi
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
