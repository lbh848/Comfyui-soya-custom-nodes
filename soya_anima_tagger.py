"""Whole-image Anima Tagger inference using the unmodified upstream package."""

import asyncio
import json
import os
from collections import deque
from contextlib import suppress
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import tempfile
import threading

import comfy.model_management
from comfy_api.latest import io
import folder_paths
import numpy as np
from PIL import Image

from .soya_anima_tagger_runtime import RESULT_PREFIX


_worker_lock = threading.Lock()
_worker = None
_other_prompt_submissions = 0
_tagging_nodes = {
    "LoadImage", "SoyaRefImageLoader_mdsoya", "SoyaAnimaTagger_mdsoya",
    "PreviewAny", "SoyaTextSender_mdsoya",
}


def _is_tagging_prompt(prompt):
    if not isinstance(prompt, dict) or not prompt:
        return False
    node_types = {node.get("class_type") for node in prompt.values() if isinstance(node, dict)}
    return "SoyaAnimaTagger_mdsoya" in node_types and node_types <= _tagging_nodes


class _AnimaTaggerWorker:
    def __init__(self, models_dir, device):
        self.device = device
        self.results = Queue()
        self.log_lines = deque(maxlen=100)
        env = os.environ.copy()
        env.update({
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONPATH": str(Path(__file__).parent / "third_party" / "anime_tools"),
            "HF_HOME": str(models_dir / "huggingface"),
            "HF_HUB_CACHE": str(models_dir / "huggingface" / "hub"),
            "HF_MODULES_CACHE": str(models_dir / "huggingface" / "modules"),
            "HF_HUB_DISABLE_TELEMETRY": "1",
        })
        command = [
            sys.executable, "-u", str(Path(__file__).with_name("soya_anima_tagger_runtime.py")),
            "--models-dir", str(models_dir), "--device", str(device),
        ]
        self.process = subprocess.Popen(
            command, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.reader = threading.Thread(target=self._read_output, daemon=True)
        self.reader.start()

    def _read_output(self):
        try:
            for line in self.process.stdout:
                if line.startswith(RESULT_PREFIX):
                    self.results.put(line[len(RESULT_PREFIX):])
                    continue
                self.log_lines.append(line)
                message = f"[Soya Anima Tagger] {line.rstrip()}"
                encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
                print(message.encode(encoding, errors="replace").decode(encoding), flush=True)
        finally:
            self.results.put(None)

    def tag(self, request):
        try:
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
        except OSError as exc:
            raise RuntimeError("Anima Tagger 워커에 요청을 보낼 수 없습니다") from exc
        while True:
            try:
                result = self.results.get(timeout=0.5)
                break
            except Empty:
                comfy.model_management.throw_exception_if_processing_interrupted()
        if result is None:
            raise RuntimeError(f"Anima Tagger 실행 실패:\n{''.join(self.log_lines).strip()}")
        return json.loads(result)

    def close(self, *, interrupt=False):
        with suppress(OSError):
            self.process.stdin.close()
        if not interrupt:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                interrupt = True
        if interrupt and self.process.poll() is None:
            if os.name == "nt":
                # The venv launcher can have a separate Python child holding CUDA memory.
                subprocess.run(
                    ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW, check=False,
                )
            if self.process.poll() is None:
                self.process.kill()
        self.process.wait()
        self.reader.join()
        self.process.stdout.close()


def _close_worker(*, interrupt=False):
    global _worker
    if _worker is not None:
        worker, _worker = _worker, None
        worker.close(interrupt=interrupt)


def _release_worker():
    with _worker_lock:
        _close_worker()


def register_anima_tagger_memory_cleanup():
    from aiohttp import web
    from server import PromptServer

    @web.middleware
    async def memory_cleanup(request, handler):
        global _other_prompt_submissions
        if request.method == "POST" and request.path in {"/prompt", "/api/prompt"}:
            body = await request.json()
            if not _is_tagging_prompt(body.get("prompt")):
                _other_prompt_submissions += 1
                try:
                    await asyncio.to_thread(_release_worker)
                    return await handler(request)
                finally:
                    _other_prompt_submissions -= 1
        response = await handler(request)
        if (request.method == "POST" and request.path in {"/free", "/api/free"}
                and 200 <= response.status < 300):
            body = await request.json()
            if body.get("unload_models") or body.get("free_memory"):
                await asyncio.to_thread(_release_worker)
        return response

    PromptServer.instance.app.middlewares.append(memory_cleanup)


class SoyaAnimaTagger_mdsoya(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SoyaAnimaTagger_mdsoya",
            display_name="Anima Tagger (Soya)",
            category="Soya/Util",
            description="Analyze each whole image with Anima Tagger. Missing model files download automatically on first use.",
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("device", options=["auto", "cpu"], default="auto"),
                io.String.Input(
                    "exclude_tags", display_name="제외할 태그", default="",
                    multiline=True, optional=True,
                    tooltip="쉼표로 구분한 태그를 추가로 제외합니다. 비워도 꺼진 분류의 태그는 제외됩니다.",
                ),
                io.Boolean.Input(
                    "include_rating", display_name="등급 태그 포함", default=False, optional=True,
                    tooltip="safe, sensitive, nsfw, explicit 같은 등급 태그를 포함합니다.",
                ),
                io.Boolean.Input(
                    "include_character", display_name="캐릭터 태그 포함", default=False, optional=True,
                    tooltip="hatsune miku 같은 캐릭터명 태그를 포함합니다. 인원수와 외형 태그에는 영향을 주지 않습니다.",
                ),
                io.Boolean.Input(
                    "include_copyright", display_name="작품 태그 포함", default=False, optional=True,
                    tooltip="original, vocaloid 같은 작품·프랜차이즈 태그를 포함합니다.",
                ),
            ],
            outputs=[io.String.Output("tags", is_output_list=True)],
            hidden=[io.Hidden.prompt],
        )

    @classmethod
    def execute(cls, image, device, exclude_tags="", include_rating=False,
                include_character=False, include_copyright=False):
        global _worker
        from server import PromptServer

        if device not in {"auto", "cpu"}:
            raise ValueError(f"Unsupported Anima Tagger device: {device}")
        torch_device = comfy.model_management.get_torch_device() if device == "auto" else "cpu"
        models_dir = Path(folder_paths.models_dir).resolve() / "captioners"
        tagging_only = _is_tagging_prompt(cls.hidden.prompt)
        temp_root = Path(folder_paths.get_temp_directory())
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="soya-anima-tagger-", dir=temp_root) as temp_dir:
            directory = Path(temp_dir)
            for index, pixels in enumerate(image):
                array = (pixels.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(array).save(directory / f"{index:06d}.png")
            with _worker_lock:
                if not tagging_only or (_worker is not None and (
                    _worker.device != torch_device or _worker.process.poll() is not None
                )):
                    _close_worker()
                if _worker is None:
                    if device == "auto" and torch_device.type != "cpu":
                        # Leave room for the tagger weights and 1008px inference workspace.
                        comfy.model_management.free_memory(4 * 1024**3, torch_device)
                    _worker = _AnimaTaggerWorker(models_dir, torch_device)
                try:
                    captions = _worker.tag({
                        "input_dir": str(directory), "exclude_tags": exclude_tags,
                        "include_rating": include_rating, "include_character": include_character,
                        "include_copyright": include_copyright,
                    })
                except BaseException:
                    _close_worker(interrupt=True)
                    raise
                # Keep CUDA memory for consecutive tag prompts, including when a
                # different workflow is being submitted while this node runs.
                _, pending = PromptServer.instance.prompt_queue.get_current_queue_volatile()
                if (not tagging_only or _other_prompt_submissions
                        or any(not _is_tagging_prompt(item[2]) for item in pending)):
                    _close_worker()
        return io.NodeOutput(captions)
