"""Whole-image Anima Tagger inference using the unmodified upstream package."""

import json
import os
from collections import deque
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

import comfy.model_management
from comfy_api.latest import io
import folder_paths
import numpy as np
from PIL import Image


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
        )

    @classmethod
    def execute(cls, image, device, exclude_tags="", include_rating=False,
                include_character=False, include_copyright=False):
        models_dir = Path(folder_paths.models_dir).resolve() / "captioners"
        if device not in {"auto", "cpu"}:
            raise ValueError(f"Unsupported Anima Tagger device: {device}")
        torch_device = comfy.model_management.get_torch_device() if device == "auto" else "cpu"
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
        temp_root = Path(folder_paths.get_temp_directory())
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="soya-anima-tagger-", dir=temp_root) as temp_dir:
            directory = Path(temp_dir)
            for index, pixels in enumerate(image):
                array = (pixels.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(array).save(directory / f"{index:06d}.png")
            result_path = directory / "captions.json"
            command = [
                sys.executable, "-u", str(Path(__file__).with_name("soya_anima_tagger_runtime.py")),
                "--models-dir", str(models_dir), "--device", str(torch_device),
                "--input-dir", str(directory), "--output", str(result_path),
                "--exclude-tags", exclude_tags,
            ]
            if include_rating:
                command.append("--include-rating")
            if include_character:
                command.append("--include-character")
            if include_copyright:
                command.append("--include-copyright")
            if device == "auto" and torch_device.type != "cpu":
                # Leave room for the tagger weights and 1008px inference workspace.
                comfy.model_management.free_memory(4 * 1024**3, torch_device)
            # The worker owns all model memory and exits after this image batch.
            # Its HF cache also stays independent of other custom nodes.
            with subprocess.Popen(
                command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            ) as process:
                log_lines = deque(maxlen=100)

                def relay_output():
                    for line in process.stdout:
                        log_lines.append(line)
                        message = f"[Soya Anima Tagger] {line.rstrip()}"
                        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
                        print(message.encode(encoding, errors="replace").decode(encoding), flush=True)

                reader = threading.Thread(target=relay_output, daemon=True)
                reader.start()
                try:
                    while True:
                        try:
                            process.wait(timeout=0.5)
                            break
                        except subprocess.TimeoutExpired:
                            comfy.model_management.throw_exception_if_processing_interrupted()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
                finally:
                    reader.join()
                if process.returncode:
                    raise RuntimeError(f"Anima Tagger 실행 실패:\n{''.join(log_lines).strip()}")
            captions = json.loads(result_path.read_text(encoding="utf-8"))
        return io.NodeOutput(captions)
