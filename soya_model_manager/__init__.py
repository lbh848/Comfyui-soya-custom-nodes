"""
Soya Model Manager – downloads ControlNet, IPAdapter, Upscale, VAE, CLIP models via CivitAI.
Separate from soya_scheduler – purely for model downloading.
"""

from .server import setup_routes
from .web_dir import WEB_DIRECTORY  # noqa: F401

setup_routes()
print("[Soya:ModelManager] Routes registered")
