"""Expose WEB_DIRECTORY so ComfyUI serves our JS extension."""
import os

WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "web")
