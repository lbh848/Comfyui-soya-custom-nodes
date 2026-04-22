import os
import sys

_ray_initialized = False

# Ensure soya_scheduler is importable as a top-level package
# (ComfyUI loads custom nodes with the full directory path as module name,
# which breaks Ray's pickle-based serialization)
_pkg_dir = os.path.dirname(os.path.abspath(__file__))       # soya_scheduler/
_parent_dir = os.path.dirname(_pkg_dir)                      # comfyui-soya-custom/

if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Register clean module names in sys.modules so Ray workers can resolve them
if "soya_scheduler" not in sys.modules:
    sys.modules["soya_scheduler"] = sys.modules[__name__]


def ensure_ray_initialized():
    global _ray_initialized
    if _ray_initialized:
        return
    import importlib
    import os as _os

    ray = importlib.import_module("ray")

    # Set PYTHONPATH so Ray worker processes can import soya_scheduler
    pp = _os.environ.get("PYTHONPATH", "")
    if _parent_dir not in pp.split(_os.pathsep):
        _os.environ["PYTHONPATH"] = _parent_dir + (_os.pathsep + pp if pp else "")

    if not ray.is_initialized():
        from .config_manager import load_config
        config = load_config()
        num_cpus = config.get("settings", {}).get("num_cpus", 1)
        # NEVER call torch.cuda.* here – it creates a CUDA context on
        # GPU 0 in the main ComfyUI process and eats VRAM.
        # The Actor worker uses CUDA directly (bypassing Ray resources).
        ray.init(ignore_reinit_error=True, num_gpus=0, num_cpus=num_cpus)
    _ray_initialized = True


_analyzer = None
_analyzer_device = None


def get_analyzer():
    """Get or create the singleton FaceAnalyzer actor.
    Detects device changes and recreates the actor so GPU memory is released."""
    global _analyzer, _analyzer_device
    ensure_ray_initialized()

    from .config_manager import load_config
    config = load_config()
    device = config.get("settings", {}).get("device", "cuda:1")

    # Device changed → kill old actor (worker process dies, GPU memory freed)
    if _analyzer is not None and _analyzer_device != device:
        try:
            import ray
            ray.kill(_analyzer)
        except Exception:
            pass
        _analyzer = None

    if _analyzer is None:
        import ray
        from .ray_worker import FaceAnalyzer

        _analyzer = FaceAnalyzer.options(name="face_analyzer").remote()
        _analyzer_device = device

    return _analyzer
