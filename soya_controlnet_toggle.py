"""
SoyaControlNetToggle – ControlNet apply + Conditioning Switch in one node.

enable=true  → apply ControlNet to positive/negative conditioning internally
enable=false → unload ControlNet from VRAM, pass through original conditioning

Replaces both "Apply ControlNet" and "Conditioning Switch (Soya)" nodes.
"""


class SoyaControlNetToggle_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable": ("STRING", {"default": "true"}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "info")
    FUNCTION = "doit"
    CATEGORY = "Soya"

    def doit(self, *, enable, positive, negative, control_net, image,
             strength, start_percent, end_percent, vae=None):

        use = enable.strip().lower() in ("true", "1", "yes")

        if not use:
            self._unload(control_net)
            info = "[ControlNet Toggle] DISABLED — ControlNet unloaded from VRAM"
            print(info)
            return (positive, negative, info)

        if strength == 0:
            info = "[ControlNet Toggle] strength=0 — skipping"
            print(info)
            return (positive, negative, info)

        print("[ControlNet Toggle] ENABLED — applying ControlNet")

        control_hint = image.movedim(-1, 1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(
                        control_hint, strength, (start_percent, end_percent), vae=vae
                    )
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                c.append([t[0], d])
            out.append(c)

        info = (f"[ControlNet Toggle] ControlNet applied | "
                f"strength={strength}, range=[{start_percent:.3f}, {end_percent:.3f}]")
        print(info)

        return (out[0], out[1], info)

    @staticmethod
    def _unload(control_net):
        """Move ControlNet model weights to CPU to free VRAM."""
        if hasattr(control_net, 'control_model_wrapped'):
            patcher = control_net.control_model_wrapped
            patcher.detach()
