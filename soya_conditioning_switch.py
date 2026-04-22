class SoyaConditioningSwitch_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cn_positive": ("CONDITIONING",),
                "cn_negative": ("CONDITIONING",),
                "use_controlnet": ("STRING", {"default": "false"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "negative", "info")
    FUNCTION = "switch"
    CATEGORY = "Soya"

    def switch(self, positive, negative, cn_positive, cn_negative, use_controlnet):
        use_cn = use_controlnet.strip().lower() in ("true", "1", "yes")

        if use_cn:
            out_pos = cn_positive
            out_neg = cn_negative
            mode = "ControlNet"
        else:
            out_pos = positive
            out_neg = negative
            mode = "Normal"

        info = f"[Conditioning Switch] Mode: {mode} | use_controlnet='{use_controlnet}'"
        print(info)

        return (out_pos, out_neg, info)
