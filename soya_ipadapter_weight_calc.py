"""
SoyaIPAdapterWeightCalc – Auto-calculate IPAdapter weights for style + faceid.

Takes style and faceid enable/weight pairs and outputs normalized weights
for Weighted Combine Embeds + Embeds Toggle.

Logic:
  Both active  → scale so sum=2 (preserve input ratio), total = max
  One active   → active side = input weight, other = 0, total = active weight
  Neither      → all zeros, enabled = "false"
"""


def _parse_bool(s):
    return s.strip().lower() in ("true", "1", "yes")


class SoyaIPAdapterWeightCalc_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style_enabled": ("STRING", {"default": "true", "multiline": False}),
                "style_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "faceid_enabled": ("STRING", {"default": "true", "multiline": False}),
                "faceid_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "FLOAT", "STRING")
    RETURN_NAMES = ("style_weight_out", "faceid_weight_out", "total_weight", "enabled")
    FUNCTION = "execute"
    CATEGORY = "ipadapter/faceid"

    def execute(self, style_enabled, style_weight, faceid_enabled, faceid_weight):
        s_on = _parse_bool(style_enabled)
        f_on = _parse_bool(faceid_enabled)

        if s_on and f_on:
            # Both active: preserve ratio, scale so sum = 2.0
            total_ratio = style_weight + faceid_weight
            if total_ratio > 0:
                style_out = (style_weight / total_ratio) * 2.0
                faceid_out = (faceid_weight / total_ratio) * 2.0
            else:
                style_out = 1.0
                faceid_out = 1.0
            total = max(style_out, faceid_out)
            enabled = "true"

        elif s_on:
            style_out = style_weight
            faceid_out = 0.0
            total = style_weight
            enabled = "true"

        elif f_on:
            style_out = 0.0
            faceid_out = faceid_weight
            total = faceid_weight
            enabled = "true"

        else:
            style_out = 0.0
            faceid_out = 0.0
            total = 0.0
            enabled = "false"

        print(f"[Soya:WeightCalc] style={style_out:.2f}, faceid={faceid_out:.2f}, total={total:.2f}, enabled={enabled}")
        return (style_out, faceid_out, total, enabled)
