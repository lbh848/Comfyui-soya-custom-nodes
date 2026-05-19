"""
SoyaWeightedCombineEmbeds – Combine two embeds with individual weights.

Applies weight per embed before combining, so each input has different influence.
Output can be fed directly into IPAdapter Embeds Toggle (Soya).
"""

import torch


COMBINE_METHODS = ["average", "norm average", "concat", "add", "subtract", "max", "min"]


def _combine_embeds(embeds, method):
    """Combine a batch of embeds [N, ...] along dim=0."""
    if method == "concat":
        return embeds
    elif method == "add":
        return torch.sum(embeds, dim=0).unsqueeze(0)
    elif method == "subtract":
        return (embeds[0] - torch.mean(embeds[1:], dim=0)).unsqueeze(0)
    elif method == "average":
        return torch.mean(embeds, dim=0).unsqueeze(0)
    elif method == "norm average":
        return torch.mean(
            embeds / torch.norm(embeds, dim=0, keepdim=True), dim=0,
        ).unsqueeze(0)
    elif method == "max":
        return torch.max(embeds, dim=0).values.unsqueeze(0)
    elif method == "min":
        return torch.min(embeds, dim=0).values.unsqueeze(0)
    return embeds


class SoyaWeightedCombineEmbeds_mdsoya:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embeds_1": ("EMBEDS",),
                "weight_1": ("FLOAT", {"default": 1.0, "min": -1, "max": 5, "step": 0.05}),
                "embeds_2": ("EMBEDS",),
                "weight_2": ("FLOAT", {"default": 1.0, "min": -1, "max": 5, "step": 0.05}),
                "combine_method": (COMBINE_METHODS,),
            },
        }

    RETURN_TYPES = ("EMBEDS",)
    RETURN_NAMES = ("embeds",)
    FUNCTION = "execute"
    CATEGORY = "ipadapter/faceid"

    def execute(self, embeds_1, weight_1, embeds_2, weight_2, combine_method):
        # Apply per-embed weights
        weighted_1 = embeds_1 * weight_1
        weighted_2 = embeds_2 * weight_2

        # Concat along batch dim, then combine
        combined_input = torch.cat([weighted_1, weighted_2], dim=0)
        result = _combine_embeds(combined_input, combine_method)

        print(f"[Soya:WeightedCombine] w1={weight_1}, w2={weight_2}, method={combine_method}, output={result.shape}")
        return (result,)
