import torch

class MaskBitwiseAnd_mdsoya:
    """
    Performs bitwise AND operation between a batch of masks and a single mask.

    This node takes a batch of masks and a single mask, then applies bitwise AND
    operation to each mask in the batch against the single mask.
    Only the overlapping (active in both) areas will remain active.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_mask": ("MASK",),
                "single_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("result_mask",)
    FUNCTION = "bitwise_and"
    CATEGORY = "Soya/Mask"
    OUTPUT_IS_LIST = (True,)

    def bitwise_and(self, batch_mask, single_mask):
        """
        Perform bitwise AND between batch masks and a single mask.

        Args:
            batch_mask: Tensor of shape [B, H, W] (batch of masks)
            single_mask: Tensor of shape [H, W] (single mask)

        Returns:
            result_mask: List of tensors, each of shape [H, W]
        """
        # Ensure batch_mask is 3D
        if batch_mask.dim() == 2:
            batch_mask = batch_mask.unsqueeze(0)

        # Ensure single_mask is 2D
        if single_mask.dim() == 3:
            single_mask = single_mask.squeeze(0)

        # Check dimensions match
        if batch_mask.shape[1:] != single_mask.shape:
            raise ValueError(
                f"Mask dimensions don't match: batch_mask {batch_mask.shape[1:]} vs single_mask {single_mask.shape}"
            )

        # Perform bitwise AND (& operation)
        # Both masks are typically in range [0, 1], so we use boolean AND
        result = batch_mask * single_mask.unsqueeze(0)

        # Convert batch to list of individual masks
        result_list = [result[i] for i in range(result.shape[0])]

        return (result_list,)

