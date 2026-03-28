import torch

class MaskBitwiseAnd:
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


class MaskBitwiseOr:
    """
    Performs bitwise OR operation between a batch of masks and a single mask.

    This node takes a batch of masks and a single mask, then applies bitwise OR
    operation to each mask in the batch against the single mask.
    Any active area in either mask will remain active in the result.
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
    FUNCTION = "bitwise_or"
    CATEGORY = "Soya/Mask"
    OUTPUT_IS_LIST = (True,)

    def bitwise_or(self, batch_mask, single_mask):
        """
        Perform bitwise OR between batch masks and a single mask.

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

        # Perform bitwise OR: max of the two masks
        result = torch.max(batch_mask, single_mask.unsqueeze(0))

        # Convert batch to list of individual masks
        result_list = [result[i] for i in range(result.shape[0])]

        return (result_list,)


class MaskBitwiseAndBatch:
    """
    Performs bitwise AND operation between two batches of masks by index.

    Each mask in batch_a is ANDed with the mask at the same index in batch_b.
    Only the overlapping (active in both) areas will remain active.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_a": ("MASK",),
                "batch_b": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("result_mask",)
    FUNCTION = "bitwise_and_batch"
    CATEGORY = "Soya/Mask"
    OUTPUT_IS_LIST = (True,)

    def bitwise_and_batch(self, batch_a, batch_b):
        if batch_a.dim() == 2:
            batch_a = batch_a.unsqueeze(0)
        if batch_b.dim() == 2:
            batch_b = batch_b.unsqueeze(0)

        if batch_a.shape != batch_b.shape:
            raise ValueError(
                f"Batch shapes don't match: batch_a {batch_a.shape} vs batch_b {batch_b.shape}"
            )

        result = batch_a * batch_b
        return ([result[i] for i in range(result.shape[0])],)


class MaskBitwiseOrBatch:
    """
    Performs bitwise OR operation between two batches of masks by index.

    Each mask in batch_a is ORed with the mask at the same index in batch_b.
    Any active area in either mask will remain active in the result.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_a": ("MASK",),
                "batch_b": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("result_mask",)
    FUNCTION = "bitwise_or_batch"
    CATEGORY = "Soya/Mask"
    OUTPUT_IS_LIST = (True,)

    def bitwise_or_batch(self, batch_a, batch_b):
        if batch_a.dim() == 2:
            batch_a = batch_a.unsqueeze(0)
        if batch_b.dim() == 2:
            batch_b = batch_b.unsqueeze(0)

        if batch_a.shape != batch_b.shape:
            raise ValueError(
                f"Batch shapes don't match: batch_a {batch_a.shape} vs batch_b {batch_b.shape}"
            )

        result = torch.max(batch_a, batch_b)
        return ([result[i] for i in range(result.shape[0])],)
