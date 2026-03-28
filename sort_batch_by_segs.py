import torch

class SortBatchBySegsBBox:
    """
    Sorts batch data (images or masks) based on segs bbox x-coordinate (left to right).

    This node takes segs and batch data as input, then sorts both based on the
    horizontal position (x-coordinate) of each segment's bounding box.
    The leftmost element becomes the first batch element,
    the rightmost element becomes the last batch element.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
            },
            "optional": {
                "batch_image": ("IMAGE",),
                "batch_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("SEGS", "IMAGE", "MASK", "INT")
    RETURN_NAMES = ("sorted_segs", "sorted_image", "sorted_mask", "segs_count")
    FUNCTION = "sort_by_bbox"
    CATEGORY = "Soya/Batch"

    def sort_by_bbox(self, segs, batch_image=None, batch_mask=None):
        """
        Sort segs and batch data based on bbox x-coordinate (left to right).

        Args:
            segs: SEGS tuple containing (size, segments_list)
                  - size: (width, height) of original image
                  - segments_list: list of SEG tuples
                    - SEG: (cropped_image, crop_region, bbox, confidence, label, control_net_wrapper)
                    - bbox: (x1, y1, x2, y2)
            batch_image: Tensor of shape [B, H, W, C] (batch of images) - optional
            batch_mask: Tensor of shape [B, H, W] (batch of masks) - optional

        Returns:
            sorted_segs: SEGS sorted by bbox x1
            sorted_image: batch_image sorted by same order (or None)
            sorted_mask: batch_mask sorted by same order (or None)
        """
        # segs 구조: (size, segments_list)
        # size: (width, height)
        # segments_list: list of SEG tuples
        size, segments = segs
        num_segs = len(segments)

        # segs가 비어있으면 입력을 그대로 반환
        if num_segs == 0:
            return (segs, batch_image, batch_mask, 0)

        # 적어도 하나의 배치 데이터가 필요
        if batch_image is None and batch_mask is None:
            raise ValueError("At least one of batch_image or batch_mask must be provided")

        # 각 세그먼트의 bbox 중심점 x 좌표를 추출하여 정렬 인덱스 생성
        # bbox: [x1, y1, x2, y2] - 원본 이미지 내 bbox 좌표
        # 중심점 = (x1 + x2) / 2
        bbox_center_coords = []
        for i, seg in enumerate(segments):
            # seg.bbox = [x1, y1, x2, y2]
            bbox = seg.bbox
            x1, x2 = float(bbox[0]), float(bbox[2])
            center_x = (x1 + x2) / 2
            bbox_center_coords.append((i, center_x))

        # 중심점 x 기준으로 오름차순 정렬 (왼쪽 -> 오른쪽)
        sorted_indices = [idx for idx, _ in sorted(bbox_center_coords, key=lambda x: x[1])]

        # segs 정렬
        sorted_segments = [segments[i] for i in sorted_indices]
        sorted_segs = (size, sorted_segments)

        # batch_image 정렬
        sorted_image = None
        if batch_image is not None:
            if isinstance(batch_image, list):
                batch_image = torch.stack(batch_image)
            if batch_image.shape[0] != num_segs:
                raise ValueError(
                    f"batch_image size ({batch_image.shape[0]}) must match number of segs ({num_segs})"
                )
            sorted_image = batch_image[sorted_indices]

        # batch_mask 정렬
        sorted_mask = None
        if batch_mask is not None:
            if isinstance(batch_mask, list):
                batch_mask = torch.stack(batch_mask)
            if batch_mask.shape[0] != num_segs:
                raise ValueError(
                    f"batch_mask size ({batch_mask.shape[0]}) must match number of segs ({num_segs})"
                )
            sorted_mask = batch_mask[sorted_indices]

        return (sorted_segs, sorted_image, sorted_mask, num_segs)
