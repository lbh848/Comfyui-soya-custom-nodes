class MergeSegs:
    """
    Merges multiple segs (from batch/list input) into a single segs.

    This node takes a batch or list of segs and combines all their SEG lists into one.
    All input segs must have the same image size.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("SEGS", "INT")
    RETURN_NAMES = ("merged_segs", "total_count")
    FUNCTION = "merge"
    CATEGORY = "Soya/Segs"

    def merge(self, segs):
        """
        Merge multiple segs from batch/list into one.

        Args:
            segs: List of SEGS tuples, each containing (size, segments_list)
                  - size: (width, height) of original image
                  - segments_list: list of SEG objects

        Returns:
            merged_segs: Combined segs with all segments
            total_count: Total number of segments after merging
        """
        if not segs:
            return ((0, 0), []), 0

        # 단일 segs가 들어온 경우 (리스트가 아닌 경우 처리)
        if not isinstance(segs, list):
            segs = [segs]

        # 첫 번째 segs의 size를 기준으로 사용
        base_size, base_segments = segs[0]
        all_segments = list(base_segments)

        # 나머지 segs들의 segment들을 추가
        for segs_item in segs[1:]:
            size, segments = segs_item

            # 모든 segs가 동일한 size를 가져야 함
            if size != base_size:
                raise ValueError(
                    f"All segs must have the same size. "
                    f"Expected {base_size}, got {size}"
                )

            all_segments.extend(segments)

        merged_segs = (base_size, all_segments)
        return (merged_segs, len(all_segments))
