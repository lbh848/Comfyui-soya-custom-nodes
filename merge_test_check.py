class MergeSegs:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"segs": ("SEGS",),}}

    INPUT_IS_LIST = True
    RETURN_TYPES = ("SEGS", "INT")
    RETURN_NAMES = ("merged_segs", "total_count")
    FUNCTION = "merge"
    CATEGORY = "Soya/Segs"

    def merge(self, segs):
        if not segs:
            return ((0, 0), []), 0
        if not isinstance(segs, list):
            segs = [segs]
        base_size, base_segments = segs[0]
        all_segments = list(base_segments)
        for segs_item in segs[1:]:
            size, segments = segs_item
            all_segments.extend(segments)
        merged_segs = (base_size, all_segments)
        return (merged_segs, len(all_segments))
