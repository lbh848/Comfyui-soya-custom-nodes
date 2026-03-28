class BBoxDebug_mdsoya:
    """
    Debug node that outputs bbox values from segs for debugging purposes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("bbox_info",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "debug_bbox"
    CATEGORY = "Soya/Debug"

    def debug_bbox(self, segs):
        """
        Extract and format bbox information from segs.

        Args:
            segs: SEGS tuple containing (size, segments_list)
                  - size: (width, height) of original image
                  - segments_list: list of SEG named tuples with attributes:
                    - cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper
                    - bbox: [x1, y1, x2, y2]
                    - crop_region: [x, y, width, height]

        Returns:
            List of formatted strings with bbox information
        """
        size, segments = segs
        num_segs = len(segments)

        if num_segs == 0:
            return (["No segments found"],)

        bbox_info_list = []
        for i, seg in enumerate(segments):
            # seg.bbox = [x1, y1, x2, y2]
            bbox = seg.bbox
            x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
            width = x2 - x1
            height = y2 - y1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # seg attributes
            label = getattr(seg, 'label', 'unknown')
            confidence = float(getattr(seg, 'confidence', [0.0])[0]) if hasattr(seg, 'confidence') else 0.0

            # crop_region도 표시
            crop_region = seg.crop_region
            cr_x, cr_y, cr_w, cr_h = crop_region[0], crop_region[1], crop_region[2], crop_region[3]

            info = f"[{i}] bbox: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) | size: {width:.1f}x{height:.1f} | center: ({center_x:.1f}, {center_y:.1f}) | crop_region: ({cr_x}, {cr_y}, {cr_w}, {cr_h}) | label: {label} | conf: {confidence:.3f}"
            bbox_info_list.append(info)

        return (bbox_info_list,)
