import torch
import torch.nn.functional as F

class MaskProportionalExpand:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "scale_factor": ("FLOAT", {"default": 2.0, "min": 0.0001, "max": 10.0, "step": 0.0001}),
                "minimum_size": ("INT", {"default": 1, "min": 1, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "expand"
    CATEGORY = "Soya"

    def expand(self, mask, scale_factor, minimum_size):
        # mask is shape [B, H, W]
        B, H, W = mask.shape
        out_masks = []
        
        for i in range(B):
            m = mask[i]
            # Find bounding box of the active mask region
            active = torch.nonzero(m > 0)
            if active.numel() == 0:
                # If mask is empty, maintain silence
                out_masks.append(m.unsqueeze(0))
                continue
                
            y_min = active[:, 0].min().item()
            y_max = active[:, 0].max().item()
            x_min = active[:, 1].min().item()
            x_max = active[:, 1].max().item()
            
            w_old = x_max - x_min + 1
            h_old = y_max - y_min + 1
            
            # Find center of the bounding box
            cx = x_min + w_old / 2.0
            cy = y_min + h_old / 2.0
            
            w_new = int(round(w_old * scale_factor))
            h_new = int(round(h_old * scale_factor))
            
            w_new = max(minimum_size, w_new)
            h_new = max(minimum_size, h_new)
                
            # Crop the original mask
            crop = m[y_min:y_max+1, x_min:x_max+1]
            
            # Resize the crop mapped to [1, 1, H, W] so F.interpolate accepts it
            crop_req = crop.unsqueeze(0).unsqueeze(0).float()
            
            # Expand the mask content proportionately
            resized = F.interpolate(crop_req, size=(h_new, w_new), mode="bilinear", align_corners=False)
            resized = resized.squeeze(0).squeeze(0)
            
            # Calculate new boundary placement
            left = int(round(cx - w_new / 2.0))
            top = int(round(cy - h_new / 2.0))
            right = left + w_new
            bottom = top + h_new
            
            # Clamp the boundaries to the image dimensions (Requirement 2)
            c_left = max(0, left)
            c_top = max(0, top)
            c_right = min(W, right)
            c_bottom = min(H, bottom)
            
            # Calculate corresponding indices in the resized crop
            s_left = c_left - left
            s_top = c_top - top
            s_right = w_new - (right - c_right)
            s_bottom = h_new - (bottom - c_bottom)
            
            new_m = torch.zeros_like(m)
            
            if c_right > c_left and c_bottom > c_top and s_right > s_left and s_bottom > s_top:
                new_m[c_top:c_bottom, c_left:c_right] = resized[s_top:s_bottom, s_left:s_right]
            
            out_masks.append(new_m.unsqueeze(0))
            
        return (torch.cat(out_masks, dim=0),)
