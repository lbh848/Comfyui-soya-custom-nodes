import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import math
import comfy.clip_vision
import comfy.model_management

class AssignCharactersCLIP_mdsoya:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "character_names": ("STRING", {"forceInput": True}),
                "clip_vision": ("CLIP_VISION",),
                "reference_image": ("IMAGE",),
                "query_image": ("IMAGE",),
                "crop": (["center", "none"],),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "FLOAT",)
    RETURN_NAMES = ("assigned_names", "scores",)
    OUTPUT_IS_LIST = (True, True,)
    FUNCTION = "assign_characters"
    CATEGORY = "Soya"

    def assign_characters(self, character_names, clip_vision, reference_image, query_image, crop):
        # Extract names from a potentially batched string input
        names_list = []
        for cn in character_names:
            if isinstance(cn, str):
                names_list.extend([name.strip() for name in cn.replace('\n', ',').split(",") if name.strip()])
            elif getattr(cn, '__iter__', False) and not isinstance(cn, dict):
                for name in cn:
                    names_list.append(str(name).strip())
            else:
                names_list.append(str(cn).strip())

        # Unwrap list inputs (INPUT_IS_LIST wraps everything in a list)
        cv = clip_vision[0] if isinstance(clip_vision, list) else clip_vision
        ref_img = reference_image[0] if isinstance(reference_image, list) else reference_image
        query_img = query_image[0] if isinstance(query_image, list) else query_image
        crop_mode = crop[0] if isinstance(crop, list) else crop

        crop_image = crop_mode == "center"

        print(f"[ref_size] {ref_img.shape}")
        print(f"[query_size] {query_img.shape}")

        N = ref_img.shape[0]
        M = query_img.shape[0]

        if N == 0 or M == 0:
            print(f"[assign_characters] Skipping: reference={N}, query={M}")
            return ([], [])

        # Encode images separately (may have different sizes)
        ref_output = cv.encode_image(ref_img, crop=crop_image)
        query_output = cv.encode_image(query_img, crop=crop_image)

        ref_embeds = ref_output.image_embeds if hasattr(ref_output, 'image_embeds') else ref_output.get("image_embeds")
        query_embeds = query_output.image_embeds if hasattr(query_output, 'image_embeds') else query_output.get("image_embeds")

        if ref_embeds is None or query_embeds is None:
            raise ValueError("Failed to extract image embeddings from CLIP Vision output.")

        if len(names_list) != N:
            print(f"Warning: Number of character names ({len(names_list)}) does not match number of reference images ({N}).")
            if len(names_list) < N:
                names_list.extend([f"Unknown_{i}" for i in range(len(names_list), N)])
            else:
                names_list = names_list[:N]

        if M == 0 or N == 0:
            return ([], [])

        if ref_embeds.dim() > 2:
            ref_embeds = ref_embeds.view(N, -1)
        if query_embeds.dim() > 2:
            query_embeds = query_embeds.view(M, -1)

        sim_matrix = F.cosine_similarity(query_embeds.unsqueeze(1), ref_embeds.unsqueeze(0), dim=2)

        K = math.ceil(M / N) if M > 0 and N > 0 else 1

        if K > 1:
            sim_matrix_expanded = sim_matrix.repeat(1, K)
        else:
            sim_matrix_expanded = sim_matrix

        cost_matrix = -sim_matrix_expanded.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_ref_indices = col_ind % N

        assigned_names = [names_list[i] for i in matched_ref_indices]

        scores = []
        for r, c in zip(row_ind, matched_ref_indices):
            scores.append(float(sim_matrix[r, c].item()))

        final_names = [None] * M
        final_scores = [0.0] * M

        for idx, (r, c) in enumerate(zip(row_ind, matched_ref_indices)):
            final_names[r] = assigned_names[idx]
            final_scores[r] = scores[idx]

        return (final_names, final_scores)
