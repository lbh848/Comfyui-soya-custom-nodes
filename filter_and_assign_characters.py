import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import math
import comfy.clip_vision


class FilterAndAssignCharacters_mdsoya:
    """
    요구사항16.md 기반:
    1) prompt에서 character_names 중 언급된 이름과 인덱스만 추출
    2) 해당 인덱스로 reference_image 필터링 (prompt에 없는 캐릭터는 제외)
    3) 이미지 정규화 (reference + query 모두)
    4) CLIP Vision 유사도 + Hungarian 알고리즘으로 쿼리 이미지에 이름 할당
       - query < ref: 확률 높은 이름만 부여
       - query > ref: 남은 query에 확률 가장 높은 이름 부여
       - 우선순위: 이름을 가능한 고르게 분배
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "character_names": ("STRING", {"forceInput": True}),
                "clip_vision": ("CLIP_VISION",),
                "reference_image": ("IMAGE",),
                "query_image": ("IMAGE",),
                "normalize": (["none", "minmax", "mean_std", "histogram_eq"], {"default": "none"}),
            },
            "optional": {
                "remove_suffix": ("STRING", {"forceInput": True}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "FLOAT",)
    RETURN_NAMES = ("assigned_names", "scores",)
    OUTPUT_IS_LIST = (True, True,)
    FUNCTION = "filter_and_assign"
    CATEGORY = "Soya/Character"

    @staticmethod
    def _normalize_images(images, mode):
        if mode == "none":
            return images

        B = images.shape[0]
        result = images.clone()

        if mode == "minmax":
            for i in range(B):
                img = result[i]
                vmin, vmax = img.min(), img.max()
                if vmax - vmin > 1e-6:
                    result[i] = (img - vmin) / (vmax - vmin)

        elif mode == "mean_std":
            for i in range(B):
                img = result[i]
                std = img.std()
                if std > 1e-6:
                    normed = (img - img.mean()) / std
                    nmin, nmax = normed.min(), normed.max()
                    result[i] = (normed - nmin) / (nmax - nmin + 1e-8)

        elif mode == "histogram_eq":
            for i in range(B):
                for c in range(images.shape[3]):
                    ch = result[i, :, :, c]
                    hist = torch.histc(ch, bins=256, min=0, max=1)
                    cdf = hist.cumsum(0)
                    cdf_min = cdf[cdf > 0].min()
                    if cdf[-1] - cdf_min > 0:
                        lut = (cdf - cdf_min) / (cdf[-1] - cdf_min)
                        result[i, :, :, c] = lut[(ch * 255).long().clamp(0, 255)]

        return result

    def filter_and_assign(self, prompt, character_names, clip_vision, reference_image, query_image, normalize, remove_suffix=None):
        # ── unwrap 스칼라 입력 ──
        prompt_text = prompt[0] if isinstance(prompt, list) else prompt
        cv = clip_vision[0] if isinstance(clip_vision, list) else clip_vision
        norm_mode = normalize[0] if isinstance(normalize, list) else normalize

        # ── unwrap 이미지: 단일 텐서 또는 리스트 모두 처리 ──
        if isinstance(reference_image, list):
            ref_img = torch.cat(reference_image, dim=0) if len(reference_image) > 1 else reference_image[0]
        else:
            ref_img = reference_image

        if isinstance(query_image, list):
            query_img = torch.cat(query_image, dim=0) if len(query_image) > 1 else query_image[0]
        else:
            query_img = query_image

        # ── suffix 설정 (optional 입력 또는 기본값) ──
        if remove_suffix is not None:
            suffix = remove_suffix[0] if isinstance(remove_suffix, list) else remove_suffix
        else:
            suffix = "_default.webp"

        # ── character_names 파싱 ──
        names_list = []
        for cn in character_names:
            if isinstance(cn, str):
                # 콤마/개행으로 분리 (각 요소가 여러 이름을 포함할 수 있음)
                names_list.extend([n.strip() for n in cn.replace('\n', ',').split(",") if n.strip()])
            elif isinstance(cn, (list, tuple)):
                # 리스트인 경우 각 요소를 개별 이름으로
                names_list.extend([str(n).strip() for n in cn if str(n).strip()])
            else:
                names_list.append(str(cn).strip())

        # ── suffix 제거 ──
        if suffix:
            names_list = [n.removesuffix(suffix) if n.endswith(suffix) else n for n in names_list]

        N_total = ref_img.shape[0]
        M = query_img.shape[0]

        # ── Step 1: prompt 필터링 ──
        # prompt에 언급된 character_names만 사용. prompt가 비어있으면 전체 사용.
        def preprocess(text):
            return text.lower().replace("_", " ").replace("-", " ")

        if prompt_text and prompt_text.strip():
            prompt_lower = preprocess(prompt_text)
            matched_indices = [
                i for i, name in enumerate(names_list)
                if i < N_total and preprocess(name) in prompt_lower
            ]

            # 중복 이름 제거: 동일한 이름이 여러 인덱스에 있으면 첫 번째만 유지
            seen_names = set()
            deduped_indices = []
            for i in matched_indices:
                name_key = preprocess(names_list[i])
                if name_key not in seen_names:
                    seen_names.add(name_key)
                    deduped_indices.append(i)
            if matched_indices:
                names_list = [names_list[i] for i in matched_indices]
                ref_img = ref_img[matched_indices]
            else:
                return ([], [])
        else:
            pass

        N = ref_img.shape[0]

        if N == 0 or M == 0:
            return ([], [])

        # ── Step 2: 이미지 정규화 ──
        ref_img = self._normalize_images(ref_img, norm_mode)
        query_img = self._normalize_images(query_img, norm_mode)

        # ── Step 3: CLIP Vision 인코딩 (center crop 고정) ──
        ref_output = cv.encode_image(ref_img, crop=True)
        query_output = cv.encode_image(query_img, crop=True)

        ref_embeds = ref_output.image_embeds if hasattr(ref_output, 'image_embeds') else ref_output.get("image_embeds")
        query_embeds = query_output.image_embeds if hasattr(query_output, 'image_embeds') else query_output.get("image_embeds")

        if ref_embeds is None or query_embeds is None:
            raise ValueError("Failed to extract image embeddings from CLIP Vision output.")

        if ref_embeds.dim() > 2:
            ref_embeds = ref_embeds.view(N, -1)
        if query_embeds.dim() > 2:
            query_embeds = query_embeds.view(M, -1)

        # ── Step 4: cosine similarity + Hungarian assignment ──
        sim_matrix = F.cosine_similarity(query_embeds.unsqueeze(1), ref_embeds.unsqueeze(0), dim=2)

        # Hungarian 알고리즘: M > N이면 패딩 컬럼(cost=0)이 추가되어
        # 가장 유사도가 낮은 query는 패딩 컬럼에 배정됨 → "unknown" 처리
        row_ind, col_ind = linear_sum_assignment(-sim_matrix.cpu().numpy())

        # query 순서 보존, 미배정 query는 "unknown"
        final_names = ["unknown"] * M
        final_scores = [0.0] * M

        for r, c in zip(row_ind, col_ind):
            if c < N:
                final_names[r] = names_list[c]
                final_scores[r] = float(sim_matrix[r, c].item())

        return (final_names, final_scores)
