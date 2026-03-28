import torch
import numpy as np
from collections import deque


class SplitPersonSegment_mdsoya:
    """
    Person Segment를 사람별로 분리하는 노드.

    Face 감지 결과를 통해 사람 수를 파악하고,
    Person 감지 마스크를 활용해 Segment를 개별 사람별로 분리합니다.
    Canny Edge를 제공하면 인물 형상을 반영한 더 자연스러운 분할이 가능합니다.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_masks": ("MASK",),
                "face_masks": ("MASK",),
                "person_segment": ("MASK",),
            },
            "optional": {
                "canny_edges": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("person_segment_batch", "person_count")
    FUNCTION = "split_person_segment"
    CATEGORY = "Soya/Segs"
    OUTPUT_IS_LIST = (True, False)

    def split_person_segment(self, person_masks, face_masks, person_segment, canny_edges=None):
        # Ensure 3D tensors
        if person_masks.dim() == 2:
            person_masks = person_masks.unsqueeze(0)
        if face_masks.dim() == 2:
            face_masks = face_masks.unsqueeze(0)
        if person_segment.dim() == 2:
            person_segment = person_segment.unsqueeze(0)
        if canny_edges is not None and canny_edges.dim() == 2:
            canny_edges = canny_edges.unsqueeze(0)

        num_people = face_masks.shape[0]
        N_pdet = person_masks.shape[0]
        H, W = person_segment.shape[-2], person_segment.shape[-1]
        seg = person_segment[0]
        device = person_segment.device

        if num_people == 0:
            print("[Soya:SplitPersonSegment] No face masks provided.")
            return ([], 0)

        # Step 1: Compute face centers and valid faces
        face_centers = []
        valid_faces = []
        for i in range(num_people):
            face = face_masks[i]
            ys, xs = torch.where(face > 0.5)
            if len(ys) > 0:
                face_centers.append((ys.float().mean().item(), xs.float().mean().item()))
                valid_faces.append(i)

        num_valid = len(valid_faces)
        if num_valid == 0:
            print("[Soya:SplitPersonSegment] No valid face masks found.")
            return ([], 0)
        if num_valid == 1:
            print("[Soya:SplitPersonSegment] Single person detected.")
            return ([seg], 1)

        print(f"[Soya:SplitPersonSegment] {num_valid} people detected.")

        # Step 2: Compute Voronoi zones (used as tiebreaker)
        ys_grid = torch.arange(H, device=device, dtype=torch.float32).unsqueeze(1).expand(H, W)
        xs_grid = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0).expand(H, W)
        min_dist = torch.full((H, W), float('inf'), device=device)
        nearest = torch.zeros(H, W, dtype=torch.long, device=device)
        for idx in range(num_valid):
            cy, cx = face_centers[idx]
            d = (ys_grid - cy) ** 2 + (xs_grid - cx) ** 2
            closer = d < min_dist
            min_dist[closer] = d[closer]
            nearest[closer] = idx

        # Step 3: Compute body masks from person detection (shared by both algorithms)
        body_masks = torch.zeros(num_valid, H, W, device=device)
        for idx, fi in enumerate(valid_faces):
            face = face_masks[fi]
            face_area = face.sum().item()
            if face_area == 0:
                continue
            union_mask = torch.zeros(H, W, device=device)
            for j in range(N_pdet):
                pdet = person_masks[j]
                overlap = (face * pdet).sum().item()
                if overlap / face_area > 0.3:
                    union_mask = torch.max(union_mask, pdet)
            body_masks[idx] = union_mask * seg

        # Step 4: Choose algorithm based on canny_edges availability
        if canny_edges is not None:
            result = self._split_watershed(
                seg, face_masks, valid_faces, nearest, canny_edges[0],
                body_masks, face_centers, H, W, device
            )
        else:
            result = self._split_voronoi(
                seg, body_masks, valid_faces, nearest, H, W, device
            )

        result_list = [result[i] for i in range(num_valid)]
        return (result_list, num_valid)

    def _split_watershed(self, seg, face_masks, valid_faces, nearest, edges,
                         body_masks, face_centers, H, W, device):
        """Hybrid: Voronoi base + edge-aware boundary refinement.

        Uses Voronoi (face center distance) for overall distribution,
        then refines boundaries near canny edges using 0-1 BFS.
        This gives fair pixel distribution with natural edge-following boundaries.
        """
        num_valid = len(valid_faces)

        edge_np = (edges > 0.5).cpu().numpy()
        seg_np = (seg > 0.5).cpu().numpy()
        nearest_np = nearest.cpu().numpy()

        # --- Phase 1: Voronoi assignment (fair distribution) ---
        voronoi = nearest_np.copy()

        # --- Phase 2: 0-1 BFS assignment (edge-aware) ---
        bfs_edge = edge_np.copy()
        seeds = []
        for idx, fi in enumerate(valid_faces):
            face_np = (face_masks[fi] > 0.5).cpu().numpy()
            seed = face_np.copy()
            for _ in range(15):
                padded = np.pad(seed, 1, mode='constant', constant_values=False)
                seed = (
                    padded[1:-1, 1:-1] |
                    padded[:-2, 1:-1] | padded[2:, 1:-1] |
                    padded[1:-1, :-2] | padded[1:-1, 2:]
                )
            seed = seed & seg_np
            seeds.append(seed)
            bfs_edge = bfs_edge & ~seed

        bfs_assignment = np.full((H, W), -1, dtype=np.int32)
        queue = deque()
        for idx in range(num_valid):
            ys, xs = np.where(seeds[idx])
            for y, x in zip(ys, xs):
                if bfs_assignment[y, x] == -1:
                    bfs_assignment[y, x] = idx
                    queue.appendleft((y, x))

        while queue:
            y, x = queue.popleft()
            owner = bfs_assignment[y, x]
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and bfs_assignment[ny, nx] == -1 and seg_np[ny, nx]:
                    bfs_assignment[ny, nx] = owner
                    if bfs_edge[ny, nx]:
                        queue.append((ny, nx))
                    else:
                        queue.appendleft((ny, nx))

        unreachable = (bfs_assignment == -1) & seg_np
        if unreachable.any():
            for idx in range(num_valid):
                bfs_assignment[unreachable & (nearest_np == idx)] = idx

        # --- Phase 3: Combine ---
        # Dilate edges to define "near edge" zone (within 5px of an edge)
        near_edge = edge_np.copy()
        for _ in range(5):
            padded = np.pad(near_edge, 1, mode='constant', constant_values=False)
            near_edge = (
                padded[1:-1, 1:-1] |
                padded[:-2, 1:-1] | padded[2:, 1:-1] |
                padded[1:-1, :-2] | padded[1:-1, 2:]
            )
        near_edge = near_edge & seg_np

        # Near edges: prefer BFS; elsewhere: keep Voronoi
        assignment = voronoi.copy()
        near_edge_disagree = near_edge & (voronoi != bfs_assignment)
        assignment[near_edge_disagree] = bfs_assignment[near_edge_disagree]

        # Build result tensors
        result = torch.zeros(num_valid, H, W, device=device)
        assignment_t = torch.from_numpy(assignment).to(device)
        for idx in range(num_valid):
            result[idx] = torch.where(assignment_t == idx, seg.float(), result[idx])

        return result

    def _split_voronoi(self, seg, body_masks, valid_faces, nearest, H, W, device):
        """Original Voronoi-based splitting using person detection masks."""
        num_valid = len(valid_faces)

        # Combine body masks with Voronoi assignment
        claim_count = (body_masks > 0.5).sum(dim=0)
        non_contested = claim_count == 1

        result = torch.zeros(num_valid, H, W, device=device)
        for idx in range(num_valid):
            mask = non_contested & (body_masks[idx] > 0.5)
            result[idx] = torch.where(mask, seg.float(), result[idx])

        needs_assignment = (seg > 0.5) & ~non_contested
        for idx in range(num_valid):
            mask = needs_assignment & (nearest == idx)
            result[idx] = torch.where(mask, seg.float(), result[idx])

        return result
