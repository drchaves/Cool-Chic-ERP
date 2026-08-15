import torch
import numpy as np
from coolchic.component.core.erp_geometry import build_erp_context_index, _latitude, _horizontal_radius_from_angle

def build_erp_context_index_opt(H, W, dim_arm, vertical_radius=4, angular_radius_deg=10.0, max_horizontal=40):
    index = np.zeros((H * W, dim_arm), dtype=np.int64)
    phi = np.array([_latitude(r, H) for r in range(H)])
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    for row in range(H):
        rel_candidates = []
        for dy in range(-vertical_radius, 1):
            rr = row + dy
            if rr < 0:
                continue
            hr = _horizontal_radius_from_angle(rr, H, W, angular_radius_deg, max_horizontal)
            if dy == 0:
                for dx in range(-hr, 0):
                    rel_candidates.append((dy, dx, rr))
            else:
                for dx in range(-hr, hr + 1):
                    rel_candidates.append((dy, dx, rr))
        
        if not rel_candidates:
            continue

        dtheta = np.array([c[1] * 2 * np.pi / W for c in rel_candidates])
        rr_arr = np.array([c[2] for c in rel_candidates])
        
        sin_phi0 = sin_phi[row]
        cos_phi0 = cos_phi[row]
        sin_phi1 = sin_phi[rr_arr]
        cos_phi1 = cos_phi[rr_arr]

        cos_angle = sin_phi0 * sin_phi1 + cos_phi0 * cos_phi1 * np.cos(dtheta)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        distances = np.arccos(cos_angle)

        # To handle ties identically to Python's built-in sort (stable sort), we use mergesort
        sorted_idx = np.argsort(distances, kind='mergesort')
        sorted_rel_cands = [rel_candidates[i] for i in sorted_idx]

        for col in range(W):
            flat_pixel = row * W + col
            valid_flat_indices = []
            
            for dy, dx, rr in sorted_rel_cands:
                if dy == 0 and col + dx < 0:
                    continue
                cc = (col + dx) % W
                valid_flat_indices.append(rr * W + cc)
                if len(valid_flat_indices) == dim_arm:
                    break
            
            while len(valid_flat_indices) < dim_arm:
                valid_flat_indices.append(0)
                
            index[flat_pixel, :] = valid_flat_indices

    return torch.from_numpy(index)

print("Computing original...")
idx_orig = build_erp_context_index(32, 64, 14)
print("Computing optimized...")
idx_opt = build_erp_context_index_opt(32, 64, 14)

diffs = (idx_orig != idx_opt).sum().item()
print(f"Differences: {diffs}")
if diffs > 0:
    print(idx_orig[10])
    print(idx_opt[10])
