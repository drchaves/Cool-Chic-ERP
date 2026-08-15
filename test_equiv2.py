import torch
import numpy as np
from coolchic.component.core.erp_geometry import build_erp_context_index, _latitude, _horizontal_radius_from_angle, _geodesic_distance, _longitude

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

        phi0 = _latitude(row, H)
        # Use exact original function to avoid floating point differences
        distances = []
        for c in rel_candidates:
            dy, dx, rr = c
            phi1 = _latitude(rr, H)
            # theta0 = longitude(col), theta1 = longitude(col + dx)
            # theta1 - theta0 = longitude(dx) - longitude(0) ... wait, longitude(col) is linear.
            # longitude(col + dx) - longitude(col) = 2*pi*(dx)/W
            dtheta = 2 * np.pi * dx / W
            
            cos_angle = np.sin(phi0) * np.sin(phi1) + np.cos(phi0) * np.cos(phi1) * np.cos(dtheta)
            dist = float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            distances.append(dist)

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

idx_orig = build_erp_context_index(32, 64, 14)
idx_opt = build_erp_context_index_opt(32, 64, 14)
diffs = (idx_orig != idx_opt).sum().item()
print(f"Differences: {diffs}")
if diffs > 0:
    mask = (idx_orig != idx_opt).sum(dim=1) > 0
    bad_idx = mask.nonzero(as_tuple=True)[0][0]
    print(f"Mismatch at pixel {bad_idx}:")
    print(idx_orig[bad_idx])
    print(idx_opt[bad_idx])

