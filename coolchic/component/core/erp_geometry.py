# Software Name: Cool-Chic
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 Orange
# SPDX-License-Identifier: BSD 3-Clause "New"
#
# This software is distributed under the BSD-3-Clause license.
#
# Authors: see CONTRIBUTORS.md

"""ERP (Equirectangular Projection) geometry utilities.

This module provides functions to compute geodesic (great-circle) distances
between ERP pixels and to build a precomputed context-index tensor that, for
each pixel in a latent grid, selects the *dim_arm* causally available
neighbours that are geodesically closest to it.

The context-index tensor is computed **once** at model initialisation and
cached as a PyTorch buffer, so it adds negligible overhead at runtime.

Polar-only mode
~~~~~~~~~~~~~~~
When ``polar_threshold_deg`` is set to a value > 0, the geodesic context is
only applied to rows whose absolute latitude exceeds that threshold (i.e. the
polar caps).  Equatorial rows fall back to the standard rectangular causal
pattern used by the baseline ARM, which has no overhead near the equator where
the ERP distortion is negligible.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ------------------------------------------------------------------ #
#  Spherical coordinate helpers                                        #
# ------------------------------------------------------------------ #

def _latitude(row: int, H: int) -> float:
    """Latitude (in radians) of the centre of ERP pixel row *row*.

    The equator corresponds to row = H/2.  North pole is row = 0.
    """
    return np.pi * (0.5 - (row + 0.5) / H)


def _longitude(col: int, W: int) -> float:
    """Longitude (in radians) of the centre of ERP pixel column *col*."""
    return 2 * np.pi * (col + 0.5) / W - np.pi


def _geodesic_distance(phi1: float, theta1: float,
                       phi2: float, theta2: float) -> float:
    """Great-circle distance (radians) between two points on the unit sphere.

    Args:
        phi1, phi2: Latitudes in radians.
        theta1, theta2: Longitudes in radians.

    Returns:
        Angular distance in radians, in [0, pi].
    """
    cos_angle = (
        np.sin(phi1) * np.sin(phi2)
        + np.cos(phi1) * np.cos(phi2) * np.cos(theta1 - theta2)
    )
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _horizontal_radius_from_angle(
    row: int,
    H: int,
    W: int,
    angular_radius_deg: float = 10.0,
    max_horizontal: int = 40,
) -> int:
    """Number of horizontal pixels needed to cover *angular_radius_deg* degrees
    on the sphere at latitude *row*.

    At the equator the horizontal pixel spacing equals 360/W degrees;
    near the poles the spacing shrinks as cos(lat), so more pixels are needed
    to cover the same angular support.

    Args:
        row: Row index (0 = north pole).
        H: Total height of the latent grid.
        W: Total width of the latent grid.
        angular_radius_deg: Desired angular support in degrees.
        max_horizontal: Hard cap on the returned radius (in pixels).

    Returns:
        Integer horizontal radius (>= 1).
    """
    phi = _latitude(row, H)
    theta_per_pixel = 2.0 * np.pi / W
    angular_radius = np.deg2rad(angular_radius_deg)
    cos_phi = max(float(np.cos(phi)), 1e-6)
    radius = int(np.ceil(angular_radius / (theta_per_pixel * cos_phi)))
    return min(max(radius, 1), max_horizontal)


# ------------------------------------------------------------------ #
#  Causal candidate generation                                         #
# ------------------------------------------------------------------ #

def _get_causal_candidates(
    row: int,
    col: int,
    H: int,
    W: int,
    vertical_radius: int = 4,
    angular_radius_deg: float = 10.0,
    max_horizontal: int = 40,
) -> List[Tuple[int, int]]:
    """Return all causal neighbour pixels within an adaptive window.

    "Causal" here means: strictly above the current pixel (any column) or on
    the same row but strictly to the left -- matching the raster-scan decoding
    order used by Cool-Chic.

    Horizontal columns wrap around (ERP is periodic in longitude).
    Vertical rows are clamped to [0, H-1] (no wrap, poles are boundaries).

    Args:
        row: Row of the pixel being decoded.
        col: Column of the pixel being decoded.
        H: Latent grid height.
        W: Latent grid width.
        vertical_radius: Number of rows above *row* to consider.
        angular_radius_deg: Angular support for the adaptive horizontal radius.
        max_horizontal: Hard cap on horizontal radius (pixels).

    Returns:
        List of (rr, cc) pixel coordinates of causal neighbours.
    """
    candidates: List[Tuple[int, int]] = []

    for dy in range(-vertical_radius, 1):
        rr = row + dy
        if rr < 0:
            continue

        # Adaptive horizontal radius for this specific row
        hr = _horizontal_radius_from_angle(rr, H, W, angular_radius_deg, max_horizontal)

        if dy == 0:
            # Same row: only pixels strictly to the left are causal.
            # Do NOT wrap around: col=0 has no same-row context.
            dx_range = range(-min(hr, col), 0)  # dx in [-hr..-1], but col+dx >= 0
            for dx in dx_range:
                cc = col + dx   # no wrap; cc >= 0 guaranteed
                candidates.append((rr, cc))
        else:
            # Rows above: all pixels in the horizontal window are causal
            # (rr < row => flat index rr*W + cc < row*W regardless of cc).
            # Longitude wrap is valid here.
            for dx in range(-hr, hr + 1):
                cc = (col + dx) % W
                candidates.append((rr, cc))

    return candidates



# ------------------------------------------------------------------ #
#  Context-index tensor                                                #
# ------------------------------------------------------------------ #

def _build_rect_context_index_row(row: int, H: int, W: int, dim_arm: int,
                                   vertical_radius: int) -> np.ndarray:
    """Return the rectangular causal context indices for all pixels in *row*.

    This replicates the fixed (non-geodesic) causal mask used by the standard
    ARM: for each pixel we collect up to *dim_arm* causal neighbours in raster
    order (pixels above, then same row to the left), starting from the nearest.

    Args:
        row: Row index in the latent grid.
        H: Grid height.
        W: Grid width.
        dim_arm: Number of context slots.
        vertical_radius: How many rows above to consider.

    Returns:
        int64 array of shape ``[W, dim_arm]`` with flat indices.
    """
    out = np.zeros((W, dim_arm), dtype=np.int64)
    for col in range(W):
        neighbours = []
        # Rows above in reverse distance order (closest first)
        for dy in range(-1, -vertical_radius - 1, -1):
            rr = row + dy
            if rr < 0:
                break
            for dx in range(0, W):
                # expand outward from directly above
                for cc_delta in ([0] if dx == 0 else [-dx, dx]):
                    cc = col + cc_delta
                    if 0 <= cc < W:
                        neighbours.append(rr * W + cc)
                    if len(neighbours) == dim_arm:
                        break
                if len(neighbours) == dim_arm:
                    break
            if len(neighbours) == dim_arm:
                break
        # Same row: pixels strictly to the left
        for dx in range(-1, -col - 1, -1):
            neighbours.append(row * W + (col + dx))
            if len(neighbours) == dim_arm:
                break
        # Pad with 0 if not enough neighbours
        while len(neighbours) < dim_arm:
            neighbours.append(0)
        out[col, :] = neighbours[:dim_arm]
    return out


def build_erp_context_index(
    H: int,
    W: int,
    dim_arm: int,
    vertical_radius: int = 4,
    angular_radius_deg: float = 10.0,
    max_horizontal: int = 40,
    polar_threshold_deg: float = 0.0,
    sigma_scale: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Build the ERP context-index tensor for a latent grid of size H x W.

    For every pixel (row, col), the function:
      1. Collects all causal candidates in the adaptive window.
      2. Sorts them by geodesic distance (ascending).
      3. Keeps the *dim_arm* closest ones.
      4. Records their flat index ``row * W + col``.

    If a pixel has fewer than *dim_arm* causal neighbours (e.g., top-left
    corner), the missing slots are filled with index 0 -- which corresponds to
    the top-left pixel.  Because the context grid is zero-padded before use,
    this safely produces a zero context for those slots.

    **Polar-only mode** (``polar_threshold_deg > 0``): rows whose absolute
    latitude is *below* the threshold fall back to a standard rectangular
    causal context, avoiding the geodesic overhead in equatorial regions where
    the ERP distortion is negligible and the extra context brings no benefit.

    Args:
        H: Latent grid height.
        W: Latent grid width.
        dim_arm: Number of context pixels required by the ARM MLP.
        vertical_radius: Rows above current pixel to consider.
        angular_radius_deg: Angular support for adaptive horizontal window.
        max_horizontal: Hard cap on horizontal radius (pixels).
        polar_threshold_deg: Absolute latitude (degrees) below which the
            standard rectangular context is used instead of the geodesic one.
            Set to 0 (default) to always use the geodesic context (original
            behaviour).
        sigma_scale: Multiplier on the geometry-derived Gaussian sigma.
            The base sigma is ``π / max(H, W)`` (the angular step of one pixel).
            ``sigma_scale = 1.0`` (default) gives good differentiation between
            immediate neighbours and farther ones.  Larger values make weights
            more uniform; smaller values make only the nearest neighbour matter.

    Returns:
        Tuple containing:
        - LongTensor of shape ``[H * W, dim_arm]`` containing flat indices.
        - FloatTensor of shape ``[H * W, dim_arm]`` containing the normalized weights.
    """
    index = np.zeros((H * W, dim_arm), dtype=np.int64)
    weights = np.zeros((H * W, dim_arm), dtype=np.float32)

    phi = np.array([_latitude(r, H) for r in range(H)])       # in radians
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    polar_threshold_rad = np.deg2rad(polar_threshold_deg)

    # Gaussian sigma derived from the angular pixel step of the latent grid.
    # pi / H is the angular step between adjacent rows (latitude step), which
    # equals the geodesic distance to the nearest vertical neighbour at the
    # equator.  For ERP images (W ≈ 2H) the longitude step 2π/W ≈ π/H matches,
    # so this choice is isotropic.  sigma_scale lets the caller tune sharpness:
    #   sigma_scale < 1  → sharper (only nearest neighbour matters)
    #   sigma_scale = 1  → moderate differentiation (recommended default)
    #   sigma_scale > 1  → softer (weights become more uniform)
    sigma = sigma_scale * (np.pi / H)

    for row in range(H):
        # ── Polar-only mode: fall back to rectangular for equatorial rows ──
        if polar_threshold_deg > 0.0 and abs(phi[row]) < polar_threshold_rad:
            rect = _build_rect_context_index_row(row, H, W, dim_arm, vertical_radius)
            index[row * W:(row + 1) * W, :] = rect
            # weights uniformly distributed for rectangular fallback
            weights[row * W:(row + 1) * W, :] = 1.0 / dim_arm
            continue

        # ── Geodesic context for polar rows (or all rows when threshold = 0) ─
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

        sorted_idx = np.argsort(distances, kind='mergesort')
        sorted_rel_cands = [rel_candidates[i] for i in sorted_idx]

        for col in range(W):
            flat_pixel = row * W + col
            valid_flat_indices = []
            valid_distances = []

            for dy, dx, rr in sorted_rel_cands:
                if dy == 0 and col + dx < 0:
                    continue
                cc = (col + dx) % W
                valid_flat_indices.append(rr * W + cc)
                dist_idx = sorted_rel_cands.index((dy, dx, rr))
                valid_distances.append(distances[sorted_idx[dist_idx]])
                if len(valid_flat_indices) == dim_arm:
                    break

            while len(valid_flat_indices) < dim_arm:
                valid_flat_indices.append(0)

            index[flat_pixel, :] = valid_flat_indices
            
            n_valid = len(valid_distances)
            if n_valid > 0:
                d = np.array(valid_distances)
                w = np.exp(-(d**2) / (2 * sigma**2))
                if np.sum(w) > 0:
                    w = w / np.sum(w)
                weights[flat_pixel, :n_valid] = w

    return torch.from_numpy(index), torch.from_numpy(weights)


# ------------------------------------------------------------------ #
#  Positional encoding for ERP context slots                           #
# ------------------------------------------------------------------ #

def build_erp_pos_encoding(
    H: int,
    W: int,
    ctx_index: torch.Tensor,
) -> torch.Tensor:
    """Build a positional encoding tensor for every ERP context slot.

    For each pixel ``p = (row, col)`` and each of its ``dim_arm`` context
    slots selected by *ctx_index*, compute the **relative spherical
    displacement** of the context pixel with respect to ``p``:

    * ``Δlat`` = (lat_ctx − lat_p) / (π/2)  — normalised to roughly [−2, 2]
    * ``Δlon`` = shortest signed longitude difference / π — normalised to [−1, 1]

    The resulting tensor has shape ``[H * W, dim_arm * 2]``, where for each
    pixel the features are arranged as::

        [Δlat₀, Δlon₀,  Δlat₁, Δlon₁,  …,  Δlatₙ, Δlonₙ]

    This tensor is intended to be concatenated to the extracted context values
    (shape ``[H * W, dim_arm]``) before being passed to the ARM MLP, giving it
    explicit spatial information about where each neighbour lies on the sphere.

    Args:
        H: Latent grid height.
        W: Latent grid width.
        ctx_index: LongTensor ``[H * W, dim_arm]`` as returned by
            :func:`build_erp_context_index`.  Must be on CPU.

    Returns:
        Float32 tensor of shape ``[H * W, dim_arm * 2]``.  Each pair of
        columns ``(2k, 2k+1)`` contains ``(Δlat_norm, Δlon_norm)`` for the
        *k*-th context slot.
    """
    ctx_index_np = ctx_index.numpy()           # [H*W, dim_arm]
    dim_arm = ctx_index_np.shape[1]

    # Precompute per-row latitudes and per-column longitudes
    lats = np.array([_latitude(r, H) for r in range(H)], dtype=np.float32)  # [H]
    lons = np.array([2.0 * np.pi * (c + 0.5) / W - np.pi
                     for c in range(W)], dtype=np.float32)                   # [W]

    # Pixel (row, col) for every flat index
    flat = np.arange(H * W, dtype=np.int64)
    px_rows = flat // W                        # [H*W]
    px_cols = flat % W                         # [H*W]
    phi0 = lats[px_rows]                       # [H*W]
    lon0 = lons[px_cols]                       # [H*W]

    # Context (row, col) for every flat context index  [H*W, dim_arm]
    ctx_rows = ctx_index_np // W
    ctx_cols = ctx_index_np % W
    phi2 = lats[ctx_rows]                      # [H*W, dim_arm]
    lon2 = lons[ctx_cols]                      # [H*W, dim_arm]

    # ── Δlat: normalised by π/2 so that ±pole ≈ ±2 ──────────────────
    delta_lat = (phi2 - phi0[:, None]) / (np.pi / 2.0)          # [H*W, dim_arm]
    delta_lat = np.clip(delta_lat, -2.0, 2.0).astype(np.float32)

    # ── Δlon: wrap to [−π, π] then normalise to [−1, 1] ─────────────
    delta_lon_raw = lon2 - lon0[:, None]                         # [H*W, dim_arm]
    delta_lon = (delta_lon_raw + np.pi) % (2.0 * np.pi) - np.pi  # wrap
    delta_lon_norm = (delta_lon / np.pi).astype(np.float32)      # [−1, 1]

    # Interleave: [H*W, dim_arm, 2] → [H*W, dim_arm * 2]
    pos_enc = np.stack([delta_lat, delta_lon_norm], axis=-1)     # [H*W, dim_arm, 2]
    pos_enc = pos_enc.reshape(H * W, dim_arm * 2)

    return torch.from_numpy(pos_enc)
