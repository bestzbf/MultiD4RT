"""Benchmark: original vs vectorized compute_depth_normalizers.

Usage:
    conda run -n d4rt python losses/benchmark_median.py
"""

import torch
import time


# ---------------------------------------------------------------------------
# Original implementation (from losses.py)
# ---------------------------------------------------------------------------
def compute_depth_normalizers_original(
    points: torch.Tensor,
    mask=None,
    groups=None,
    eps: float = 1e-6,
):
    z_coords = points[..., 2]
    if mask is not None:
        valid_mask = mask.bool()
    else:
        valid_mask = torch.ones_like(z_coords, dtype=torch.bool)

    if groups is not None:
        mean_depth_safe = torch.ones_like(z_coords)
        for batch_idx in range(z_coords.shape[0]):
            valid_queries = valid_mask[batch_idx]
            if not valid_queries.any():
                continue
            for group_id in torch.unique(groups[batch_idx, valid_queries]):
                group_mask = valid_queries & (groups[batch_idx] == group_id)
                group_z = z_coords[batch_idx, group_mask]
                group_median = group_z.median()
                group_median = torch.nan_to_num(group_median, nan=1.0, posinf=1.0, neginf=1.0)
                mean_depth_safe[batch_idx, group_mask] = torch.clamp(
                    torch.abs(group_median), min=eps
                )
        return mean_depth_safe, valid_mask

    if mask is not None:
        median_depth_safe = torch.ones(z_coords.shape[0], device=z_coords.device, dtype=z_coords.dtype)
        for b in range(z_coords.shape[0]):
            valid_z = z_coords[b][valid_mask[b]]
            if valid_z.numel() > 0:
                med = valid_z.median()
                med = torch.nan_to_num(med, nan=1.0, posinf=1.0, neginf=1.0)
                median_depth_safe[b] = torch.clamp(torch.abs(med), min=eps)
        median_depth_safe = median_depth_safe.unsqueeze(-1).expand_as(z_coords)
    else:
        median_depth_safe = torch.ones(z_coords.shape[0], device=z_coords.device, dtype=z_coords.dtype)
        for b in range(z_coords.shape[0]):
            med = z_coords[b].median()
            med = torch.nan_to_num(med, nan=1.0, posinf=1.0, neginf=1.0)
            median_depth_safe[b] = torch.clamp(torch.abs(med), min=eps)
        median_depth_safe = median_depth_safe.unsqueeze(-1).expand_as(z_coords)

    return median_depth_safe, valid_mask


# ---------------------------------------------------------------------------
# Vectorized implementation
# ---------------------------------------------------------------------------
def compute_depth_normalizers_vectorized(
    points: torch.Tensor,
    mask=None,
    groups=None,
    eps: float = 1e-6,
):
    """Vectorized version — zero Python loops in the grouped path."""
    z_coords = points[..., 2]  # (B, N)

    if mask is not None:
        valid_mask = mask.bool()
    else:
        valid_mask = torch.ones_like(z_coords, dtype=torch.bool)

    B, N = z_coords.shape
    out = torch.ones_like(z_coords)

    if groups is not None:
        if groups.shape != z_coords.shape:
            raise ValueError(
                f"Expected groups to match points batch/query shape, got "
                f"{tuple(groups.shape)} for points {tuple(points.shape)}"
            )
        # Per-batch: sort by group, then segment-sort within each group.
        # Only B iterations (typically 2) — each iteration is fully GPU-vectorized.
        for batch_idx in range(B):
            z_b = z_coords[batch_idx]      # (N,)
            g_b = groups[batch_idx]        # (N,)
            v_b = valid_mask[batch_idx]    # (N,)

            valid_z = z_b[v_b]
            valid_g = g_b[v_b]

            if valid_z.numel() == 0:
                continue

            num_groups = int(g_b.max().item()) + 1

            # Step 1: sort by group → same-group elements contiguous
            group_order = valid_g.argsort()
            sorted_z = valid_z[group_order]
            sorted_g = valid_g[group_order]

            # Step 2: compute intra-group offset for each element
            group_sizes = torch.bincount(valid_g, minlength=num_groups)
            group_starts = torch.cumsum(group_sizes, 0) - group_sizes
            intra_offset = torch.arange(len(sorted_z), device=z_coords.device) - group_starts[sorted_g]

            # Step 3: segment-sort by value within each group
            # Use a composite key that keeps groups intact: group_id * (max_intra+1) + normalized_value
            # Normalize values to [0,1) range within the entire array to avoid key collisions
            z_min, z_max = sorted_z.min(), sorted_z.max()
            z_range = z_max - z_min
            if z_range > 0:
                z_norm = (sorted_z - z_min) / z_range  # [0, 1]
            else:
                z_norm = torch.zeros_like(sorted_z)
            # Key = group_id * 2 + z_norm (z_norm in [0,1) so won't overflow into next group)
            key = sorted_g.to(torch.double) * 2.0 + z_norm.to(torch.double)
            segment_order = key.argsort()
            segment_z = sorted_z[segment_order]

            # Step 4: group boundaries are still valid (groups are contiguous after segment sort)
            # torch.Tensor.median() returns lower median for even-length: index = (n-1)//2
            median_pos = group_starts + (group_sizes - 1) // 2
            median_pos = median_pos.clamp(max=len(segment_z) - 1)
            median_vals = segment_z[median_pos]

            # Mask empty groups
            empty = group_sizes == 0
            median_vals = median_vals.masked_fill(empty, 1.0)
            median_vals = torch.nan_to_num(median_vals, nan=1.0, posinf=1.0, neginf=1.0)
            median_vals = torch.clamp(torch.abs(median_vals), min=eps)

            out[batch_idx, v_b] = median_vals[valid_g]

        return out, valid_mask

    # Non-grouped: one median per batch element — just 2 iterations, negligible
    for b in range(B):
        if mask is not None:
            valid_z = z_coords[b][valid_mask[b]]
        else:
            valid_z = z_coords[b]
        if valid_z.numel() > 0:
            med = valid_z.median()
            med = torch.nan_to_num(med, nan=1.0, posinf=1.0, neginf=1.0)
            out[b] = torch.clamp(torch.abs(med), min=eps)

    return out, valid_mask


# ---------------------------------------------------------------------------
# Correctness test
# ---------------------------------------------------------------------------
def test_correctness(device="cuda"):
    print("=" * 60)
    print("CORRECTNESS TEST")
    print("=" * 60)

    B, N = 2, 2048
    num_groups = 12  # per batch

    torch.manual_seed(42)
    points = torch.randn(B, N, 3, device=device) * 5
    points[..., 2] = points[..., 2].abs() + 0.1  # positive z

    mask = torch.rand(B, N, device=device) > 0.1  # 10% invalid

    groups = torch.randint(0, num_groups, (B, N), device=device)
    # Some groups may have no valid elements — test edge case

    # Original
    out_orig, mask_orig = compute_depth_normalizers_original(points, mask=mask, groups=groups)
    # Vectorized
    out_vec, mask_vec = compute_depth_normalizers_vectorized(points, mask=mask, groups=groups)

    # Compare
    diff = (out_orig - out_vec).abs().max().item()
    mask_match = (mask_orig == mask_vec).all().item()

    print(f"  Max absolute difference: {diff:.2e}")
    print(f"  Mask match: {mask_match}")

    # Check per-group consistency
    for b in range(B):
        for g in range(num_groups):
            gmask = mask[b] & (groups[b] == g)
            if gmask.any():
                orig_val = out_orig[b, gmask].unique()
                vec_val = out_vec[b, gmask].unique()
                assert len(orig_val) == 1 and len(vec_val) == 1, \
                    f"Batch {b} Group {g}: non-uniform values!"
                assert abs(orig_val.item() - vec_val.item()) < 1e-5, \
                    f"Batch {b} Group {g}: orig={orig_val.item():.6f} vec={vec_val.item():.6f}"

    print("  Per-group values match: PASS")

    # Test without groups
    out_orig_ng, _ = compute_depth_normalizers_original(points, mask=mask, groups=None)
    out_vec_ng, _ = compute_depth_normalizers_vectorized(points, mask=mask, groups=None)
    diff_ng = (out_orig_ng - out_vec_ng).abs().max().item()
    print(f"  No-group max diff: {diff_ng:.2e}")
    print(f"  No-group match: PASS" if diff_ng < 1e-6 else f"  No-group FAIL: {diff_ng}")

    # Test without mask
    out_orig_nm, _ = compute_depth_normalizers_original(points, mask=None, groups=groups)
    out_vec_nm, _ = compute_depth_normalizers_vectorized(points, mask=None, groups=groups)
    diff_nm = (out_orig_nm - out_vec_nm).abs().max().item()
    print(f"  No-mask max diff: {diff_nm:.2e}")
    print(f"  No-mask match: PASS" if diff_nm < 1e-6 else f"  No-mask FAIL: {diff_nm}")

    print()


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------
def benchmark(device="cuda"):
    print("=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)

    B, N = 2, 2048
    num_groups = 12

    torch.manual_seed(42)
    points = torch.randn(B, N, 3, device=device) * 5
    points[..., 2] = points[..., 2].abs() + 0.1
    mask = torch.rand(B, N, device=device) > 0.1
    groups = torch.randint(0, num_groups, (B, N), device=device)

    # Warmup
    for _ in range(3):
        compute_depth_normalizers_original(points, mask=mask, groups=groups)
        compute_depth_normalizers_vectorized(points, mask=mask, groups=groups)
    torch.cuda.synchronize()

    # Benchmark original (grouped)
    n_iter = 50
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        compute_depth_normalizers_original(points, mask=mask, groups=groups)
    torch.cuda.synchronize()
    t_orig_grouped = (time.perf_counter() - t0) / n_iter * 1000

    # Benchmark vectorized (grouped)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        compute_depth_normalizers_vectorized(points, mask=mask, groups=groups)
    torch.cuda.synchronize()
    t_vec_grouped = (time.perf_counter() - t0) / n_iter * 1000

    # Benchmark original (no groups)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        compute_depth_normalizers_original(points, mask=mask, groups=None)
    torch.cuda.synchronize()
    t_orig_nogroup = (time.perf_counter() - t0) / n_iter * 1000

    # Benchmark vectorized (no groups)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        compute_depth_normalizers_vectorized(points, mask=mask, groups=None)
    torch.cuda.synchronize()
    t_vec_nogroup = (time.perf_counter() - t0) / n_iter * 1000

    print(f"\n  Config: B={B}, N={N}, num_groups={num_groups}/batch")
    print(f"  Iterations: {n_iter}")
    print()
    print(f"  WITH GROUPS (the hot path):")
    print(f"    Original:   {t_orig_grouped:.3f} ms/call")
    print(f"    Vectorized: {t_vec_grouped:.3f} ms/call")
    print(f"    Speedup:    {t_orig_grouped / t_vec_grouped:.1f}x")
    print()
    print(f"  WITHOUT GROUPS:")
    print(f"    Original:   {t_orig_nogroup:.3f} ms/call")
    print(f"    Vectorized: {t_vec_nogroup:.3f} ms/call")
    print(f"    Speedup:    {t_orig_nogroup / t_vec_nogroup:.1f}x")

    # Test with larger group count
    print()
    print("  --- Larger group count (50 groups/batch) ---")
    groups_large = torch.randint(0, 50, (B, N), device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        compute_depth_normalizers_original(points, mask=mask, groups=groups_large)
    torch.cuda.synchronize()
    t_orig_large = (time.perf_counter() - t0) / n_iter * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        compute_depth_normalizers_vectorized(points, mask=mask, groups=groups_large)
    torch.cuda.synchronize()
    t_vec_large = (time.perf_counter() - t0) / n_iter * 1000

    print(f"    Original:   {t_orig_large:.3f} ms/call")
    print(f"    Vectorized: {t_vec_large:.3f} ms/call")
    print(f"    Speedup:    {t_orig_large / t_vec_large:.1f}x")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU (timings not representative)")
        device = "cpu"
    else:
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name()}")
    test_correctness(device)
    benchmark(device)
