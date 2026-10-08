"""Loss functions for D4RT training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

def compute_depth_normalizers(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    groups: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the median-depth denominators used by scale-invariant 3D losses.

    Uses median instead of mean so that a small number of far-away valid points
    (large Z) cannot inflate the normalizer and suppress the loss for nearby points.
    """
    z_coords = points[..., 2]  # (B, N)

    if mask is not None:
        valid_mask = mask.bool()
    else:
        valid_mask = torch.ones_like(z_coords, dtype=torch.bool)

    if groups is not None:
        if groups.shape != z_coords.shape:
            raise ValueError(
                f"Expected groups to match points batch/query shape, got {tuple(groups.shape)} "
                f"for points {tuple(points.shape)}"
            )
        B, N = z_coords.shape
        out = torch.ones_like(z_coords)
        # Per-batch vectorized median; B is small, each iteration stays GPU-vectorized.
        for batch_idx in range(B):
            z_b = z_coords[batch_idx]
            g_b = groups[batch_idx]
            v_b = valid_mask[batch_idx]

            valid_z = z_b[v_b]
            valid_g = g_b[v_b]

            if valid_z.numel() == 0:
                continue

            num_groups = int(g_b.max().item()) + 1

            # Sort by group → same-group elements contiguous
            group_order = valid_g.argsort()
            sorted_z = valid_z[group_order]
            sorted_g = valid_g[group_order]

            group_sizes = torch.bincount(valid_g, minlength=num_groups)
            group_starts = torch.cumsum(group_sizes, 0) - group_sizes


            z_min, z_max = sorted_z.min(), sorted_z.max()
            z_range = z_max - z_min
            if z_range > 0:
                z_norm = (sorted_z - z_min) / z_range
            else:
                z_norm = torch.zeros_like(sorted_z)
            key = sorted_g.to(torch.double) * 2.0 + z_norm.to(torch.double)
            segment_order = key.argsort()
            segment_z = sorted_z[segment_order]

            # Lower median matches torch.Tensor.median() for even-sized groups.
            median_pos = group_starts + (group_sizes - 1) // 2
            median_pos = median_pos.clamp(max=len(segment_z) - 1)
            median_vals = segment_z[median_pos]

            empty = group_sizes == 0
            median_vals = median_vals.masked_fill(empty, 1.0)
            median_vals = torch.nan_to_num(median_vals, nan=1.0, posinf=1.0, neginf=1.0)
            median_vals = torch.clamp(torch.abs(median_vals), min=eps)

            out[batch_idx, v_b] = median_vals[valid_g]

        return out, valid_mask

    if mask is not None:
        # median over valid points only
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


def normalize_points(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    groups: Optional[torch.Tensor] = None,
    eps: float = 1e-6
) -> torch.Tensor:
    """Normalize points by their median depth, accounting for invalid points.

    Args:
        points: (B, N, 3) point positions
        mask: (B, N) optional validity mask
        groups: (B, N) optional group ids. When provided, median-depth normalization is
            computed independently per group inside each batch element.
        eps: Small value for numerical stability

    Returns:
        normalized_points: (B, N, 3) normalized points
    """
    mean_depth_safe, valid_mask = compute_depth_normalizers(
        points,
        mask=mask,
        groups=groups,
        eps=eps,
    )

    if valid_mask is not None:
        points = torch.where(valid_mask.unsqueeze(-1), points, torch.zeros_like(points))

    normalized = points / mean_depth_safe.unsqueeze(-1)
    return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_points_by(
    points: torch.Tensor,
    normalizer: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Normalize points by a pre-computed per-point depth denominator.

    Unlike ``normalize_points`` which computes the denominator from *points*
    itself, this helper accepts an external ``normalizer`` tensor (typically
    derived from the **target** points).  This keeps depth-scale information
    in the loss so that systematic scale errors produce non-zero gradients.

    Args:
        points: (B, N, 3) point positions
        normalizer: (B, N) per-point depth denominator (e.g. from
            ``compute_depth_normalizers(target, ...)``).  Must already be
            positive and clamped.
        mask: (B, N) optional validity mask.  Invalid points are zeroed.

    Returns:
        normalized_points: (B, N, 3)
    """
    if mask is not None:
        points = torch.where(mask.bool().unsqueeze(-1), points, torch.zeros_like(points))
    normalized = points / normalizer.unsqueeze(-1)
    return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def compute_depth_statistics(
    points: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    groups: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute raw absolute depth and effective normalization depth for logging."""
    mean_depth_safe, valid_mask = compute_depth_normalizers(
        points,
        mask=mask,
        groups=groups,
        eps=eps,
    )
    abs_depth = torch.abs(points[..., 2])
    abs_depth = torch.where(valid_mask, abs_depth, torch.zeros_like(abs_depth))
    mean_depth_safe = torch.where(valid_mask, mean_depth_safe, torch.zeros_like(mean_depth_safe))
    denom = valid_mask.sum().to(points.dtype) + eps
    return abs_depth.sum() / denom, mean_depth_safe.sum() / denom

# # ==========================================
# # 下方代码替换 D4RTLoss.forward 中的 3D Loss 计算部分
# # ==========================================

# # 获取 confidence
# confidence = predictions['confidence'].squeeze(-1)  # (B, N)

# # 1. 带 Mask 的归一化
# pred_norm = normalize_points(predictions['pos_3d'], mask=mask_3d)
# target_norm = normalize_points(targets['pos_3d'], mask=mask_3d)

# # 2. 对数抑制变换 sign(x) * log(1 + |x|)
# pred_log = log_transform(pred_norm)
# target_log = log_transform(target_norm)

# # 3. 计算单个点的 L1 距离 (在特征维度 x,y,z 上取平均)
# point_loss = torch.abs(pred_log - target_log).mean(dim=-1)  # (B, N)

# # 4. 强制进行置信度加权 (对应公式: c * L_3D)
# weighted_loss = confidence * point_loss  # (B, N)

# # 5. Mask 过滤与聚合
# if mask_3d is not None:
#     losses['loss_3d'] = (weighted_loss * mask_3d).sum() / (mask_3d.sum() + 1e-6)
# else:
#     losses['loss_3d'] = weighted_loss.mean()



def compute_3d_loss_logspace(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depth-invariant 3D loss in log/angular space (方案B).

    Z axis:  |log(z_pred) - log(z_gt)|   — relative depth error, gradient ∝ 1/z.
    XY axes: |x/z_pred - x/z_gt|         — angular (projection-plane) error, also ∝ 1/z.

    Both terms have uniform gradient magnitude across all depths.  No median
    normalization is needed because the loss is already scale-invariant per point.

    Args:
        pred:   (B, N, 3) predicted camera-space 3D positions
        target: (B, N, 3) target camera-space 3D positions
        mask:   (B, N) optional validity mask
        eps:    minimum Z clamp to avoid log(0)

    Returns:
        point_loss: (B, N) per-point loss (for confidence weighting upstream)
        scalar_loss: scalar mean loss
    """
    z_pred = pred[..., 2].clamp(min=eps)
    z_gt   = target[..., 2].clamp(min=eps)

    # Z: log-space relative depth error
    loss_z = torch.abs(torch.log(z_pred) - torch.log(z_gt))

    # XY: angular error (depth-normalised projection-plane deviation)
    loss_x = torch.abs(pred[..., 0] / z_pred - target[..., 0] / z_gt)
    loss_y = torch.abs(pred[..., 1] / z_pred - target[..., 1] / z_gt)

    point_loss = (loss_x + loss_y + loss_z) / 3.0  # (B, N)

    if mask is not None:
        valid_mask = mask.bool()
        point_loss_masked = torch.where(valid_mask, point_loss, torch.zeros_like(point_loss))
        scalar = point_loss_masked.sum() / (valid_mask.sum() + 1e-6)
    else:
        scalar = point_loss.mean()

    return point_loss, scalar


def log_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply log transform to dampen influence of far points.

    Transform: sign(x) * log(1 + |x|)

    Args:
        x: Input tensor

    Returns:
        Transformed tensor
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def compute_3d_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    groups: Optional[torch.Tensor] = None,
    normalize: bool = True,
    use_log_transform: bool = True,
    shared_depth_normalization: bool = False,
) -> torch.Tensor:
    """Compute L1 loss on 3D positions.

    Args:
        pred: (B, N, 3) predicted 3D positions
        target: (B, N, 3) target 3D positions
        mask: (B, N) optional validity mask
        groups: (B, N) optional group ids used for per-group median-depth normalization
        normalize: Whether to normalize by median depth
        use_log_transform: Whether to apply log transform
        shared_depth_normalization: If True, use target median depth for both
            pred and target (scale-aware).  If False, each is normalized by
            its own median depth (paper default, scale-invariant).

    Returns:
        loss: Scalar loss value
    """
    if normalize:
        if shared_depth_normalization:
            tgt_normalizer, _ = compute_depth_normalizers(target, mask=mask, groups=groups)
            pred = normalize_points_by(pred, tgt_normalizer, mask=mask)
            target = normalize_points_by(target, tgt_normalizer, mask=mask)
        else:
            pred = normalize_points(pred, mask=mask, groups=groups)
            target = normalize_points(target, mask=mask, groups=groups)

    if use_log_transform:
        pred = log_transform(pred)
        target = log_transform(target)

    # L1 loss
    loss = torch.abs(pred - target)

    if mask is not None:
        valid_mask = mask.bool().unsqueeze(-1).expand_as(loss)
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        return loss.sum() / (valid_mask.sum() + 1e-6)

    return loss.mean()


def compute_2d_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute L1 loss on 2D reprojection.

    Args:
        pred: (B, N, 2) predicted 2D positions
        target: (B, N, 2) target 2D positions
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar loss value
    """
    loss = torch.abs(pred - target)

    if mask is not None:
        valid_mask = mask.bool().unsqueeze(-1).expand_as(loss)
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        return loss.sum() / (valid_mask.sum() + 1e-6)

    return loss.mean()


def compute_visibility_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute binary cross-entropy loss for visibility prediction.

    Args:
        pred: (B, N, 1) predicted visibility logits
        target: (B, N) target visibility (0 or 1)
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar loss value
    """
    pred = pred.squeeze(-1)  # (B, N)

    loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')

    if mask is not None:
        valid_mask = mask.bool()
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        return loss.sum() / (valid_mask.sum() + 1e-6)

    return loss.mean()


def compute_displacement_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute L1 loss on motion displacement.

    Args:
        pred: (B, N, 3) predicted displacement
        target: (B, N, 3) target displacement
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar loss value
    """
    loss = torch.abs(pred - target)

    if mask is not None:
        valid_mask = mask.bool().unsqueeze(-1).expand_as(loss)
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        return loss.sum() / (valid_mask.sum() + 1e-6)

    return loss.mean()


def compute_normal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute cosine similarity loss for surface normals.

    Args:
        pred: (B, N, 3) predicted normals (should be normalized)
        target: (B, N, 3) target normals (should be normalized)
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar loss value (1 - cosine_similarity)
    """
    # Normalize to unit vectors
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)

    # Cosine similarity
    cos_sim = (pred * target).sum(dim=-1)  # (B, N)

    # Loss is 1 - cosine_similarity
    loss = 1.0 - cos_sim

    if mask is not None:
        valid_mask = mask.bool()
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        return loss.sum() / (valid_mask.sum() + 1e-6)

    return loss.mean()


def compute_confidence_loss(
    confidence: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute the Kendall confidence penalty term.

    Kendall log-parameterization: s = log(σ²), σ² is the predicted variance.
    Full loss per point: exp(-s) * L_3D + s  ≡  c * L_3D - log(c)  where c = exp(-s).
    This function computes only the penalty term: s = log(σ²) = -log(c).
    The optimal s* = log(L_3D / λ_conf), so s* < 0 at convergence when loss_3d < lambda_conf.

    Args:
        confidence: (B, N, 1) s = log(σ²) from the network (uncertainty output)
        mask: (B, N) optional validity mask

    Returns:
        loss: Scalar mean of s over valid points
    """
    s = confidence.squeeze(-1)  # (B, N)

    # Penalty = s = log(σ²); negative at convergence (high confidence) is expected
    loss = s

    if mask is not None:
        valid_mask = mask.bool()
        loss = torch.where(valid_mask, loss, torch.zeros_like(loss))
        return loss.sum() / (valid_mask.sum() + 1e-6)

    return loss.mean()


def compute_raw_3d_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute raw camera-space 3D errors for logging only."""
    l1 = torch.abs(pred - target).mean(dim=-1)
    euclidean = torch.linalg.norm(pred - target, dim=-1)

    if mask is not None:
        valid_mask = mask.bool()
        l1 = torch.where(valid_mask, l1, torch.zeros_like(l1))
        euclidean = torch.where(valid_mask, euclidean, torch.zeros_like(euclidean))
        denom = valid_mask.sum() + 1e-6
        return l1.sum() / denom, euclidean.sum() / denom

    return l1.mean(), euclidean.mean()


def masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Return the mean of values over a boolean mask, or zero when empty."""
    if mask is None:
        return values.mean()
    valid_mask = mask.bool()
    values = torch.where(valid_mask, values, torch.zeros_like(values))
    return values.sum() / (valid_mask.sum() + 1e-6)


class D4RTLoss(nn.Module):
    """Combined loss function for D4RT training.

    Computes weighted sum of:
    - L_3D: L1 loss on normalized, log-transformed 3D positions (weighted by confidence)
    - L_raw3D: optional L1 loss on raw camera-space 3D positions
    - L_2D: L1 loss on 2D reprojection
    - L_vis: BCE loss for visibility
    - L_disp: L1 loss on motion displacement
    - L_normal: Cosine similarity loss for surface normals
    - L_conf: Kendall penalty s = log(σ²), weighted by lambda_conf
    """

    def __init__(
        self,
        lambda_3d: float = 1.0,
        lambda_raw_3d: float = 0.0,
        lambda_2d: float = 0.1,
        lambda_vis: float = 0.1,
        lambda_disp: float = 0.1,
        lambda_normal: float = 0.5,
        lambda_conf: float = 0.2,
        use_confidence_weighting: Optional[bool] = None,
        debug_3d_loss_mode: str = "scale_invariant",
        static_reprojection_weight: float = 1.0,
        shared_depth_normalization: bool = False,
        dust3r_metric_branch: bool = False,
    ):
        super().__init__()
        self.lambda_3d = lambda_3d
        self.lambda_raw_3d = lambda_raw_3d
        self.lambda_2d = lambda_2d
        self.lambda_vis = lambda_vis
        self.lambda_disp = lambda_disp
        self.lambda_normal = lambda_normal
        self.lambda_conf = lambda_conf
        self.static_reprojection_weight = static_reprojection_weight
        self.shared_depth_normalization = shared_depth_normalization
        # DUSt3R-style two-branch normalization: when True, the per-sample
        # is_metric flag (passed to forward) selects between
        #   - metric=True  -> shared normalizer z̄=z (scale-anchoring)
        #   - metric=False -> independent z, z̄        (scale-invariant, paper default)
        # When False (default), behaviour is unchanged: every sample uses
        # `shared_depth_normalization` globally.
        self.dust3r_metric_branch = dust3r_metric_branch
        if debug_3d_loss_mode not in {"scale_invariant", "raw_l1", "log_space"}:
            raise ValueError(
                f"Unsupported debug_3d_loss_mode={debug_3d_loss_mode!r}. "
                "Expected one of {'scale_invariant', 'raw_l1', 'log_space'}."
            )
        self.debug_3d_loss_mode = debug_3d_loss_mode
        self.use_confidence_weighting = (
            lambda_conf > 0.0 if use_confidence_weighting is None else use_confidence_weighting
        )
        self.confidence_weighting_factor = 1.0 if self.use_confidence_weighting else 0.0

    def _reduce_point_loss(
        self,
        point_loss: torch.Tensor,
        uncertainty: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply Kendall confidence weighting exp(-s) * L_3D.

        The s penalty term (lambda_conf * s) is computed and added separately
        in forward(), preserving the correct two-term Kendall balance:
            lambda_3d * exp(-s) * L_3D + lambda_conf * s
        Optimal s* = log(L_3D / lambda_conf), minimum value = lambda_conf * (1 + log(L/lambda_conf)).
        This stays positive as long as L_3D > lambda_conf / e.
        """
        if self.use_confidence_weighting and self.lambda_conf > 0:
            confidence_weight = torch.exp(-uncertainty)
            confidence_weight = 1.0 + self.confidence_weighting_factor * (confidence_weight - 1.0)
            point_loss = confidence_weight * point_loss

        if mask is not None:
            valid_mask = mask.bool()
            point_loss = torch.where(valid_mask, point_loss, torch.zeros_like(point_loss))
            return point_loss.sum() / (valid_mask.sum() + 1e-6)

        return point_loss.mean()

    def set_confidence_schedule(self, lambda_conf: float, weighting_factor: Optional[float] = None) -> None:
        """Set confidence loss weight and weighting factor for gradual ramp-up.

        Args:
            lambda_conf: Weight for confidence penalty loss
            weighting_factor: Factor in [0,1] for gradual confidence weighting ramp-up.
                            0.0 = no weighting (weight=1.0), 1.0 = full Kendall weighting (weight=exp(-s))
        """
        self.lambda_conf = float(lambda_conf)
        if weighting_factor is not None:
            self.confidence_weighting_factor = float(weighting_factor)
        else:
            self.confidence_weighting_factor = 1.0
        self.use_confidence_weighting = True

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        normalize_groups: Optional[torch.Tensor] = None,
        is_metric: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            predictions: Dictionary with model outputs:
                - pos_3d: (B, N, 3) predicted 3D positions
                - pos_2d: (B, N, 2) predicted 2D positions
                - visibility: (B, N, 1) predicted visibility logits
                - displacement: (B, N, 3) predicted displacement
                - normal: (B, N, 3) predicted normals
                - confidence: (B, N, 1) predicted positive confidence

            targets: Dictionary with ground truth:
                - pos_3d: (B, N, 3) target 3D positions
                - pos_2d: (B, N, 2) target 2D positions (optional)
                - visibility: (B, N) target visibility
                - displacement: (B, N, 3) target displacement (optional)
                - normal: (B, N, 3) target normals (optional)
                - mask_3d: (B, N) validity mask for 3D loss
                - mask_2d: (B, N) validity mask for 2D loss (optional)
                - mask_vis: (B, N) validity mask for visibility (optional)
                - mask_disp: (B, N) validity mask for displacement (optional)
                - mask_normal: (B, N) validity mask for normals (optional)
            is_metric: (B,) bool tensor. Only consulted when
                ``self.dust3r_metric_branch`` is True. For samples where
                is_metric=True, the 3D loss uses the target normalizer for
                both pred and target (DUSt3R metric branch, z̄=z). For
                samples where is_metric=False, each is normalized by its own
                median depth (paper default, scale-invariant). When None or
                when dust3r_metric_branch=False, falls back to the existing
                global ``shared_depth_normalization`` behaviour.

        Returns:
            Dictionary with:
                - loss: Total weighted loss
                - loss_3d: 3D position loss
                - loss_2d: 2D position loss
                - loss_vis: Visibility loss
                - loss_disp: Displacement loss
                - loss_normal: Normal loss
                - loss_conf: Confidence penalty
        """
        losses = {}

        # Get masks (default to all valid if not provided)
        mask_3d = targets.get('mask_3d')
        mask_2d = targets.get('mask_2d', mask_3d)
        mask_vis = targets.get('mask_vis', mask_3d)
        mask_disp = targets.get('mask_disp')
        mask_normal = targets.get('mask_normal')
        static_mask = targets.get('is_static_reprojection')
        if static_mask is not None:
            static_mask = static_mask.bool()
        valid_mask_3d = mask_3d.bool() if mask_3d is not None else None
        if mask_normal is not None:
            mask_normal = mask_normal.bool()
            if valid_mask_3d is not None:
                mask_normal = mask_normal & valid_mask_3d

        metric_dtype = predictions['pos_3d'].dtype
        metric_device = predictions['pos_3d'].device
        zero_metric = torch.tensor(0.0, device=metric_device, dtype=metric_dtype)

        losses['metric_static_query_ratio'] = (
            static_mask.float().mean().to(metric_dtype) if static_mask is not None else zero_metric
        )
        losses['metric_temporal_query_ratio'] = (
            (~static_mask).float().mean().to(metric_dtype) if static_mask is not None else zero_metric
        )
        losses['metric_normal_query_ratio'] = (
            mask_normal.float().mean().to(metric_dtype) if mask_normal is not None else zero_metric
        )
        point_loss_for_metrics: Optional[torch.Tensor] = None

        # Per-point weight for static reprojection queries (has_tracks=False).
        # Default weight=1.0 → identical to previous behaviour.
        if static_mask is not None and self.static_reprojection_weight != 1.0:
            _w = predictions['pos_3d'].new_full(static_mask.shape, self.static_reprojection_weight)
            static_point_weight: Optional[torch.Tensor] = torch.where(static_mask, _w, torch.ones_like(_w))
        else:
            static_point_weight = None

        # 3D position loss. Confidence weighting is only enabled once the
        # confidence penalty itself participates in optimization; otherwise the
        # model can minimize this term by collapsing confidence toward zero.
        uncertainty = predictions['uncertainty'].squeeze(-1)  # (B, N) - Kendall's s

        if self.debug_3d_loss_mode == "raw_l1":
            point_loss = torch.abs(predictions['pos_3d'] - targets['pos_3d']).mean(dim=-1)
            point_loss_for_metrics = point_loss
            # Unweighted version (no static weight — kept as pure metric)
            if valid_mask_3d is not None:
                unweighted = torch.where(valid_mask_3d, point_loss, torch.zeros_like(point_loss))
                losses['loss_3d_unweighted'] = unweighted.sum() / (valid_mask_3d.sum() + 1e-6)
            else:
                losses['loss_3d_unweighted'] = point_loss.mean()
            weighted_point_loss = point_loss * static_point_weight if static_point_weight is not None else point_loss
            losses['loss_3d'] = self._reduce_point_loss(
                weighted_point_loss,
                uncertainty,
                mask=mask_3d,
            )
        elif self.debug_3d_loss_mode == "log_space":
            # 方案B: depth-invariant log/angular loss — no median normalization needed.
            point_loss, _ = compute_3d_loss_logspace(
                predictions['pos_3d'], targets['pos_3d'], mask=mask_3d
            )
            point_loss_for_metrics = point_loss
            if valid_mask_3d is not None:
                unweighted = torch.where(valid_mask_3d, point_loss, torch.zeros_like(point_loss))
                losses['loss_3d_unweighted'] = unweighted.sum() / (valid_mask_3d.sum() + 1e-6)
            else:
                losses['loss_3d_unweighted'] = point_loss.mean()
            weighted_point_loss = point_loss * static_point_weight if static_point_weight is not None else point_loss
            losses['loss_3d'] = self._reduce_point_loss(
                weighted_point_loss,
                uncertainty,
                mask=mask_3d,
            )
        elif mask_3d is not None:
            use_metric_branch = (
                self.dust3r_metric_branch
                and is_metric is not None
                and bool(is_metric.any().item())
            )
            if use_metric_branch:
                # DUSt3R two-branch normalization:
                #   metric=True  -> shared (z̄=z) using target normalizer for both
                #   metric=False -> independent (each by own median depth)
                # Both branches are computed in vectorized form and combined per sample.
                tgt_normalizer, _ = compute_depth_normalizers(
                    targets['pos_3d'], mask=mask_3d, groups=normalize_groups
                )
                pred_shared = normalize_points_by(
                    predictions['pos_3d'], tgt_normalizer, mask=mask_3d
                )
                target_shared = normalize_points_by(
                    targets['pos_3d'], tgt_normalizer, mask=mask_3d
                )
                pred_indep = normalize_points(
                    predictions['pos_3d'], mask=mask_3d, groups=normalize_groups
                )
                target_indep = normalize_points(
                    targets['pos_3d'], mask=mask_3d, groups=normalize_groups
                )
                # is_metric is (B,) bool/0-1. Broadcast to (B, 1, 1) for point shape (B, N, 3).
                m = is_metric.to(predictions['pos_3d'].dtype).view(-1, 1, 1)
                pred_norm = m * pred_shared + (1.0 - m) * pred_indep
                target_norm = m * target_shared + (1.0 - m) * target_indep
                # Log per-sample metric ratio for monitoring.
                losses['metric_dust3r_metric_ratio'] = (
                    is_metric.float().mean().to(metric_dtype)
                )
            elif self.shared_depth_normalization:
                # Shared normalization: compute median depth from *target* only and
                # apply it to both pred and target.  This preserves depth-scale
                # information so the model receives gradients when its predictions
                # have a systematic scale offset.
                tgt_normalizer, _ = compute_depth_normalizers(
                    targets['pos_3d'], mask=mask_3d, groups=normalize_groups
                )
                pred_norm = normalize_points_by(predictions['pos_3d'], tgt_normalizer, mask=mask_3d)
                target_norm = normalize_points_by(targets['pos_3d'], tgt_normalizer, mask=mask_3d)
            else:
                # Independent normalization (paper default): pred and target are
                # each divided by their own median depth.  Scale-invariant but
                # blind to systematic depth-scale offsets.
                pred_norm = normalize_points(predictions['pos_3d'], mask=mask_3d, groups=normalize_groups)
                target_norm = normalize_points(targets['pos_3d'], mask=mask_3d, groups=normalize_groups)
            pred_log = log_transform(pred_norm)
            target_log = log_transform(target_norm)

            point_loss = torch.abs(pred_log - target_log).mean(dim=-1)  # (B, N)
            point_loss_for_metrics = point_loss
            # Unweighted version (no static weight — kept as pure metric)
            unweighted = torch.where(valid_mask_3d, point_loss, torch.zeros_like(point_loss))
            losses['loss_3d_unweighted'] = unweighted.sum() / (valid_mask_3d.sum() + 1e-6)
            weighted_point_loss = point_loss * static_point_weight if static_point_weight is not None else point_loss
            losses['loss_3d'] = self._reduce_point_loss(
                weighted_point_loss,
                uncertainty,
                mask=mask_3d,
            )
        else:
            losses['loss_3d'] = compute_3d_loss(
                predictions['pos_3d'],
                targets['pos_3d'],
                mask=None,
                groups=normalize_groups,
                normalize=True,
                use_log_transform=True,
                shared_depth_normalization=self.shared_depth_normalization,
            )
            losses['loss_3d_unweighted'] = losses['loss_3d']

        losses['loss_raw_3d'] = compute_3d_loss(
            predictions['pos_3d'],
            targets['pos_3d'],
            mask=mask_3d,
            groups=None,
            normalize=False,
            use_log_transform=False,
        )

        # 2D position loss (if target provided)
        if 'pos_2d' in targets:
            losses['loss_2d'] = compute_2d_loss(
                predictions['pos_2d'],
                targets['pos_2d'],
                mask=mask_2d
            )
        else:
            losses['loss_2d'] = torch.tensor(0.0, device=predictions['pos_3d'].device)

        # Visibility loss
        if 'visibility' in targets:
            losses['loss_vis'] = compute_visibility_loss(
                predictions['visibility'],
                targets['visibility'],
                mask=mask_vis
            )
        else:
            losses['loss_vis'] = torch.tensor(0.0, device=predictions['pos_3d'].device)

        # Displacement loss (if target provided)
        if 'displacement' in targets and mask_disp is not None:
            losses['loss_disp'] = compute_displacement_loss(
                predictions['displacement'],
                targets['displacement'],
                mask=mask_disp
            )
        else:
            losses['loss_disp'] = torch.tensor(0.0, device=predictions['pos_3d'].device)

        # Normal loss (if target provided)
        if 'normal' in targets and mask_normal is not None:
            losses['loss_normal'] = compute_normal_loss(
                predictions['normal'],
                targets['normal'],
                mask=mask_normal
            )
        else:
            losses['loss_normal'] = torch.tensor(0.0, device=predictions['pos_3d'].device)

        # Confidence penalty (s term of Kendall loss, weighted separately by lambda_conf)
        losses['loss_conf'] = compute_confidence_loss(
            predictions['uncertainty'],
            mask=mask_3d
        )

        raw_3d_l1, raw_3d_euclidean = compute_raw_3d_metrics(
            predictions['pos_3d'],
            targets['pos_3d'],
            mask=mask_3d,
        )
        losses['metric_raw_3d_l1'] = raw_3d_l1
        losses['metric_raw_3d_euclidean'] = raw_3d_euclidean

        if valid_mask_3d is not None:
            losses['metric_valid_3d_query_ratio'] = valid_mask_3d.float().mean().to(metric_dtype)
        else:
            losses['metric_valid_3d_query_ratio'] = zero_metric

        if static_mask is not None and valid_mask_3d is not None:
            static_valid_3d = valid_mask_3d & static_mask
            temporal_valid_3d = valid_mask_3d & (~static_mask)
            valid_3d_denom = valid_mask_3d.sum().to(metric_dtype) + 1e-6
            losses['metric_static_valid3d_ratio'] = static_valid_3d.sum().to(metric_dtype) / valid_3d_denom
            losses['metric_temporal_valid3d_ratio'] = temporal_valid_3d.sum().to(metric_dtype) / valid_3d_denom
            if point_loss_for_metrics is not None:
                losses['metric_loss_3d_static_unweighted'] = masked_mean(point_loss_for_metrics, static_valid_3d)
                losses['metric_loss_3d_temporal_unweighted'] = masked_mean(point_loss_for_metrics, temporal_valid_3d)
            else:
                losses['metric_loss_3d_static_unweighted'] = zero_metric
                losses['metric_loss_3d_temporal_unweighted'] = zero_metric

            raw_l1_per_point = torch.abs(predictions['pos_3d'] - targets['pos_3d']).mean(dim=-1)
            raw_euc_per_point = torch.linalg.norm(predictions['pos_3d'] - targets['pos_3d'], dim=-1)
            losses['metric_raw_3d_l1_static'] = masked_mean(raw_l1_per_point, static_valid_3d)
            losses['metric_raw_3d_l1_temporal'] = masked_mean(raw_l1_per_point, temporal_valid_3d)
            losses['metric_raw_3d_euclidean_static'] = masked_mean(raw_euc_per_point, static_valid_3d)
            losses['metric_raw_3d_euclidean_temporal'] = masked_mean(raw_euc_per_point, temporal_valid_3d)
        else:
            losses['metric_static_valid3d_ratio'] = zero_metric
            losses['metric_temporal_valid3d_ratio'] = zero_metric
            losses['metric_loss_3d_static_unweighted'] = zero_metric
            losses['metric_loss_3d_temporal_unweighted'] = zero_metric
            losses['metric_raw_3d_l1_static'] = zero_metric
            losses['metric_raw_3d_l1_temporal'] = zero_metric
            losses['metric_raw_3d_euclidean_static'] = zero_metric
            losses['metric_raw_3d_euclidean_temporal'] = zero_metric

        if mask_normal is not None:
            losses['metric_normal_valid3d_ratio'] = (
                (mask_normal.bool() & valid_mask_3d).sum().to(metric_dtype) / (valid_mask_3d.sum().to(metric_dtype) + 1e-6)
                if valid_mask_3d is not None else mask_normal.float().mean().to(metric_dtype)
            )
        else:
            losses['metric_normal_valid3d_ratio'] = zero_metric

        pred_abs_depth_mean, pred_norm_depth_mean = compute_depth_statistics(
            predictions['pos_3d'],
            mask=mask_3d,
            groups=normalize_groups,
        )
        target_abs_depth_mean, target_norm_depth_mean = compute_depth_statistics(
            targets['pos_3d'],
            mask=mask_3d,
            groups=normalize_groups,
        )
        losses['metric_pred_abs_depth_mean'] = pred_abs_depth_mean
        losses['metric_target_abs_depth_mean'] = target_abs_depth_mean
        losses['metric_pred_norm_depth_mean'] = pred_norm_depth_mean
        losses['metric_target_norm_depth_mean'] = target_norm_depth_mean
        if mask_3d is not None:
            valid_mask_3d = mask_3d.bool()
            conf_mean = torch.where(valid_mask_3d, uncertainty, torch.zeros_like(uncertainty)).sum()
            conf_mean = conf_mean / (valid_mask_3d.sum() + 1e-6)
        else:
            conf_mean = uncertainty.mean()
        losses['metric_conf_mean'] = conf_mean

        # Total loss
        losses['loss'] = (
            self.lambda_3d * losses['loss_3d'] +
            self.lambda_raw_3d * losses['loss_raw_3d'] +
            self.lambda_2d * losses['loss_2d'] +
            self.lambda_vis * losses['loss_vis'] +
            self.lambda_disp * losses['loss_disp'] +
            self.lambda_normal * losses['loss_normal'] +
            self.lambda_conf * losses['loss_conf']
        )

        return losses


class DepthLoss(nn.Module):
    """Loss function for depth estimation evaluation."""

    def __init__(self, scale_invariant: bool = True):
        super().__init__()
        self.scale_invariant = scale_invariant

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        """Compute depth metrics.

        Args:
            pred: (B, H, W) predicted depth
            target: (B, H, W) target depth
            mask: (B, H, W) validity mask

        Returns:
            Dictionary with metrics
        """
        if mask is None:
            mask = torch.ones_like(pred, dtype=torch.bool)

        # Flatten
        pred_flat = pred[mask]
        target_flat = target[mask]

        if self.scale_invariant:
            # Scale alignment
            scale = (target_flat / (pred_flat + 1e-6)).median()
            pred_flat = pred_flat * scale

        # AbsRel
        abs_rel = torch.abs(pred_flat - target_flat) / (target_flat + 1e-6)
        abs_rel = abs_rel.mean()

        # RMSE
        rmse = torch.sqrt(((pred_flat - target_flat) ** 2).mean())

        # Log RMSE
        log_rmse = torch.sqrt(((torch.log(pred_flat + 1e-6) - torch.log(target_flat + 1e-6)) ** 2).mean())

        return {
            'abs_rel': abs_rel,
            'rmse': rmse,
            'log_rmse': log_rmse
        }
