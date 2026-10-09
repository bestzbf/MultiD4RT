"""Frozen legacy depth protocol; independent of the D4RT model."""
import torch

def _per_frame_align_depth(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    max_depth: float | None = None,
) -> torch.Tensor:
    """Per-frame depth alignment, MonST3R/DUSt3R/CUT3R Sintel protocol.

    mode="scale": scale_t = median(gt_t / pred_t) over valid pixels of frame t.
    mode="scale_shift": robust median-centered scale + median-residual shift,
    fitted independently per frame. Each frame gets its own (s, t).
    """
    out = pred.clone()
    if max_depth is not None:
        out = out.clamp(min=1e-6, max=float(max_depth))
    T = pred.shape[0]
    for t in range(T):
        valid_t = mask[t]
        if not valid_t.any():
            continue
        p_t = out[t][valid_t].clamp(min=1e-6)
        g_t = gt[t][valid_t]
        if mode == "scale":
            scale = (g_t / p_t).median()
            if torch.isfinite(scale) and scale > 0:
                out[t] = out[t] * scale
        elif mode == "scale_shift":
            p_med = p_t.median()
            g_med = g_t.median()
            p_centered = p_t - p_med + 1e-6
            g_centered = g_t - g_med + 1e-6
            finite = torch.isfinite(p_centered) & torch.isfinite(g_centered) & (p_centered.abs() > 1e-6)
            if not finite.any():
                scale = (g_t / p_t).median()
                if torch.isfinite(scale) and scale > 0:
                    out[t] = out[t] * scale
                continue
            scale = (g_centered[finite] / p_centered[finite]).median()
            if not torch.isfinite(scale) or scale <= 0:
                scale = (g_t / p_t).median()
                if torch.isfinite(scale) and scale > 0:
                    out[t] = out[t] * scale
                continue
            shift = (g_t - scale * p_t).median()
            out[t] = out[t] * scale + shift
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return out
