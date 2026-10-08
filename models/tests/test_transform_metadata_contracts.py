"""
Regression / contract tests for transform_metadata and highres patch provider.

Covers the three boundaries identified in the audit:
  1. sampled_highres requires transform_metadata — must raise without it
  2. canonical_space != CROP_NORMALIZED must raise
  3. crop_size_hw happy path: sampled_highres forward runs end-to-end

Run:
    cd /workspace/openclaw/d4rt
    python -m pytest models/tests/test_transform_metadata_contracts.py -v
    # or
    python models/tests/test_transform_metadata_contracts.py
"""

from __future__ import annotations

import os
import re
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.decoder import D4RTDecoder, CANONICAL_QUERY_SPACE_CROP_NORMALIZED


def assert_raises(exc_type, fn, *args, match: str | None = None, **kwargs):
    """Tiny pytest.raises replacement so the test runs without pytest."""
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        if match is not None and re.search(match, str(exc)) is None:
            raise AssertionError(
                f"Expected {exc_type.__name__} message to match {match!r}, got: {exc}"
            ) from exc
        return
    except Exception as exc:  # pragma: no cover - defensive failure path
        raise AssertionError(
            f"Expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"Expected {exc_type.__name__} to be raised")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_decoder(patch_provider: str = "sampled_highres") -> D4RTDecoder:
    return D4RTDecoder(
        embed_dim=64,
        depth=1,
        num_heads=4,
        max_timesteps=16,
        patch_size=3,
        patch_provider=patch_provider,
        num_fourier_freqs=8,
    )


def _make_inputs(B: int = 2, T: int = 4, H: int = 32, W: int = 32, N: int = 8):
    """Return minimal tensors for decoder._embed_query_patches / build_query."""
    frames = torch.rand(B, T, 3, H, W)
    coords = torch.rand(B, N, 2)
    t_src = torch.zeros(B, N, dtype=torch.long)
    return frames, coords, t_src


def _make_transform_metadata(B: int, crop_h: float = 24.0, crop_w: float = 20.0,
                              canonical_space_val: int = CANONICAL_QUERY_SPACE_CROP_NORMALIZED) -> dict:
    return {
        "canonical_space": torch.full((B,), canonical_space_val, dtype=torch.long),
        "original_hw": torch.tensor([[480.0, 640.0]] * B),
        "crop_offset_xy": torch.tensor([[10.0, 8.0]] * B),
        "crop_size_hw": torch.tensor([[crop_h, crop_w]] * B),
        "resized_hw": torch.tensor([[32.0, 32.0]] * B),
    }


# ---------------------------------------------------------------------------
# Test 1: sampled_highres requires transform_metadata
# ---------------------------------------------------------------------------

def test_sampled_highres_requires_transform_metadata():
    """patch_provider='sampled_highres' must raise ValueError when transform_metadata is None."""
    decoder = _make_decoder(patch_provider="sampled_highres")
    frames, coords, t_src = _make_inputs()

    assert_raises(
        ValueError,
        decoder._embed_query_patches,
        frames=frames,
        coords=coords,
        t_src=t_src,
        local_patches=None,
        transform_metadata=None,  # <-- missing
        match="sampled_highres",
    )


# ---------------------------------------------------------------------------
# Test 2: canonical_space != CROP_NORMALIZED must raise
# ---------------------------------------------------------------------------

def test_non_crop_normalized_canonical_space_raises():
    """_validate_transform_metadata must raise when canonical_space != 0."""
    decoder = _make_decoder(patch_provider="sampled_highres")
    B = 2
    bad_metadata = _make_transform_metadata(B, canonical_space_val=1)  # 1 is not supported

    assert_raises(
        ValueError,
        decoder._validate_transform_metadata,
        bad_metadata,
        batch_size=B,
        match="crop-normalized",
    )


# ---------------------------------------------------------------------------
# Test 3: crop_size_hw happy path — sampled_highres forward runs without error
# ---------------------------------------------------------------------------

def test_sampled_highres_forward_happy_path():
    """sampled_highres with valid transform_metadata runs end-to-end and produces correct shapes."""
    B, T, H, W, N = 2, 4, 32, 32, 8
    embed_dim = 64

    decoder = _make_decoder(patch_provider="sampled_highres")
    decoder.eval()

    frames, coords, t_src = _make_inputs(B=B, T=T, H=H, W=W, N=N)
    t_tgt = torch.zeros(B, N, dtype=torch.long)
    t_cam = torch.zeros(B, N, dtype=torch.long)
    encoder_features = torch.rand(B, 16, embed_dim)
    transform_metadata = _make_transform_metadata(B, crop_h=24.0, crop_w=20.0)

    with torch.no_grad():
        out = decoder(
            encoder_features=encoder_features,
            frames=frames,
            coords=coords,
            t_src=t_src,
            t_tgt=t_tgt,
            t_cam=t_cam,
            local_patches=None,
            transform_metadata=transform_metadata,
        )

    assert out["pos_3d"].shape == (B, N, 3), f"pos_3d shape: {out['pos_3d'].shape}"
    assert out["pos_2d"].shape == (B, N, 2), f"pos_2d shape: {out['pos_2d'].shape}"
    assert torch.isfinite(out["pos_3d"]).all(), "pos_3d contains NaN/Inf"
    assert torch.isfinite(out["pos_2d"]).all(), "pos_2d contains NaN/Inf"


# ---------------------------------------------------------------------------
# Test 4: validate_transform_metadata rejects missing required fields
# ---------------------------------------------------------------------------

def test_validate_transform_metadata_missing_field():
    """_validate_transform_metadata raises ValueError for each missing required field."""
    decoder = _make_decoder()
    B = 2
    required_fields = ["canonical_space", "original_hw", "crop_offset_xy", "crop_size_hw", "resized_hw"]

    for missing in required_fields:
        meta = _make_transform_metadata(B)
        del meta[missing]
        assert_raises(
            ValueError,
            decoder._validate_transform_metadata,
            meta,
            batch_size=B,
            match=missing,
        )


# ---------------------------------------------------------------------------
# Test 5: sampled_resized (default) works without transform_metadata
# ---------------------------------------------------------------------------

def test_sampled_resized_does_not_require_transform_metadata():
    """patch_provider='sampled_resized' must not raise when transform_metadata is None."""
    decoder = _make_decoder(patch_provider="sampled_resized")
    frames, coords, t_src = _make_inputs()

    # Should not raise
    result = decoder._embed_query_patches(
        frames=frames,
        coords=coords,
        t_src=t_src,
        local_patches=None,
        transform_metadata=None,
    )
    assert result.shape[0] == frames.shape[0]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_sampled_highres_requires_transform_metadata()
    print("✓ test_sampled_highres_requires_transform_metadata")

    test_non_crop_normalized_canonical_space_raises()
    print("✓ test_non_crop_normalized_canonical_space_raises")

    test_sampled_highres_forward_happy_path()
    print("✓ test_sampled_highres_forward_happy_path")

    test_validate_transform_metadata_missing_field()
    print("✓ test_validate_transform_metadata_missing_field")

    test_sampled_resized_does_not_require_transform_metadata()
    print("✓ test_sampled_resized_does_not_require_transform_metadata")

    print("\nAll transform_metadata contract tests passed.")
