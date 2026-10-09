"""Check the actual production widths against 4RC without allocating giant weights."""

import torch
from torch import nn

from multid4rt import MultiD4RT
from multid4rt.checkpoint import _is_extension_key
from multid4rt.vendor.dinov2.dinov2 import DinoV2
from multid4rt.vendor.heads.cam_dec import CameraDec
from multid4rt.vendor.heads.dpt_head import DPTHead
from multid4rt.vendor.heads.dualdpt import DualDPT
from multid4rt.vendor.heads.motiondecoder import MotionDecoder


def test_giant_preserves_all_original_arc_parameter_names_and_shapes():
    # Build the five original Arc modules with its published constructor values.
    # Meta tensors verify a 1.5B-parameter model without a multi-GB CPU allocation.
    with torch.device("meta"):
        model = MultiD4RT()
        original = nn.Module()
        original.backbone = DinoV2(name="vitg", out_layers=[19, 27, 33, 39],
                                  alt_start=13, qknorm_start=13, rope_start=13,
                                  cat_token=True, has_time_token=True)
        original.head = DualDPT(dim_in=3072, output_dim=2, features=256)
        original.cam_dec = CameraDec(dim_in=3072)
        original.motion_decoder = MotionDecoder(patch_size=14, embed_dim=1536,
                                                use_adaln=True, depth=4)
        original.track_head = DPTHead(dim_in=1536, output_dim=4, activation="inv_log",
                                     conf_activation="expp1", intermediate_layer_idx=[0, 1, 2, 3])
    old, new = original.state_dict(), model.state_dict()
    assert len(old) == 1078
    for name, parameter in old.items():
        assert name in new
        assert parameter.shape == new[name].shape, name
    assert all(_is_extension_key(name) for name in set(new)-set(old))
    assert model.backbone.pretrained.embed_dim == 1536
    assert len(model.backbone.pretrained.blocks) == 40
    assert len(model.motion_decoder.cross_blocks) == 4
