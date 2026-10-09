"""Decoder, checkpoint and dense-head guarantees for synchronized views."""

import copy

import pytest
import torch

from multid4rt.motion import MotionDPT, MultiViewMotionDecoder
from multid4rt.vendor.heads.dpt_head import DPTHead
from multid4rt.vendor.heads.motiondecoder import MotionDecoder


@pytest.fixture(scope="module", autouse=True)
def _limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _decoder():
    torch.manual_seed(19)
    decoder = MultiViewMotionDecoder(
        embed_dim=64, depth=4, num_heads=4,
        attention_chunk_size=3, checkpoint=True,
    )
    # Exercise every self-attention/MLP branch instead of only its initially
    # zero AdaLN gates, and make camera relabel tests cover nonzero bias.
    with torch.no_grad():
        decoder.camera_bias.uniform_(-0.4, 0.4)
        for block in decoder.self_blocks:
            block.adaLN_modulation[-1].weight.normal_(std=0.015)
            block.adaLN_modulation[-1].bias.normal_(std=0.015)
    return decoder


def _head():
    return MotionDPT(embed_dim=64, features=32, out_channels=(32, 64, 64, 64))


def _inputs(requires_grad=False):
    generator = torch.Generator().manual_seed(7)
    source = torch.randn(2, 6, 64, generator=generator, requires_grad=requires_grad)
    memory = torch.randn(2, 3, 12, 64, generator=generator, requires_grad=requires_grad)
    cond = torch.randn(2, 3, 64, generator=generator, requires_grad=requires_grad)
    source_camera = torch.tensor([0, 1])
    memory_camera = torch.tensor([0] * 6 + [1] * 6).view(1, 1, 12).expand(2, 3, -1)
    valid = torch.ones(2, 3, 12, dtype=torch.bool)
    valid[1, 1, 6:] = False
    return source, memory, cond, source_camera, memory_camera, valid, (2, 3)


def test_complete_shared_decoder_and_dpt_forward_backward():
    decoder, head = _decoder(), _head()
    source, memory, cond, source_camera, memory_camera, valid, grid = _inputs(True)
    # Four backbone feature taps share one decoder, as in the complete model.
    taps = [decoder(
        source * (1.0 + tap / 10.0), memory, cond,
        source_camera, memory_camera, valid, grid,
    ) for tap in range(4)]
    assert all(tap.shape == (2, 3, 7, 64) for tap in taps)
    prediction = head(taps, output_hw=(28, 42))
    assert prediction["displacement"].shape == (2, 3, 28, 42, 3)
    assert prediction["confidence"].shape == (2, 3, 28, 42)
    assert prediction["features"].shape == (2, 3, 16, 28, 42)
    assert all(torch.isfinite(value).all() for value in prediction.values())
    assert (prediction["confidence"] >= 1.0).all()

    loss = prediction["displacement"].square().mean() + prediction["confidence"].mean()
    loss.backward()
    for value in (source, memory, cond):
        assert value.grad is not None and torch.isfinite(value.grad).all()
    assert decoder.camera_bias.grad is not None
    assert torch.isfinite(decoder.camera_bias.grad).all()
    assert decoder.camera_bias.grad.abs().sum() > 0
    for module in (decoder, head):
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_four_round_checkpoint_matches_every_parameter_and_input_gradient():
    checkpointed = _decoder().train()
    plain = copy.deepcopy(checkpointed).train()
    plain.use_checkpointing = False
    original = _inputs()
    first = tuple(value.clone().requires_grad_() for value in original[:3])
    second = tuple(value.clone().requires_grad_() for value in original[:3])
    first_output = checkpointed(*first, *original[3:])
    second_output = plain(*second, *original[3:])
    torch.testing.assert_close(first_output, second_output)
    first_output.square().mean().backward()
    second_output.square().mean().backward()

    # Comparing every round catches closures that accidentally recompute the
    # last block instead of the block used during the forward pass.
    for (name, parameter), (reference_name, reference) in zip(
        checkpointed.named_parameters(), plain.named_parameters()
    ):
        assert name == reference_name
        assert parameter.grad is not None and reference.grad is not None, name
        torch.testing.assert_close(parameter.grad, reference.grad, atol=3e-7, rtol=3e-5, msg=name)
    for value, reference in zip(first, second):
        torch.testing.assert_close(value.grad, reference.grad, atol=3e-7, rtol=3e-5)


def test_camera_relabel_and_camera_major_kv_reorder_preserve_trajectories():
    decoder = _decoder().eval()
    source, memory, cond, source_camera, memory_camera, valid, grid = _inputs()
    with torch.no_grad():
        expected = decoder(source, memory, cond, source_camera, memory_camera, valid, grid)
        renamed_source = torch.where(source_camera == 0, 17, 5)
        renamed_memory = torch.where(memory_camera == 0, 17, 5)
        renamed = decoder(source, memory, cond, renamed_source, renamed_memory, valid, grid)
        torch.testing.assert_close(renamed, expected, atol=1e-6, rtol=1e-5)

        order = torch.cat((torch.arange(6, 12), torch.arange(6)))
        swapped = decoder(
            source, memory[:, :, order], cond, source_camera,
            memory_camera[:, :, order], valid[:, :, order], grid,
        )
        torch.testing.assert_close(swapped, expected, atol=3e-6, rtol=3e-5)


def test_masked_camera_memory_has_no_influence_or_gradient():
    decoder = _decoder().eval()
    source, memory, cond, source_camera, memory_camera, valid, grid = _inputs(True)
    expected = decoder(source, memory, cond, source_camera, memory_camera, valid, grid)
    changed = memory.detach().clone()
    changed[~valid] = 1000.0
    with torch.no_grad():
        actual = decoder(source, changed, cond, source_camera, memory_camera, valid, grid)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    expected.square().mean().backward()
    torch.testing.assert_close(memory.grad[~valid], torch.zeros_like(memory.grad[~valid]), atol=0, rtol=0)


def test_motion_checkpoint_parameter_names_and_shapes_match_4rc():
    decoder = _decoder()
    original = MotionDecoder(embed_dim=64, depth=4, num_heads=4, use_adaln=True)
    state, reference = decoder.state_dict(), original.state_dict()
    assert set(state) == set(reference) | {"camera_bias"}
    assert state["camera_bias"].shape == (4, 4)
    for key, value in reference.items():
        assert state[key].shape == value.shape, key


def test_motion_dpt_matches_original_dpt_for_identical_weights():
    torch.manual_seed(31)
    head = _head().eval()
    original = DPTHead(
        dim_in=64, features=32, out_channels=[32, 64, 64, 64],
        intermediate_layer_idx=[0, 1, 2, 3],
    ).eval()
    original.load_state_dict(head.state_dict(), strict=True)
    taps = [torch.randn(2, 3, 7, 64) for _ in range(4)]
    with torch.no_grad():
        actual = head(taps, output_hw=(28, 42))
        displacement, confidence = original(
            taps, images=torch.empty(2, 3, 3, 28, 42),
            patch_start_idx=1, frames_chunk_size=None,
        )
    torch.testing.assert_close(actual["displacement"], displacement, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(actual["confidence"], confidence)


def test_signed_expm1_activation_has_unit_derivative_at_zero():
    head = _head()
    taps = [torch.zeros(1, 2, 7, 64) for _ in range(4)]
    last_layer = head.scratch.output_conv2[-1]
    with torch.no_grad():
        last_layer.weight.zero_()
        last_layer.bias.zero_()
    prediction = head(taps, output_hw=(28, 42))
    torch.testing.assert_close(prediction["displacement"], torch.zeros_like(prediction["displacement"]), atol=0, rtol=0)
    prediction["displacement"].sum().backward()
    expected = torch.tensor([2 * 28 * 42] * 3 + [0.0])
    torch.testing.assert_close(last_layer.bias.grad, expected)


@pytest.mark.parametrize("logit", [-1000.0, 1000.0])
def test_motion_dpt_extreme_logits_remain_finite(logit):
    head = _head()
    taps = [torch.zeros(1, 2, 7, 64) for _ in range(4)]
    last_layer = head.scratch.output_conv2[-1]
    with torch.no_grad():
        last_layer.weight.zero_()
        last_layer.bias.fill_(logit)
    prediction = head(taps, output_hw=(28, 42))
    assert all(torch.isfinite(value).all() for value in prediction.values())
    assert (prediction["confidence"] >= 1.0).all()
    assert (prediction["displacement"] * logit > 0).all()
