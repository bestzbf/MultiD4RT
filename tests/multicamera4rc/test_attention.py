"""Numerical guarantees for chunked camera-relation attention."""

import pytest
import torch
import torch.nn.functional as F

from multicamera4rc.attention import relation_attention, _feature_relation_attention


@pytest.fixture(scope="module", autouse=True)
def _limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _inputs():
    generator = torch.Generator().manual_seed(23)
    q = torch.randn(2, 3, 7, 8, dtype=torch.float64, generator=generator)
    k = torch.randn(2, 3, 9, 8, dtype=torch.float64, generator=generator)
    v = torch.randn(2, 3, 9, 8, dtype=torch.float64, generator=generator)
    weight = torch.tensor([0.2, -0.7, 1.1], dtype=torch.float64)
    query_camera = torch.tensor([[-1, 0, 0, 1, 1, 2, 2], [-1, 2, 1, 0, 2, 1, 0]])
    key_camera = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, -1], [0, 0, 1, 1, 1, 2, 2, 2, -1]])
    key_valid = torch.tensor([[True] * 8 + [False], [True, False, True, True, True, True, True, False, False]])
    return (q, k, v, weight), query_camera, key_camera, key_valid


@pytest.mark.parametrize("checkpoint_chunks", [False, True])
@pytest.mark.parametrize("query_chunk_size", [3, None])
def test_chunked_forward_and_gradients_match_full_sdpa(checkpoint_chunks, query_chunk_size):
    values, query_camera, key_camera, key_valid = _inputs()
    actual_values = [value.clone().requires_grad_() for value in values]
    expected_values = [value.clone().requires_grad_() for value in values]
    q, k, v, weight = actual_values
    actual = relation_attention(
        q, k, v, query_camera, key_camera, weight, key_valid,
        query_chunk_size=query_chunk_size, checkpoint_chunks=checkpoint_chunks,
    )

    q_ref, k_ref, v_ref, weight_ref = expected_values
    same_camera = (
        (query_camera[:, :, None] == key_camera[:, None, :])
        & (query_camera[:, :, None] >= 0)
        & (key_camera[:, None, :] >= 0)
    )
    bias = same_camera[:, None].to(q_ref.dtype) * weight_ref[None, :, None, None]
    bias = bias.masked_fill(~key_valid[:, None, None, :], float("-inf"))
    expected = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask=bias)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)

    probe = torch.linspace(-1.0, 1.0, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    for actual_value, expected_value in zip(actual_values, expected_values):
        assert actual_value.grad is not None
        torch.testing.assert_close(actual_value.grad, expected_value.grad, atol=1e-11, rtol=1e-10)


def test_whole_query_attention_uses_one_sdpa_call(monkeypatch):
    values, query_camera, key_camera, key_valid = _inputs()
    original = F.scaled_dot_product_attention
    calls = []
    def recording_sdpa(q, *args, **kwargs):
        calls.append(q.shape[-2])
        return original(q, *args, **kwargs)
    monkeypatch.setattr(F, "scaled_dot_product_attention", recording_sdpa)
    relation_attention(*values[:3], query_camera, key_camera, values[3], key_valid,
                       query_chunk_size=None, checkpoint_chunks=True)
    assert calls == [7]


def test_camera_feature_factorization_matches_dense_relation_and_all_gradients():
    values, query_camera, key_camera, key_valid = _inputs()
    actual_values = [x.clone().requires_grad_() for x in values]
    expected_values = [x.clone().requires_grad_() for x in values]
    actual = _feature_relation_attention(*actual_values, query_camera, key_camera, key_valid, 0.)
    q, k, v, weight = expected_values
    same = ((query_camera[:, :, None] == key_camera[:, None, :])
            & (query_camera[:, :, None] >= 0) & (key_camera[:, None, :] >= 0))
    bias = (same[:, None]*weight[None, :, None, None]).masked_fill(~key_valid[:, None, None, :],float('-inf'))
    expected = F.scaled_dot_product_attention(q,k,v,attn_mask=bias)
    torch.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-12)
    probe=torch.linspace(-1.,1.,actual.numel(),dtype=actual.dtype).reshape_as(actual)
    (actual*probe).sum().backward();(expected*probe).sum().backward()
    for x,y in zip(actual_values,expected_values):
        torch.testing.assert_close(x.grad,y.grad,rtol=1e-10,atol=1e-11)


def test_camera_feature_factorization_zero_weight_and_arbitrary_id_keep_gradient():
    q=torch.zeros(1,1,1,1);k=torch.zeros(1,1,2,1);v=torch.tensor([[[[2.],[4.]]]])
    weight=torch.zeros(1,requires_grad=True)
    result=_feature_relation_attention(q,k,v,weight,torch.tensor([[100000]]),
        torch.tensor([[100000,29]]),None,0.)
    torch.testing.assert_close(result,torch.full_like(result,3.))
    result.sum().backward();torch.testing.assert_close(weight.grad,torch.tensor([-.5]))


def test_prefix_camera_has_zero_relation_bias_and_gradient():
    values, query_camera, key_camera, key_valid = _inputs()
    q, k, v, _ = values
    weight = torch.tensor([2.0, -3.0, 4.0], dtype=q.dtype, requires_grad=True)
    actual = relation_attention(
        q, k, v, query_camera, key_camera, weight, key_valid,
        query_chunk_size=2, checkpoint_chunks=True,
    )
    expected_prefix = F.scaled_dot_product_attention(
        q[:, :, :1], k, v, attn_mask=key_valid[:, None, None, :],
    )
    torch.testing.assert_close(actual[:, :, :1], expected_prefix, atol=1e-12, rtol=1e-11)
    actual[:, :, :1].square().sum().backward()
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight), atol=0, rtol=0)


@pytest.mark.parametrize("checkpoint_chunks", [False, True])
def test_zero_initialized_relation_weight_keeps_nonzero_gradient(checkpoint_chunks):
    # logits [alpha,0] and values [2,4] give output 3 and d/dalpha=-0.5
    # at alpha=0. A zero-weight shortcut would silently remove this gradient.
    q = torch.zeros(1, 1, 1, 1)
    k = torch.zeros(1, 1, 2, 1)
    v = torch.tensor([[[[2.0], [4.0]]]])
    weight = torch.zeros(1, requires_grad=True)
    actual = relation_attention(
        q, k, v, torch.tensor([[0]]), torch.tensor([[0, 1]]), weight,
        query_chunk_size=1, checkpoint_chunks=checkpoint_chunks,
    )
    torch.testing.assert_close(actual, torch.full_like(actual, 3.0))
    actual.sum().backward()
    torch.testing.assert_close(weight.grad, torch.tensor([-0.5]))


@pytest.mark.parametrize("with_relation", [False, True])
def test_all_invalid_keys_return_finite_zero_and_zero_gradients(with_relation):
    values, query_camera, key_camera, key_valid = _inputs()
    q, k, v, weight = [value.clone().requires_grad_() for value in values]
    actual = relation_attention(
        q, k, v, query_camera, key_camera,
        weight if with_relation else None, torch.zeros_like(key_valid),
        query_chunk_size=2, checkpoint_chunks=True,
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, torch.zeros_like(actual), atol=0, rtol=0)
    actual.sum().backward()
    for value in (q, k, v) + ((weight,) if with_relation else ()):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        torch.testing.assert_close(value.grad, torch.zeros_like(value), atol=0, rtol=0)


@pytest.mark.parametrize("with_relation", [False, True])
def test_masked_key_contents_do_not_change_output_or_receive_gradients(with_relation):
    values, query_camera, key_camera, key_valid = _inputs()
    q, k, v, weight = [value.clone().requires_grad_() for value in values]
    kwargs = dict(
        query_camera=query_camera, key_camera=key_camera,
        relation_weight=weight if with_relation else None, key_valid=key_valid,
        query_chunk_size=3, checkpoint_chunks=True,
    )
    expected = relation_attention(q, k, v, **kwargs)
    masked = (~key_valid)[:, None, :, None].expand_as(k)
    changed_k = k.detach().clone().masked_fill(masked, 1000.0)
    changed_v = v.detach().clone().masked_fill(masked, -1000.0)
    actual = relation_attention(q, changed_k, changed_v, **kwargs)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
    expected.square().sum().backward()
    torch.testing.assert_close(k.grad[masked], torch.zeros_like(k.grad[masked]), atol=0, rtol=0)
    torch.testing.assert_close(v.grad[masked], torch.zeros_like(v.grad[masked]), atol=0, rtol=0)
