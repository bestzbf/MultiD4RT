"""Loss semantics, checkpointed schedules and reproducible dense evaluation."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler

from multicamera4rc.evaluate import fixed_sources, make_validation_plan
from multicamera4rc.geometry import decode_camera
from multicamera4rc.losses import MultiCamera4RCLoss
from multicamera4rc.schedule import LearningRatePlan
from multicamera4rc.train import _CursorBatchSampler


def _only(task):
    config = MultiCamera4RCLoss.balanced_dense().to_config()
    config["weights"] = {name: float(name == task) for name in config["weights"]}
    return MultiCamera4RCLoss(**config)


@pytest.mark.parametrize("foreground_count", [2, 20, 80])
def test_dense_group_balance_is_independent_of_object_area_and_keeps_background_gradients(foreground_count):
    truth = torch.zeros(1, 1, 2, 1, 100, 3)
    estimate = truth.clone()
    estimate[..., 0] = 4
    estimate[..., :foreground_count, 0] = 2
    estimate.requires_grad_()
    valid = torch.ones(truth.shape[:-1], dtype=torch.bool)
    foreground = valid.clone()
    foreground[..., foreground_count:] = False
    result = _only("displacement")({"displacement": estimate},
        {"displacement": truth, "displacement_valid": valid, "trajectory_foreground": foreground})
    torch.testing.assert_close(result["loss"], torch.tensor(3.))
    result["loss"].backward()
    assert (estimate.grad[..., 0] > 0).all()
    torch.testing.assert_close(estimate.grad[..., :foreground_count, 0].sum(), torch.tensor(.5))
    torch.testing.assert_close(estimate.grad[..., foreground_count:, 0].sum(), torch.tensor(.5))


def test_dense_balance_handles_static_only_and_invalid_pixels():
    truth = torch.zeros(1, 1, 1, 2, 2, 3)
    truth[..., 1, 1, :] = float("nan")
    valid = torch.ones(truth.shape[:-1], dtype=torch.bool)
    valid[..., 1, 1] = False
    estimate = torch.ones_like(truth).requires_grad_()
    result = _only("displacement")({"displacement": estimate},
        {"displacement": truth, "displacement_valid": valid, "trajectory_foreground": torch.zeros_like(valid)})
    torch.testing.assert_close(result["loss"], torch.tensor(math.sqrt(3)))
    result["loss"].backward()
    assert torch.isfinite(estimate.grad).all()
    assert not estimate.grad[..., 1, 1, :].any()


def test_geometry_error_does_not_train_motion_confidence():
    truth = torch.zeros(1, 1, 2, 2, 2, 3)
    estimate = torch.ones_like(truth).requires_grad_()
    confidence = torch.full(truth.shape[:-1], 2., requires_grad=True)
    result = _only("trajectory")({"trajectory": estimate, "motion_confidence": confidence,
                                 "source_time_idx": torch.tensor([[0]]), "query_time_idx": torch.tensor([[0, 1]])},
        {"trajectory": truth, "trajectory_valid": torch.ones(truth.shape[:-1], dtype=torch.bool)})
    result["loss"].backward()
    assert confidence.grad is None
    assert estimate.grad.abs().sum() > 0
    assert result["confidence"] == 0


def test_confidence_adds_plain_residual_and_excludes_source_identity_regularizer():
    pred = torch.tensor([[1., 0., 0.], [1., 0., 0.]], requires_grad=True)
    confidence = torch.tensor([2., 2.], requires_grad=True)
    fit, regularizer = _only("displacement")._regression(pred, torch.zeros_like(pred),
        torch.ones(2, dtype=torch.bool), confidence, torch.tensor([False, True]), task="displacement")
    torch.testing.assert_close(fit, torch.tensor(2.5))
    torch.testing.assert_close(regularizer, torch.tensor(-.1*math.log(2)))
    (fit+regularizer).backward()
    torch.testing.assert_close(confidence.grad, torch.tensor([0., .45]))


def test_camera_quaternion_sign_and_native_principal_offsets_are_supervised():
    height, width = 14, 28
    pose = torch.tensor([[[0., 0., 0., 0., 0., 0., -1.,
                          2*math.atan(7/30), 2*math.atan(14/40), -6/28, -2/14]]], requires_grad=True)
    K, camera = decode_camera(pose, height, width)
    truth_K = torch.tensor([[[[40., 0., 8.], [0., 30., 5.], [0., 0., 1.]]]])
    target = {"K": truth_K, "c2w": torch.eye(4)[None, None], "camera_valid": torch.ones(1, 1, dtype=torch.bool),
              "depth": torch.ones(1, 1, height, width)}
    pred = {"K": K, "c2w": camera, "camera_encoding": pose, "depth": target["depth"]}
    result = _only("camera")(pred, target)
    torch.testing.assert_close(result["loss"], torch.tensor(0.), atol=1e-6, rtol=0)
    modified = pose.detach().clone()
    modified[..., 9] += .1
    modified.requires_grad_()
    K, camera = decode_camera(modified, height, width)
    result = _only("camera")({**pred, "K": K, "c2w": camera, "camera_encoding": modified}, target)
    torch.testing.assert_close(result["camera_principal"], torch.tensor(.05), atol=1e-6, rtol=0)
    torch.testing.assert_close(result["loss"], torch.tensor(.025), atol=1e-6, rtol=0)
    result["loss"].backward()
    assert torch.isfinite(modified.grad).all()
    assert modified.grad[..., 9].abs().sum() > 0


def test_multiscale_depth_gradient_uses_gt_edges_without_confidence_regularizers():
    truth = torch.ones(1, 1, 16, 16)
    estimate = (truth+.1*torch.arange(16)[None, None, None]).requires_grad_()
    confidence = torch.full_like(truth, 100., requires_grad=True)
    result = _only("depth_gradient")({"depth": estimate, "depth_conf": confidence},
                                    {"depth": truth, "depth_valid": torch.ones_like(truth, dtype=torch.bool)})
    torch.testing.assert_close(result["depth_gradient"], torch.tensor(.1875))
    result["loss"].backward()
    assert confidence.grad is None
    assert torch.isfinite(estimate.grad).all()


def test_ray_norm_is_independent_of_motion_vector_norm():
    truth = torch.zeros(1, 1, 2, 2, 6)
    result = _only("ray")({"ray": torch.ones_like(truth)}, {"ray": truth})
    torch.testing.assert_close(result["loss"], torch.tensor(6.))


def test_loss_configuration_roundtrip_preserves_objective():
    criterion = MultiCamera4RCLoss.balanced_dense()
    assert MultiCamera4RCLoss(**criterion.to_config()).to_config() == criterion.to_config()


def test_resuming_skips_dataset_reads_for_consumed_batches():
    class RecordedDataset(Dataset):
        reads = []
        def __len__(self): return 10
        def __getitem__(self, index):
            self.reads.append(index)
            return index
    dataset = RecordedDataset()
    sampler = _CursorBatchSampler(SequentialSampler(dataset), 2, drop_last=False)
    sampler.start_batch = 3
    assert len(sampler) == 2
    batches = list(DataLoader(dataset, batch_sampler=sampler))
    assert dataset.reads == [6, 7, 8, 9]
    assert [b.tolist() for b in batches] == [[6, 7], [8, 9]]


def test_learning_rate_phase_warms_up_decays_and_resumes_exactly():
    plan = LearningRatePlan("cosine", 2000, 2100, [1e-5, 1e-4], warmup_steps=5, min_ratio=.1)
    assert plan.factor(2000) == .2
    assert plan.factor(2004) == 1
    assert plan.factor(2005) == 1
    assert plan.factor(2099) == pytest.approx(.1)
    assert plan.factor(2200) == pytest.approx(.1)
    restored = LearningRatePlan(**plan.to_config())
    assert [restored.factor(i) for i in range(2000, 2100)] == [plan.factor(i) for i in range(2000, 2100)]


def test_validation_plan_reuses_scenes_across_layouts_and_selects_simultaneous_endpoints():
    pool = [(f"scene_{variant}_{n:06d}", 24) for variant in "CDEF" for n in range(4)]
    dataset = SimpleNamespace(sources=[dict(pool=pool, num_times=t, cameras=c, strides=[1, 2]) for c, t in ((2, 12), (3, 8), (4, 6))])
    plan = make_validation_plan(dataset)
    assert plan == make_validation_plan(dataset)
    assert len(plan["cases"]) == 24
    sets = [{case["sequence"] for case in plan["cases"] if case["layout"] == layout} for layout in range(3)]
    assert sets[0] == sets[1] == sets[2]
    batch = {"observation_valid": torch.ones(1, 6, dtype=torch.bool),
             "camera_id": torch.tensor([[7, 7, 7, 2, 2, 2]]), "time_idx": torch.tensor([[0, 2, 5, 0, 2, 5]])}
    assert fixed_sources(batch, 4).tolist() == [[3, 0, 5, 2]]
