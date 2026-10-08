"""Real two-rank Gloo training plus AMP overflow and visible-device contracts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import h5py
import pytest
import torch
from torch import distributed as dist

from multicamera4rc.datasets.demo import make_demo_data
from multicamera4rc.distributed import DistributedContext, resolve_amp_dtype, resolve_device
from multicamera4rc import train


_WORKER = r'''
import functools
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from torch import distributed as dist
from multicamera4rc import train
from multicamera4rc.datasets import mixture

records = {"writes": [], "sampler": [], "layouts": []}
original_forward = train._TrainingStep.forward
def recording_forward(self, images, camera_id, time_idx, query_time_idx, *args, **kwargs):
    records["layouts"].append([int(camera_id[0].unique().numel()), int(query_time_idx.shape[1])])
    return original_forward(self, images, camera_id, time_idx, query_time_idx, *args, **kwargs)
train._TrainingStep.forward = recording_forward
original_ddp = train.DistributedDataParallel
@functools.wraps(original_ddp)
def recording_ddp(*args, **kwargs):
    wrapped = original_ddp(*args, **kwargs)
    records["ddp"] = wrapped
    return wrapped
train.DistributedDataParallel = recording_ddp

original_sampler = train.DistributedSampler
class RecordingSampler(original_sampler):
    def __iter__(self):
        indices = list(super().__iter__())
        records["sampler"].append({"epoch": self.epoch, "indices": indices})
        return iter(indices)
train.DistributedSampler = RecordingSampler

original_dataset = mixture.create_dataset
def recording_dataset(*args, **kwargs):
    dataset = original_dataset(*args, **kwargs)
    records["dataset"] = dataset
    return dataset
mixture.create_dataset = recording_dataset

original_save = train.save_checkpoint
def recording_save(path, *args, **kwargs):
    records["writes"].append(Path(path).name)
    return original_save(path, *args, **kwargs)
train.save_checkpoint = recording_save

checkpoint = train.main(sys.argv[1:])
model = records.pop("ddp").module.model
digest = hashlib.sha256()
for name, value in sorted(model.state_dict().items()):
    digest.update(name.encode())
    digest.update(value.detach().contiguous().cpu().numpy().tobytes())
records["digest"] = digest.hexdigest()
records["all_finite"] = all(torch.isfinite(value).all().item() for value in model.state_dict().values())
records["dataset_epoch"] = records.pop("dataset").epoch
records["group_initialized_after_main"] = dist.is_initialized()
records["rank"] = int(os.environ["RANK"])
Path(checkpoint).parent.joinpath(f"rank_{records['rank']}.json").write_text(json.dumps(records))
'''


def _torchrun(tmp_path, config, output, steps, resume=None, no_query_chunking=False, reset_data_cursor=False,
              world_size=2):
    worker = tmp_path / "ddp_worker.py"
    worker.write_text(textwrap.dedent(_WORKER))
    command = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
               "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:0", "--local-addr=127.0.0.1",
               f"--nproc_per_node={world_size}", str(worker), "--data-config", str(config),
               "--output", str(output), "--preset", "tiny", "--device", "cpu",
               "--steps", str(steps), "--batch-size", "1", "--source-count", "1",
               "--cpu-threads", "1", "--save-every", "1", "--amp",
               "--amp-dtype", "bfloat16"]
    if resume:
        command.extend(("--resume", str(resume)))
    if no_query_chunking:
        command.append("--no-query-chunking")
    if reset_data_cursor:
        command.append("--reset-data-cursor")
    environment = os.environ.copy()
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(key, None)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "1"
    root = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run(command, cwd=root, env=environment, capture_output=True,
                            text=True, timeout=45)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    logs = []
    for line in result.stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "step" in record:
            logs.append(record)
    records = [json.loads((output / f"rank_{rank}.json").read_text()) for rank in range(world_size)]
    return records, logs


def test_rank_resize_cursor_preserves_samples_independently_of_total_steps():
    state = dict(world_size=2, batch_size=1, data_epoch=3, batches_consumed=500)
    assert train._resume_data_cursor(state, 55500, 8000, 4, 1) == (3, 250)
    assert train._resume_data_cursor(state, 55500, 8000, 2, 1) == (3, 500)
    state['batches_consumed'] = 499
    with pytest.raises(ValueError, match='aligned'):
        train._resume_data_cursor(state, 55500, 8000, 4, 1)
    assert train._resume_data_cursor(state, 55500, 8000, 4, 1, reset=True) == (0, 0)
    state['batches_consumed'] = 4000
    assert train._resume_data_cursor(state, 55500, 8000, 4, 1) == (4, 0)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_two_to_four_ranks_preserves_sample_prefix_and_resumes_four_rank_state(tmp_path):
    config = make_demo_data(tmp_path / 'data')
    data = json.loads(config.read_text()); data['epoch_size'] = 16
    config.write_text(json.dumps(data))
    old, _ = _torchrun(tmp_path, config, tmp_path/'two', 2)
    path = tmp_path/'two/latest.pt'
    before = torch.load(path, weights_only=True)
    # Total optimizer steps need not track the current mixture's data cursor.
    before['step'] = before['training_state']['optimizer_updates'] = 55000
    for state in before['optimizer']['state'].values():
        state['step'].fill_(55000)
    torch.save(before, path)
    resized, logs = _torchrun(tmp_path, config, tmp_path/'four', 55001, path, world_size=4)
    assert [log['step'] for log in logs] == [55001]
    assert logs[0]['world_size'] == 4 and len(logs[0]['rank_batches']) == 4
    assert not logs[0]['optimizer_step_skipped']
    assert len({r['digest'] for r in resized}) == 1
    assert all(r['all_finite'] for r in resized)
    assert not any(r['writes'] for r in resized[1:])
    after = torch.load(tmp_path/'four/latest.pt', weights_only=True)
    state = after['training_state']
    assert (state['data_epoch'], state['batches_consumed'], state['world_size']) == (0, 2, 4)
    assert len(after['rank_rng']) == 4
    assert state['lr_plan'] == before['training_state']['lr_plan']
    assert state['loss_config'] == before['training_state']['loss_config']
    assert any(s['step'].item() == 55001 for s in after['optimizer']['state'].values())
    old_prefix = {r['sampler'][0]['indices'][i] for r in old for i in range(2)}
    new_prefix = {r['sampler'][0]['indices'][0] for r in resized}
    next_samples = {r['sampler'][0]['indices'][1] for r in resized}
    assert old_prefix == new_prefix and next_samples.isdisjoint(old_prefix)
    # Saving and resuming again with four ranks preserves the exact trajectory.
    continued, _ = _torchrun(tmp_path, config, tmp_path/'continued', 55002,
                             tmp_path/'four/latest.pt', world_size=4)
    direct, _ = _torchrun(tmp_path, config, tmp_path/'direct', 55002, path, world_size=4)
    assert {r['digest'] for r in continued} == {r['digest'] for r in direct}


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_torchrun_two_rank_native_bf16_training_saves_once_and_resumes(tmp_path):
    config = make_demo_data(tmp_path / "data")
    # Rank-local GT differs: MUGEN trains geometry only, Kubric also trains
    # motion/visibility. Both ranks have no ray GT. DDP must reduce correctly
    # despite changing unused task heads, with activation checkpointing enabled.
    with h5py.File(config.parent / "dynamic.h5", "r+") as handle:
        handle["train/scene/visibility_valids"][:] = False
    records, logs = _torchrun(tmp_path, config, tmp_path / "first", steps=2)
    assert [log["step"] for log in logs] == [1, 2]  # Rank 0 emits one global log.
    assert all(log["world_size"] == 2 and log["amp_dtype"] == "bfloat16" for log in logs)
    assert all(not log["optimizer_step_skipped"] for log in logs)
    assert records[0]["digest"] == records[1]["digest"]
    assert all(record["all_finite"] for record in records)
    assert records[0]["writes"] == ["step_00000001.pt", "step_00000002.pt", "latest.pt"]
    assert records[1]["writes"] == []
    assert all(not record["group_initialized_after_main"] for record in records)
    for rank in range(2):
        assert [sample["epoch"] for sample in records[rank]["sampler"]] == [0, 1]
        assert records[rank]["dataset_epoch"] == 1
    for epoch in range(2):
        left = set(records[0]["sampler"][epoch]["indices"])
        right = set(records[1]["sampler"][epoch]["indices"])
        assert left.isdisjoint(right) and left | right == {0, 1}
    first_path = tmp_path / "first" / "latest.pt"
    first = torch.load(first_path, weights_only=True)
    assert first["step"] == first["training_state"]["optimizer_updates"] == 2
    assert first["training_state"]["data_epoch"] == 2
    assert first["training_state"]["batches_consumed"] == 0
    assert len(first["rank_rng"]) == 2
    assert first["optimizer"]["state"]
    assert not any(key.startswith("module.") or key.startswith("model.") for key in first["model"])
    resumed_records, resumed_logs = _torchrun(tmp_path, config, tmp_path / "resumed", 3, first_path)
    assert [log["step"] for log in resumed_logs] == [3]
    assert resumed_logs[0]["data_epoch"] == 2
    assert resumed_records[0]["digest"] == resumed_records[1]["digest"]
    assert resumed_records[0]["digest"] != records[0]["digest"]
    assert resumed_records[0]["writes"] == ["step_00000003.pt", "latest.pt"]
    assert resumed_records[1]["writes"] == []
    assert all(not record["group_initialized_after_main"] for record in resumed_records)
    resumed = torch.load(tmp_path / "resumed" / "latest.pt", weights_only=True)
    assert resumed["step"] == resumed["training_state"]["optimizer_updates"] == 3
    assert resumed["training_state"]["data_epoch"] == 3
    uninterrupted_records, uninterrupted_logs = _torchrun(tmp_path, config, tmp_path / "uninterrupted", 3)
    assert [log["step"] for log in uninterrupted_logs] == [1, 2, 3]
    # Per-rank RNG, data cursor, optimizer moments and BF16 execution resume
    # to the same weights as an uninterrupted three-step run.
    assert uninterrupted_records[0]["digest"] == resumed_records[0]["digest"]


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_two_ranks_can_train_different_time_counts_with_whole_query_attention(tmp_path):
    config = make_demo_data(tmp_path / "data")
    data = json.loads(config.read_text())
    data["epoch_size"] = 8
    data["sources"][0].update(num_times=5, strides=[1, 2])
    data["sources"][1].update(num_times=3, strides=[1, 3])
    config.write_text(json.dumps(data))
    records, logs = _torchrun(tmp_path, config, tmp_path / "variable", 1, no_query_chunking=True)
    assert {tuple(row) for record in records for row in record["layouts"]} == {(1, 3), (2, 5)}
    assert records[0]["digest"] == records[1]["digest"]
    assert all(record["all_finite"] for record in records)
    assert len(logs) == 1 and not logs[0]["optimizer_step_skipped"]
    checkpoint = torch.load(tmp_path / "variable/latest.pt", weights_only=True)
    assert checkpoint["config"]["attention_chunk_size"] is None
    assert checkpoint["config"]["visibility_chunk_size"] is None
    assert checkpoint["training_state"]["batches_consumed"] == 1
    resumed, resumed_logs = _torchrun(tmp_path, config, tmp_path / "reset", 2,
                                     tmp_path / "variable/latest.pt",
                                     no_query_chunking=True, reset_data_cursor=True)
    assert resumed[0]["digest"] == resumed[1]["digest"]
    assert [record["layouts"] for record in resumed] == [record["layouts"] for record in records]
    assert resumed_logs[0]["step"] == 2
    state = torch.load(tmp_path / "reset/latest.pt", weights_only=True)["training_state"]
    assert state["data_epoch"] == 0 and state["batches_consumed"] == 1


def test_cuda_visible_devices_are_addressed_by_local_rank(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert resolve_device("cuda", local_rank=0, distributed=True) == torch.device("cuda:0")
    assert resolve_device("cuda", local_rank=1, distributed=True) == torch.device("cuda:1")
    assert resolve_device("cuda:4", local_rank=1, distributed=True) == torch.device("cuda:1")
    with pytest.raises(ValueError, match="visible GPU"):
        resolve_device("cuda:4")


@pytest.mark.parametrize("supported", [True, False])
def test_auto_cuda_amp_prefers_bf16_and_falls_back_to_fp16(monkeypatch, supported):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: supported)
    expected = torch.bfloat16 if supported else torch.float16
    assert resolve_amp_dtype(torch.device("cuda:0"), "auto") == expected
    assert resolve_amp_dtype(torch.device("cuda:0"), "float16") == torch.float16
    if supported:
        assert resolve_amp_dtype(torch.device("cuda:0"), "bfloat16") == torch.bfloat16
    else:
        with pytest.raises(ValueError, match="does not support bfloat16"):
            resolve_amp_dtype(torch.device("cuda:0"), "bfloat16")


def test_cpu_amp_uses_bf16_and_rejects_unscaled_fp16_training():
    assert resolve_amp_dtype(torch.device("cpu"), "auto") == torch.bfloat16
    assert resolve_amp_dtype(torch.device("cpu"), "bfloat16") == torch.bfloat16
    with pytest.raises(ValueError, match="float16 requires CUDA"):
        resolve_amp_dtype(torch.device("cpu"), "float16")


def test_scaled_overflow_skips_optimizer_and_reduces_scale_without_clip_error():
    # CPU GradScaler exercises the same standard overflow protocol as CUDA
    # FP16. An explicit overflowing gradient avoids needing GPU hardware.
    model = torch.nn.Linear(1, 1, bias=False)
    initial = model.weight.detach().clone()
    model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=16.)
    norm, skipped = train._optimizer_step(model(torch.ones(1, 1)).sum(), model, optimizer, scaler, 1.)
    assert not torch.isfinite(norm)
    assert skipped and scaler.get_scale() == 8.
    assert not optimizer.state
    torch.testing.assert_close(model.weight.detach(), initial)


def test_unscaled_nonfinite_gradients_fail_before_optimizer_update():
    model = torch.nn.Linear(1, 1, bias=False)
    initial = model.weight.detach().clone()
    model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    with pytest.raises(RuntimeError, match="non-finite"):
        train._optimizer_step(model(torch.ones(1, 1)).sum(), model, optimizer, scaler, 1.)
    torch.testing.assert_close(model.weight.detach(), initial)


def test_training_exception_always_closes_owned_context(tmp_path, monkeypatch):
    closed = []
    class Context:
        def close(self):
            closed.append(True)
    monkeypatch.setattr(DistributedContext, "initialize", classmethod(lambda cls, requested: Context()))
    def fail(*args):
        raise RuntimeError("intentional training failure")
    monkeypatch.setattr(train, "_train", fail)
    with pytest.raises(RuntimeError, match="intentional training failure"):
        train.main(["--manifest", "unused.json", "--output", str(tmp_path), "--device", "cpu"])
    assert closed == [True]


def test_incomplete_torchrun_environment_fails_before_starting_group(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    for key in ("RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="launch distributed training with torchrun"):
        DistributedContext.initialize("cpu")
