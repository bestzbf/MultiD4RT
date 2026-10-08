"""Verify resource gating and the exact environment passed to a DDP run."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from multicamera4rc import launch


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    config = tmp_path / "data.json"
    config.write_text(json.dumps({"image_size": [280, 504], "num_times": 3,
                                  "num_queries": 1024, "epoch_size": 10000,
                                  "sources": [{"adapter": "kubric_multicam", "root": str(root),
                                               "cameras": 2, "strides": [1, 2, 5]}]}))
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"not-loaded-by-preflight")
    output = tmp_path / "training"
    devices = {4: launch.GPU(4, "GPU-four", "B300", 262144, 1024),
               5: launch.GPU(5, "GPU-five", "B300", 262144, 1024),
               0: launch.GPU(0, "GPU-zero", "B300", 262144, 4096)}
    monkeypatch.setattr(launch, "gpu_snapshot", lambda: (devices, []))
    args = ["--python", sys.executable, "--data-config", str(config),
            "--pretrained", str(weights), "--output", str(output)]
    return args, output, devices


def test_busy_dry_run_is_reviewable_without_launch_or_files(inputs, monkeypatch, capsys):
    args, output, devices = inputs
    process = launch.GPUProcess("GPU-four", 3940351, "canonical4d/python", "96000")
    monkeypatch.setattr(launch, "gpu_snapshot", lambda: (devices, [process]))
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("Dry run started a subprocess"))
    assert launch.main(args + ["--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["existing_compute_processes"][0]["pid"] == 3940351
    assert report["local_rank_to_physical_gpu"] == {"0": 4, "1": 5}
    assert report["environment"]["CUDA_VISIBLE_DEVICES"] == "GPU-four,GPU-five"
    assert "--nproc_per_node=2" in report["argv"]
    assert "--rdzv-endpoint=127.0.0.1:0" in report["argv"]
    assert "--local-addr=127.0.0.1" in report["argv"]
    assert "--standalone" not in report["argv"]
    assert report["global_batch_size"] == 2
    assert report["argv"][report["argv"].index("--amp-dtype") + 1] == "bfloat16"
    assert not output.exists()


def test_busy_launch_refuses_without_starting_child(inputs, monkeypatch, capsys):
    args, output, devices = inputs
    monkeypatch.setattr(launch, "gpu_snapshot", lambda: (devices, [launch.GPUProcess("GPU-five", 51, "other.py", "80000")]))
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("Busy GPU gate started training"))
    assert launch.main(args) == 2
    assert "training was not started" in capsys.readouterr().err
    assert not output.exists()


def test_process_on_unselected_gpu_does_not_block_requested_gpus(inputs, monkeypatch, capsys):
    args, output, devices = inputs
    monkeypatch.setattr(launch, "gpu_snapshot", lambda: (devices, [launch.GPUProcess("GPU-zero", 11, "other.py", "4000")]))
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launch.subprocess, "run", execute)
    assert launch.main(args + ["--steps", "7", "--batch-size", "3", "--attention-chunk-size", "1024"]) == 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-four,GPU-five"
    assert "shell" not in kwargs
    assert command[command.index("--steps") + 1] == "7"
    assert command[command.index("--attention-chunk-size") + 1] == "1024"
    report = json.loads((output / "launch.json").read_text())
    assert report["existing_compute_processes"] == []
    assert report["global_batch_size"] == 6
    assert "--module" in command and "multicamera4rc.train" in command


def test_explicit_share_allows_busy_gpus(inputs, monkeypatch, capsys):
    args, output, devices = inputs
    monkeypatch.setattr(launch, "gpu_snapshot", lambda: (devices, [launch.GPUProcess("GPU-four", 41, "other.py", "1000")]))
    calls = []
    monkeypatch.setattr(launch.subprocess, "run", lambda command, **kw: calls.append(command) or subprocess.CompletedProcess(command, 17))
    assert launch.main(args + ["--allow-shared", "--no-query-chunking",
                              "--decoder-memory", "all_observations"]) == 17
    assert len(calls) == 1
    assert "--no-query-chunking" in calls[0]
    assert calls[0][calls[0].index("--decoder-memory")+1] == "all_observations"
    assert json.loads((output / "launch.json").read_text())["existing_compute_processes"][0]["pid"] == 41


def test_curriculum_lr_flags_reach_distributed_training(inputs, monkeypatch, capsys):
    args, _, _ = inputs
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("dry run launched training"))
    assert launch.main(args + ["--dry-run", "--steps", "32000", "--lr-end-step", "50000",
        "--warmup-steps", "500", "--warmup-start-ratio", ".3"]) == 0
    command = json.loads(capsys.readouterr().out)["argv"]
    assert command[command.index("--lr-end-step")+1] == "50000"
    assert command[command.index("--warmup-start-ratio")+1] == "0.3"


def test_explicit_foreground_wait_starts_only_after_selected_gpus_are_free(inputs, monkeypatch, capsys):
    args, output, devices = inputs
    snapshots = iter([(devices, [launch.GPUProcess("GPU-four", 41, "other.py", "1000")]), (devices, [])])
    monkeypatch.setattr(launch, "gpu_snapshot", lambda: next(snapshots))
    monkeypatch.setattr(launch.time, "monotonic", lambda: 0)
    sleeps, launches = [], []
    monkeypatch.setattr(launch.time, "sleep", sleeps.append)
    monkeypatch.setattr(launch.subprocess, "run", lambda command, **kw: launches.append(command) or subprocess.CompletedProcess(command, 0))
    assert launch.main(args + ["--wait-free", "30"]) == 0
    assert sleeps == [5] and len(launches) == 1
    assert json.loads((output / "launch.json").read_text())["existing_compute_processes"] == []


def test_explicit_wait_timeout_does_not_launch(inputs, monkeypatch, capsys):
    args, output, devices = inputs
    monkeypatch.setattr(launch, "gpu_snapshot", lambda: (devices, [launch.GPUProcess("GPU-four", 41, "other.py", "1000")]))
    clock = iter([0, 31])
    monkeypatch.setattr(launch.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(launch.time, "sleep", lambda *_: pytest.fail("Expired wait should not sleep"))
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("Expired resource gate started training"))
    assert launch.main(args + ["--wait-free", "30"]) == 2
    assert not output.exists()


def test_requested_venv_interpreter_symlink_is_preserved(inputs, tmp_path, monkeypatch, capsys):
    args, _, _ = inputs
    requested_python = tmp_path / "venv" / "bin" / "python"
    requested_python.parent.mkdir(parents=True)
    requested_python.symlink_to(sys.executable)
    args[args.index("--python") + 1] = str(requested_python)
    assert launch.main(args + ["--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["argv"][0] == str(requested_python)


def test_missing_initialization_is_rejected_before_gpu_inspection(inputs, monkeypatch):
    args, _, _ = inputs
    monkeypatch.setattr(launch, "gpu_snapshot", lambda: pytest.fail("Missing initialization reached GPU preflight"))
    args[args.index("--pretrained") + 1] += ".missing"
    with pytest.raises(SystemExit) as error:
        launch.main(args + ["--dry-run"])
    assert error.value.code == 2


def test_nvidia_metadata_matches_processes_by_uuid(monkeypatch):
    replies = iter([
        "4, GPU-four, NVIDIA B300, 262144, 96000\n5, GPU-five, NVIDIA B300, 262144, 96000\n",
        "GPU-five, 3940352, /env/bin/python, 96000\nGPU-four, 3940351, /env/bin/python, 96000\n",
    ])
    monkeypatch.setattr(launch.subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0, stdout=next(replies)))
    selected, active = launch._selected_snapshot((4, 5))
    assert [device.uuid for device in selected] == ["GPU-four", "GPU-five"]
    assert {process.pid for process in active} == {3940351, 3940352}


def test_fresh_training_does_not_overwrite_existing_run(inputs, monkeypatch, capsys):
    args, output, _ = inputs
    output.mkdir()
    old = output / "latest.pt"
    old.write_bytes(b"existing-checkpoint")
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **kw: pytest.fail("Existing run was overwritten"))
    assert launch.main(args) == 2
    assert old.read_bytes() == b"existing-checkpoint"
    assert not (output / "launch.json").exists()


@pytest.mark.parametrize("chunk_size", ["0", "-1"])
def test_invalid_attention_chunk_size_fails_before_gpu_inspection(inputs, monkeypatch, chunk_size):
    args, _, _ = inputs
    monkeypatch.setattr(launch, "gpu_snapshot", lambda: pytest.fail("Invalid chunk size reached GPU preflight"))
    with pytest.raises(SystemExit) as error:
        launch.main(args + ["--dry-run", "--attention-chunk-size", chunk_size])
    assert error.value.code == 2


def test_reset_data_cursor_requires_resume_before_gpu_inspection(inputs, monkeypatch):
    args, _, _ = inputs
    monkeypatch.setattr(launch, "gpu_snapshot", lambda: pytest.fail("Invalid reset reached GPU preflight"))
    with pytest.raises(SystemExit) as error:
        launch.main(args + ["--reset-data-cursor", "--dry-run"])
    assert error.value.code == 2


def test_custom_loss_launcher_command_is_accepted_by_training_parser(inputs, tmp_path, monkeypatch, capsys):
    from multicamera4rc import train

    args, _, _ = inputs
    loss_config = tmp_path / "loss.json"
    loss_config.write_text(json.dumps({"foreground_fraction": 0.5}))
    assert launch.main(args + ["--loss-config", str(loss_config), "--dry-run"]) == 0
    command = json.loads(capsys.readouterr().out)["argv"]
    training_arguments = command[command.index("multicamera4rc.train")+1:]
    monkeypatch.setattr(train.DistributedContext, "initialize", lambda *_: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(train, "_train", lambda parsed, _: parsed)
    parsed = train.main(training_arguments)
    assert parsed.loss_config == str(loss_config.resolve())
    assert parsed.loss_profile == "auto"
