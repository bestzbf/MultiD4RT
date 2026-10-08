"""Verify source balance, real C/T labels and cross-stage optimizer/LR resume."""
import copy
from collections import Counter
import json
from pathlib import Path
import sys

import pytest
import torch

from multicamera4rc.datasets.mixture import create_dataset, read_data_config
from multicamera4rc.datasets.sampling import coverage_permutation, source_quotas
from multicamera4rc.schedule import LearningRatePlan
from multicamera4rc.train import _sources, main as train_main
from test_adapter_mixture import _write_configuration
from scripts.build_variable_curriculum import FAMILY_PERCENT, family, make_config


def test_coverage_permutation_visits_each_window_without_allocating_pool():
    for size in (1, 2, 12, 97, 1000):
        a, b = coverage_permutation(size, 49)
        assert {(a*i+b) % size for i in range(size)} == set(range(size))
    assert source_quotas([35, 12, 12, 11, 4, 7, 4, 3, 3, 3, 4, 2], 100000)[-1] == 2000


def test_balanced_coverage_controls_dataset_ratios_and_rotates_large_pools(tmp_path):
    config = read_data_config(_write_configuration(tmp_path))
    config.update(sampling="balanced_coverage", epoch_size=20)
    config["sources"][0].update(weight=9., strides=[1])
    config["sources"][1].update(weight=1., strides=[1])
    data = create_dataset(config)
    plans = [data.sample_plan(i) for i in range(len(data))]
    assert Counter(p["source"]["name"] for p in plans) == {"kubric_multicam":18, "mugen_h5":2}
    # There are four windows in the second pool; two visits per epoch must
    # reach the remaining two next epoch rather than restart its prefix.
    first = {tuple(p["times"]) for p in plans if p["source"]["name"] == "mugen_h5"}
    data.set_epoch(1)
    second = {tuple(data.sample_plan(i)["times"]) for i in (18,19)}
    assert len(first | second) == 4


def test_variable_camera_time_layout_keeps_actual_gt_and_frame_gaps(tmp_path):
    config = read_data_config(_write_configuration(tmp_path))
    source = config["sources"][0]
    source.update(strides=[1,2], layouts=[dict(cameras=1,num_times=3),dict(cameras=2,num_times=5)])
    config.update(sources=[source], sampling="balanced_coverage", epoch_size=30)
    data = create_dataset(config); layouts=set()
    for i in range(len(data)):
        plan = data.sample_plan(i); sample=data[i]
        c,t = plan["cameras"],plan["num_times"]
        layouts.add((c,t))
        assert sample["images"].shape[0] == c*t
        assert sample["camera_id"].unique().numel() == c
        assert sample["targets"]["sparse_trajectory"].shape[1] == t
        assert sample["targets"]["sparse_visibility"].shape[1:3] == (t,c)
        # Native frame IDs and gaps also index renderer object poses; neither
        # may be silently renumbered when the sampled clip starts after zero.
        assert sample["query_time_idx"].tolist() == plan["times"]
        assert sample["query_time_idx"].unique().numel() == t
    assert layouts == {(1,3),(2,5)}


def test_balanced_coverage_preserves_short_native_sequences_and_rejects_fake_flow_gaps(tmp_path):
    config = read_data_config(_write_configuration(tmp_path))
    source = config["sources"][1]
    source.update(num_times=32,strides=[1],layouts=[dict(cameras=1,num_times=16),dict(cameras=1,num_times=32)])
    config.update(sources=[source],sampling="balanced_coverage",epoch_size=2)
    data=create_dataset(config)
    assert len(data.sources[0]["pool"]) == 1
    assert data.sample_plan(0)["num_times"] == 11
    invalid = copy.deepcopy(config)
    invalid["sources"][0].update(adapter="dynamic_replica",dense_kind="replica_flow",strides=[1,2])
    with pytest.raises(ValueError, match="adjacent flow"):
        create_dataset(invalid)


def test_lr_continuation_warmup_is_continuous_and_stage_resume_preserves_end(tmp_path):
    plan=LearningRatePlan("cosine",25000,50000,[1e-6,1e-5],warmup_steps=500,
                          min_ratio=.1,warmup_start_ratio=.3)
    assert plan.factor(25000)*1e-5 == pytest.approx(3.014e-6)
    assert plan.factor(25499) == 1.
    assert .1 < plan.factor(32000) < 1.
    assert plan.factor(49999) == pytest.approx(.1)
    config=_write_configuration(tmp_path)
    first=train_main(["--data-config",str(config),"--output",str(tmp_path/"first"),
        "--preset","tiny","--device","cpu","--steps","1","--lr-end-step","10",
        "--lr-schedule","cosine","--warmup-steps","2","--warmup-start-ratio",".3",
        "--batch-size","1","--source-count","1","--cpu-threads","1"])
    saved=torch.load(first,weights_only=True)
    second=train_main(["--data-config",str(config),"--output",str(tmp_path/"second"),
        "--resume",str(first),"--device","cpu","--steps","2","--reset-data-cursor",
        "--batch-size","1","--source-count","1","--cpu-threads","1"])
    resumed=torch.load(second,weights_only=True)
    assert resumed["training_state"]["lr_plan"] == saved["training_state"]["lr_plan"]
    assert resumed["training_state"]["lr_plan"]["end_step"] == 10
    assert resumed["optimizer"]["state"]
    assert all(int(v["step"]) == 2 for v in resumed["optimizer"]["state"].values())


def test_long_time_budget_reduces_references_without_dropping_input_times():
    for times,references in [(8,4),(16,4),(24,2),(32,2)]:
        batch=dict(observation_valid=torch.ones(1,times,dtype=torch.bool),
                   query_time_idx=torch.arange(times)[None],targets={})
        selected=_sources(batch,4,max_source_time_pairs=64)
        assert selected.shape == (1,references)
        assert batch["observation_valid"].all() and batch["query_time_idx"].numel() == times


def test_curriculum_balances_families_preserves_manifests_and_caps_native_motion(tmp_path):
    root=Path(__file__).resolve().parents[2]
    original=read_data_config(root/"multicamera4rc/audits/native_motion_stereo_20261002/measurements/full_mixture_native_motion_stereo.yaml")
    for stage in range(3):
        config=make_config(original,stage,tmp_path/"invalid.jsonl")
        assert config["sampling"] == "balanced_coverage" and config["epoch_size"] == 100000
        names={s["name"]:s for s in config["sources"]}
        for old in original["sources"]:
            name="kubric_variable" if family(old["name"]) == "kubric" else old["name"]
            assert names[name]["manifest"] == (original["sources"][0]["manifest"] if name == "kubric_variable" else old["manifest"])
        assert {g:sum(s["weight"] for s in config["sources"] if family(s["name"]) == g) for g in FAMILY_PERCENT} == FAMILY_PERCENT
        assert {v["cameras"] for v in names["kubric_variable"]["layouts"]} == set(range(1,11))
        assert names["dynamic_replica"]["num_times"] <= 16
        assert names["pointodyssey"]["num_times"] <= 24
        assert names["vkitti2_Camera_0"]["num_times"] <= 24
        assert names["dynamic_replica"]["strides"] == [1]
    assert names["scannet"]["num_times"] == 32


def test_stage_changes_preserve_sampling_cursor_and_lr_after_initial_cutover(monkeypatch):
    root=Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root/"scripts"))
    from queue_variable_continuation import training_argv
    stage=dict(data_config="data.yaml",end_step=32000)
    first=training_argv("python",root,"checkpoint.pt",stage,"output",True)
    later=training_argv("python",root,"checkpoint.pt",stage,"output",False)
    assert "--reset-data-cursor" in first and "--lr-end-step" in first
    assert "--reset-data-cursor" not in later and "--override-lr" not in later and "--lr-end-step" not in later
    assert later[later.index("--max-source-time-pairs")+1] == "64"


def test_interrupted_curriculum_restores_checkpoint_without_restarting_lr_or_cursor():
    from scripts.queue_variable_continuation import training_argv
    from scripts.recover_variable_continuation import recovery_phases
    root = Path("/repo")
    stages = [dict(data_config="first.yaml", end_step=32000),
              dict(data_config="second.yaml", end_step=40000)]
    phases = [dict(name="training_stage_1", argv=training_argv("python", root, "25000.pt", stages[0], "first", True)),
              dict(name="fixed_validation_32000", argv=["python", "-m", "multicamera4rc.evaluate", "--plan", "fixed.json"]),
              dict(name="training_stage_2", argv=training_argv("python", root, "first/latest.pt", stages[1], "second", False))]
    unchanged = copy.deepcopy(phases)
    recovered = recovery_phases(phases, 0, "27500.pt", "target_time", 1000)
    argv = recovered[0]["argv"]
    assert argv[argv.index("--resume")+1] == "27500.pt"
    assert all(flag not in argv for flag in ("--override-lr", "--reset-data-cursor", "--lr-schedule",
        "--lr", "--backbone-lr", "--warmup-steps", "--warmup-start-ratio", "--min-lr-ratio", "--lr-end-step"))
    for phase in recovered:
        assert phase["argv"][phase["argv"].index("--decoder-memory")+1] == "target_time"
    assert argv[argv.index("--save-every")+1] == "1000"
    assert recovered[2]["argv"][recovered[2]["argv"].index("--resume")+1] == "first/latest.pt"
    assert phases == unchanged  # Previous failure evidence stays immutable.
    second = recovery_phases(phases, 2, "35000.pt", "target_time", 1000)
    assert len(second) == 1 and second[0]["argv"][second[0]["argv"].index("--resume")+1] == "35000.pt"
    with pytest.raises(ValueError, match="failed training phase"):
        recovery_phases(phases, 1, "32000.pt", "target_time", 1000)


@pytest.mark.parametrize("saved_step", [27500, 25000])
def test_recovery_controller_requires_selected_step_and_runs_remaining_phase(tmp_path, monkeypatch, saved_step):
    from scripts import queue_variable_continuation as controller
    previous = tmp_path/"previous"
    control = tmp_path/"recovery"
    active_dir = tmp_path/"outputs/training_preflight"
    for path in (previous, control, active_dir):
        path.mkdir(parents=True)
    old = dict(pid=99999999, control_dir=str(previous), stage="failed")
    (previous/"job.json").write_text(json.dumps(old))
    (previous/"progress.json").write_text(json.dumps(dict(child_pid=99999998)))
    (active_dir/"active_training.json").write_text(json.dumps(old))
    checkpoint = tmp_path/"checkpoint.pt"
    torch.save(dict(step=saved_step, optimizer=dict(state={0: dict(step=torch.tensor(saved_step))}),
                    training_state=dict(world_size=2, batch_size=1, optimizer_updates=saved_step)), checkpoint)
    job = dict(repo=str(tmp_path), control_dir=str(control), previous_job=old,
               recovery_of=str(previous/"job.json"), commit="verified", resume_step=27500,
               selected_checkpoint=str(checkpoint), evidence_sha256={}, environment={},
               run=str(tmp_path/"run"), data_config="unchanged.yaml", phases=[
                   dict(name="training_stage_1", argv=[sys.executable, "-c", "pass"], target_step=32000)])
    job_file = control/"job.json"
    job_file.write_text(json.dumps(job))
    monkeypatch.setattr(controller, "proc", lambda pid: None)
    monkeypatch.setattr(controller.subprocess, "check_output", lambda *a, **kw: "verified\n")
    if saved_step != 27500:
        with pytest.raises(RuntimeError, match="expected 27500 checkpoint"):
            controller.run_controller(job_file)
        assert json.loads((active_dir/"active_training.json").read_text()) == old
    else:
        controller.run_controller(job_file)
        report = json.loads((control/"progress.json").read_text())
        assert report["stage"] == "passed"
        assert report["restored_training_state"]["optimizer_updates"] == 27500
        assert report["phases"][0]["exit_code"] == 0
        assert json.loads((active_dir/"active_training.json").read_text())["resume_step"] == 27500
