"""Exercise real file I/O, optimizer checkpoint resume and prediction export."""

import json

import pytest
import torch

from multicamera4rc.data import MultiCameraManifestDataset, collate_multicamera
from multicamera4rc.demo import make_demo_dataset
from multicamera4rc.infer import main as infer_main
from multicamera4rc.train import _sources, main as train_main


def test_manifest_camera_labels_reorder_visibility_without_changing_observations(tmp_path):
    manifest = make_demo_dataset(tmp_path/"data")
    record = json.loads(manifest.read_text())
    record["clips"][0]["target_camera_id"] = [1, 0]
    manifest.write_text(json.dumps(record))
    target = torch.load(manifest.parent/"targets.pt", weights_only=True)
    target["visibility"] = torch.zeros(6, 3, 2, 28, 42)
    target["visibility"][:, :, 0] = 1  # camera1, before label reordering
    torch.save(target, manifest.parent/"targets.pt")
    sample = MultiCameraManifestDataset(manifest)[0]
    assert sample["camera_id"].tolist() == [0, 0, 0, 1, 1, 1]
    assert sample["targets"]["visibility"][:, :, 1].eq(1).all()
    assert sample["targets"]["visibility"][:, :, 0].eq(0).all()
    batch = collate_multicamera([sample, sample])
    assert batch["images"].shape == (2, 6, 3, 28, 42)


@pytest.mark.parametrize("seed", [0, 2, 3])
def test_source_sampling_prioritizes_valid_sparse_supervision(seed):
    torch.manual_seed(seed)
    batch = {
        "images": torch.zeros(1, 3, 3, 28, 28),
        "observation_valid": torch.ones(1, 3, dtype=torch.bool),
        "targets": {
            "sparse_queries": torch.tensor([[[0., 2., 1.], [2., 2., 1.]]]),
            "sparse_trajectory": torch.zeros(1, 2, 2, 3),
            "sparse_trajectory_valid": torch.tensor([[[False, False], [True, True]]]),
        },
    }
    # Source 0 is a padded/invalid query despite being a valid image observation.
    # A one-source step must keep source 2, which has the only supervised track.
    assert _sources(batch, count=1).tolist() == [[2]]


def test_train_resume_and_infer_real_manifest(tmp_path):
    manifest = make_demo_dataset(tmp_path/"data")
    checkpoint = train_main(["--manifest", str(manifest), "--output", str(tmp_path/"train"),
                             "--preset", "tiny", "--device", "cpu", "--steps", "1",
                             "--source-count", "2", "--cpu-threads", "1",
                             "--decoder-memory", "target_time"])
    first = torch.load(checkpoint, weights_only=True)
    assert first["step"] == 1 and first["optimizer"]["state"]
    assert first["architecture"] == "multicamera4rc"
    # Emulate a pre-scope native checkpoint while retaining real optimizer state.
    first["config"].pop("decoder_memory")
    torch.save(first, checkpoint)
    checkpoint = train_main(["--manifest", str(manifest), "--output", str(tmp_path/"resumed"),
                             "--resume", str(checkpoint), "--device", "cpu", "--steps", "2",
                             "--source-count", "2", "--cpu-threads", "1", "--no-checkpointing"])
    second = torch.load(checkpoint, weights_only=True)
    assert second["step"] == 2
    assert second["config"]["decoder_memory"] == "target_time"
    assert second["config"]["gradient_checkpointing"] is False
    assert not torch.equal(first["model"]["track_head.scratch.output_conv2.2.weight"],
                           second["model"]["track_head.scratch.output_conv2.2.weight"])
    infer_main(["--manifest", str(manifest), "--checkpoint", str(checkpoint),
                "--output", str(tmp_path/"prediction"), "--device", "cpu", "--source-obs", "1",
                "--query-times", "10", "--cpu-threads", "1", "--amp", "--no-query-chunking"])
    prediction = torch.load(tmp_path/"prediction"/"clip_000000.pt", weights_only=True)
    assert prediction["trajectory"].shape == (1, 1, 1, 28, 42, 3)
    assert prediction["visibility"].shape == (1, 1, 1, 2, 28, 42)
    assert prediction["source_time_idx"].tolist() == [[5]]
    assert prediction["query_time_idx"].tolist() == [[10]]
    assert prediction["clip_id"] == "synthetic_smoke"
    assert torch.isfinite(prediction["trajectory"]).all()
    assert all(value.device.type == "cpu" for value in prediction["ray_list"])
    settings = json.loads((tmp_path/"prediction"/"inference.json").read_text())
    assert settings["model_config"]["decoder_memory"] == "target_time"
    assert settings["model_config"]["attention_chunk_size"] is None
    assert settings["amp"] is True
