"""Explicit frame/camera labels, source anchors and synchronized packing."""

import pytest
import torch

from multicamera4rc.metadata import build_metadata, pack_synchronized_images


def _complete():
    return torch.tensor([[0, 0, 0, 1, 1, 1]]), torch.tensor([[0, 5, 10, 0, 5, 10]])


def test_six_observations_group_into_three_physical_times_and_source_anchors():
    cameras, frames = _complete()
    metadata = build_metadata(cameras, frames)
    assert metadata.query_times.tolist() == [[0, 5, 10]]
    assert metadata.internal_times.tolist() == [[0, 5, 10]]
    assert metadata.source_times.tolist() == [[0, 5, 10, 0, 5, 10]]
    assert metadata.source_slot.tolist() == [[0, 1, 2, 0, 1, 2]]
    assert metadata.memory_index.tolist() == [[[0, 3], [1, 4], [2, 5]]]
    assert metadata.memory_valid.all()
    assert metadata.camera_values.tolist() == [[0, 1]]


def test_query_subset_adds_source_anchor_without_exposing_extra_output_time():
    cameras, frames = _complete()
    metadata = build_metadata(cameras, frames, query_time_idx=[10], source_obs_idx=[1])
    assert metadata.query_times.tolist() == [[10]]
    assert metadata.source_times.tolist() == [[5]]
    assert metadata.internal_times.tolist() == [[10, 5]]
    assert metadata.target_slot.tolist() == [[0]]
    assert metadata.source_slot.tolist() == [[1]]
    assert metadata.memory_index.tolist() == [[[2, 5], [1, 4]]]


def test_per_batch_sources_missing_views_and_different_camera_counts():
    cameras = torch.tensor([[7, 7, 7, 9, 9, -1], [3, 3, 8, 8, 12, 12]])
    frames = torch.tensor([[0, 5, 10, 0, 10, 99], [100, 120, 100, 120, 100, 120]])
    valid = torch.tensor([[True, True, True, True, True, False], [True] * 6])
    metadata = build_metadata(
        cameras, frames, query_time_idx=[[0, 10], [100, 120]],
        source_obs_idx=[[1, 4], [0, 3]], observation_valid=valid,
    )
    assert metadata.source_index.tolist() == [[1, 4], [0, 3]]
    assert metadata.source_times.tolist() == [[5, 10], [100, 120]]
    assert metadata.source_slot.tolist() == [[2, 1], [0, 1]]
    assert metadata.camera_values.tolist() == [[7, 9, -1], [3, 8, 12]]
    assert metadata.camera_valid.tolist() == [[True, True, False], [True, True, True]]
    assert metadata.internal_times.tolist() == [[0, 10, 5], [100, 120, 0]]
    assert metadata.internal_valid.tolist() == [[True, True, True], [True, True, False]]
    assert metadata.memory_valid.tolist() == [
        [[True, True, False], [True, True, False], [True, False, False]],
        [[True, True, True], [True, True, True], [False, False, False]],
    ]
    # A missing camera at frame 5 must be masked, rather than filled with a
    # nearby camera/time observation and presented as actual evidence.
    assert metadata.memory_index[0, 2, 0].item() == 1
    assert metadata.memory_index[1, 1].tolist() == [1, 3, 5]


def test_shared_source_subset_is_broadcast_and_invalid_sources_stay_marked():
    cameras, frames = _complete()
    cameras, frames = cameras.expand(2, -1).clone(), frames.expand(2, -1).clone()
    valid = torch.ones_like(cameras, dtype=torch.bool)
    valid[1, 4] = False
    metadata = build_metadata(cameras, frames, source_obs_idx=[1, 4], observation_valid=valid)
    assert metadata.source_index.tolist() == [[1, 4], [1, 4]]
    assert metadata.source_valid.tolist() == [[True, True], [True, False]]


def test_duplicate_camera_time_observation_is_rejected_but_padding_is_ignored():
    with pytest.raises(ValueError, match="duplicate observation"):
        build_metadata([[0, 0]], [[5, 5]])
    metadata = build_metadata(
        [[0, -1, -1]], [[5, 0, 0]], observation_valid=[[True, False, False]],
    )
    assert metadata.query_times.tolist() == [[5]]
    assert metadata.source_valid.tolist() == [[True, False, False]]


@pytest.mark.parametrize("query", [[15], [0, 0]])
def test_missing_or_duplicate_query_frame_labels_are_rejected(query):
    cameras, frames = _complete()
    with pytest.raises(ValueError, match="observed target image|unique frame labels"):
        build_metadata(cameras, frames, query_time_idx=query)


@pytest.mark.parametrize("sources", [[6], [-1], [1, 1]])
def test_invalid_or_duplicate_source_selection_is_rejected(sources):
    cameras, frames = _complete()
    with pytest.raises(ValueError, match="out of range|duplicates"):
        build_metadata(cameras, frames, source_obs_idx=sources)


def test_frame_gaps_are_preserved_and_never_renumbered_as_sample_positions():
    metadata = build_metadata([[0, 0, 0, 1, 1, 1]], [[100, 105, 120, 100, 105, 120]])
    assert metadata.query_times.tolist() == [[100, 105, 120]]
    assert torch.diff(metadata.query_times).tolist() == [[5, 15]]
    assert (metadata.internal_times - metadata.source_times[:, 1:2]).tolist() == [[-5, 0, 15]]


def test_pack_synchronized_images_preserves_camera_major_images_and_labels():
    images = torch.arange(2 * 2 * 3 * 3 * 2 * 3).reshape(2, 2, 3, 3, 2, 3)
    packed = pack_synchronized_images(
        images, frame_indices=[[0, 5, 10], [100, 105, 110]], camera_ids=[[17, 9], [3, 22]],
    )
    assert packed["images"].shape == (2, 6, 3, 2, 3)
    assert packed["camera_id"].tolist() == [[17, 17, 17, 9, 9, 9], [3, 3, 3, 22, 22, 22]]
    assert packed["time_idx"].tolist() == [[0, 5, 10, 0, 5, 10], [100, 105, 110, 100, 105, 110]]
    assert packed["query_time_idx"].tolist() == [[0, 5, 10], [100, 105, 110]]
    torch.testing.assert_close(packed["images"][0, 4], images[0, 1, 1])
    torch.testing.assert_close(packed["images"][1, 2], images[1, 0, 2])


def test_pack_synchronized_images_has_a_simple_shared_frame_default():
    packed = pack_synchronized_images(torch.zeros(1, 2, 3, 3, 2, 3))
    assert packed["camera_id"].tolist() == [[0, 0, 0, 1, 1, 1]]
    assert packed["time_idx"].tolist() == [[0, 1, 2, 0, 1, 2]]
