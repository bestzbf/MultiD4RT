"""Utilities for safe camera-major layouts on tubelet video encoders.

The encoder groups consecutive input slots into temporal tubelets.  A camera
boundary must therefore coincide with a tubelet boundary; otherwise a single
3-D convolution receives pixels from two unrelated cameras.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraMajorSlotPlan:
    slot_camera: np.ndarray
    slot_time: np.ndarray
    valid_slot: np.ndarray
    physical_times_per_camera: int


@dataclass(frozen=True)
class LogicalBCTSlotMapping:
    """Logical camera/time masks and the map back to fixed input slots.

    Camera ids in adapter metadata are local ids and need not be globally
    meaningful.  The mapping canonicalizes the valid ids in sorted order so a
    mixed batch can be padded to one ``[B,C,T]`` shape by the collate layer.
    """

    camera_valid: np.ndarray       # [C]
    time_valid: np.ndarray         # [C,T]
    bct_slot_index: np.ndarray     # [C,T], -1 for absent cells
    canonical_slot_camera: np.ndarray  # [S], -1 for invalid slots
    canonical_slot_time: np.ndarray    # [S], -1 for invalid slots


def aligned_times_per_camera(num_times: int, tubelet_size: int = 2) -> int:
    """Return the physical camera-block length needed for ``num_times``.

    A one-frame observation becomes ``[real, invalid-zero]`` for a size-2
    tubelet.  More generally each camera block is rounded up independently.
    """
    if num_times < 1:
        raise ValueError(f"num_times must be positive, got {num_times}")
    if tubelet_size < 1:
        raise ValueError(f"tubelet_size must be positive, got {tubelet_size}")
    return ((int(num_times) + int(tubelet_size) - 1) // int(tubelet_size)) * int(tubelet_size)


def valid_time_choices(
    *,
    num_slots: int,
    num_cameras: int,
    min_times: int = 1,
    max_times: int | None = None,
    tubelet_size: int = 2,
    even_multi_time: bool = True,
) -> list[int]:
    """Enumerate logical time counts that fit without crossing cameras.

    With ``even_multi_time=True``, one time is retained as a supported special
    case while all multi-time choices are multiples of ``tubelet_size``.
    """
    if num_slots < 1 or num_cameras < 1:
        return []
    upper = num_slots // num_cameras
    if max_times is not None:
        upper = min(upper, int(max_times))
    lower = max(1, int(min_times))
    choices: list[int] = []
    for num_times in range(lower, upper + 1):
        if even_multi_time and num_times != 1 and num_times % tubelet_size:
            continue
        required = num_cameras * aligned_times_per_camera(num_times, tubelet_size)
        if required <= num_slots:
            choices.append(num_times)
    return choices


def build_camera_major_slot_plan(
    *,
    num_slots: int,
    num_cameras: int,
    num_times: int,
    tubelet_size: int = 2,
) -> CameraMajorSlotPlan:
    """Build a fixed-length camera-major plan with per-camera padding."""
    if num_slots < 1:
        raise ValueError(f"num_slots must be positive, got {num_slots}")
    if num_cameras < 1:
        raise ValueError(f"num_cameras must be positive, got {num_cameras}")
    block = aligned_times_per_camera(num_times, tubelet_size)
    required = num_cameras * block
    if required > num_slots:
        raise ValueError(
            f"layout needs {required} slots ({num_cameras} cameras x {block}) "
            f"but only {num_slots} are available"
        )

    slot_camera = np.zeros(num_slots, dtype=np.int64)
    slot_time = np.zeros(num_slots, dtype=np.int64)
    valid_slot = np.zeros(num_slots, dtype=bool)
    for camera in range(num_cameras):
        start = camera * block
        stop = start + block
        slot_camera[start:stop] = camera
        slot_time[start:stop] = np.minimum(np.arange(block), num_times - 1)
        valid_slot[start : start + num_times] = True
    return CameraMajorSlotPlan(slot_camera, slot_time, valid_slot, block)


def validate_slot_layout_schema(
    slot_camera: np.ndarray,
    slot_time: np.ndarray,
    valid_slot: np.ndarray,
    *,
    num_cameras: int | None = None,
    num_times: int | None = None,
    tubelet_size: int = 2,
    require_tubelet_divisible: bool = True,
) -> tuple[int, int]:
    """Validate fixed-slot camera/time metadata used by the V2 path.

    The fixed encoder consumes consecutive tubelets, so valid observations
    from different cameras may not share one tubelet.  Valid cells must also
    be unique in logical ``(camera,time)`` space; otherwise a later scatter
    into BCT memory would have an ambiguous destination.
    """
    camera = np.asarray(slot_camera, dtype=np.int64)
    time = np.asarray(slot_time, dtype=np.int64)
    valid = np.asarray(valid_slot, dtype=bool)
    if camera.ndim != 1 or time.ndim != 1 or valid.ndim != 1:
        raise ValueError("slot_camera, slot_time, and valid_slot must be 1-D")
    if not (camera.shape == time.shape == valid.shape):
        raise ValueError("slot_camera, slot_time, and valid_slot must have equal lengths")
    if tubelet_size < 1:
        raise ValueError(f"tubelet_size must be positive, got {tubelet_size}")
    # A single observation is a supported special case; the encoder's
    # tubelet path treats it as one real frame plus an invalid zero pad.
    if require_tubelet_divisible and camera.size % tubelet_size and camera.size != 1:
        raise ValueError(
            f"slot count {camera.size} is not divisible by tubelet_size={tubelet_size}"
        )
    if np.any(camera < 0) or np.any(time < 0):
        raise ValueError("slot_camera and slot_time must be non-negative")
    if not valid.any():
        raise ValueError("slot layout must contain at least one valid slot")

    inferred_cameras = int(camera[valid].max()) + 1
    inferred_times = int(time[valid].max()) + 1
    cameras = inferred_cameras if num_cameras is None else int(num_cameras)
    times = inferred_times if num_times is None else int(num_times)
    if cameras < inferred_cameras or times < inferred_times:
        raise ValueError(
            "layout_num_cameras/layout_num_times are smaller than valid metadata"
        )
    if cameras < 1 or times < 1:
        raise ValueError("layout dimensions must be positive")
    if np.any(camera[valid] >= cameras) or np.any(time[valid] >= times):
        raise ValueError("valid slot metadata exceeds declared layout dimensions")

    seen: set[tuple[int, int]] = set()
    for camera_value, time_value in zip(camera[valid], time[valid]):
        cell = (int(camera_value), int(time_value))
        if cell in seen:
            raise ValueError(
                "duplicate valid camera/time observation: "
                f"camera={cell[0]}, time={cell[1]}"
            )
        seen.add(cell)

    if camera.size % tubelet_size == 0:
        assert_no_cross_camera_tubelets(camera, valid, tubelet_size=tubelet_size)
    return cameras, times


def derive_slot_time_ids(
    slot_camera: np.ndarray,
    slot_time: np.ndarray,
    valid_slot: np.ndarray,
    *,
    slot_time_id: np.ndarray | None = None,
) -> np.ndarray:
    """Return discrete per-camera time ids without quantizing physical time.

    Older adapters used integer ``slot_time`` for both meanings.  Preserve that
    contract when those values already form a valid logical layout.  Otherwise
    assign ids in valid slot order independently for each camera; invalid slots
    inherit the preceding id in their camera block and remain masked.
    """
    camera = np.asarray(slot_camera, dtype=np.int64)
    physical_time = np.asarray(slot_time)
    valid = np.asarray(valid_slot, dtype=bool)
    if camera.ndim != 1 or physical_time.ndim != 1 or valid.ndim != 1:
        raise ValueError("slot_camera, slot_time, and valid_slot must be 1-D")
    if not (camera.shape == physical_time.shape == valid.shape):
        raise ValueError("slot_camera, slot_time, and valid_slot must have equal lengths")
    if not np.isfinite(physical_time[valid]).all():
        raise ValueError("valid physical slot_time values must be finite")

    if slot_time_id is not None:
        result = np.asarray(slot_time_id)
        if result.shape != camera.shape or result.ndim != 1:
            raise ValueError("slot_time_id must be 1-D and match slot_time")
        if not np.issubdtype(result.dtype, np.integer):
            if not np.isfinite(result).all() or not np.equal(result, np.rint(result)).all():
                raise ValueError("slot_time_id must contain integer values")
        result = result.astype(np.int64, copy=False)
        if np.any(result < 0):
            raise ValueError("slot_time_id must be non-negative")
        valid_cells = list(zip(camera[valid].tolist(), result[valid].tolist()))
        if len(valid_cells) != len(set(valid_cells)):
            raise ValueError("slot_time_id contains duplicate valid camera/time cells")
        return result

    # Backward compatibility is deliberately narrow: only exact, non-negative
    # integers with unique valid (camera,time) cells are safe as array ids.
    if np.issubdtype(physical_time.dtype, np.integer):
        legacy = physical_time.astype(np.int64, copy=False)
        legacy_cells = list(zip(camera[valid].tolist(), legacy[valid].tolist()))
        if np.all(legacy >= 0) and len(legacy_cells) == len(set(legacy_cells)):
            return legacy.copy()
    else:
        exact_integer = (
            np.isfinite(physical_time).all()
            and np.equal(physical_time, np.rint(physical_time)).all()
            and np.all(physical_time >= 0)
        )
        if exact_integer:
            legacy = np.rint(physical_time).astype(np.int64)
            legacy_cells = list(zip(camera[valid].tolist(), legacy[valid].tolist()))
            if len(legacy_cells) == len(set(legacy_cells)):
                return legacy

    result = np.zeros(camera.shape, dtype=np.int64)
    for camera_id in np.unique(camera):
        camera_slots = np.flatnonzero((camera == camera_id) & valid)
        if camera_slots.size:
            result[camera_slots] = np.arange(camera_slots.size, dtype=np.int64)
            camera_all_slots = np.flatnonzero(camera == camera_id)
            for slot in camera_all_slots:
                previous = camera_slots[camera_slots <= slot]
                result[slot] = max(int(previous.size) - 1, 0)
    return result


def infer_num_local_times(slot_time_id: np.ndarray, valid_slot: np.ndarray) -> int:
    """Infer the dense BCT time dimension from valid discrete ids."""
    time_id = np.asarray(slot_time_id, dtype=np.int64)
    valid = np.asarray(valid_slot, dtype=bool)
    if time_id.shape != valid.shape or time_id.ndim != 1:
        raise ValueError("slot_time_id and valid_slot must be matching 1-D arrays")
    return int(time_id[valid].max()) + 1 if valid.any() else 0


def build_logical_bct_slot_mapping(
    slot_camera: np.ndarray,
    slot_time: np.ndarray,
    valid_slot: np.ndarray,
    *,
    num_cameras: int | None = None,
    num_times: int | None = None,
) -> LogicalBCTSlotMapping:
    """Build ``camera_valid``, ``time_valid`` and BCT-to-slot indices.

    This is intentionally derived from the authoritative fixed-slot fields so
    adapters cannot provide a second, stale mapping that disagrees with the
    video tensor.
    """
    camera = np.asarray(slot_camera, dtype=np.int64)
    time = np.asarray(slot_time, dtype=np.int64)
    valid = np.asarray(valid_slot, dtype=bool)
    cameras, times = validate_slot_layout_schema(
        camera,
        time,
        valid,
        num_cameras=num_cameras,
        num_times=num_times,
    )

    camera_values = np.unique(camera[valid])
    canonical_camera = np.full(camera.shape, -1, dtype=np.int64)
    canonical_time = np.full(time.shape, -1, dtype=np.int64)
    value_to_local = {int(value): index for index, value in enumerate(camera_values)}
    camera_valid = np.zeros(cameras, dtype=bool)
    time_valid = np.zeros((cameras, times), dtype=bool)
    bct_slot_index = np.full((cameras, times), -1, dtype=np.int64)
    for slot in np.flatnonzero(valid):
        local_camera = value_to_local[int(camera[slot])]
        local_time = int(time[slot])
        canonical_camera[slot] = local_camera
        canonical_time[slot] = local_time
        camera_valid[local_camera] = True
        time_valid[local_camera, local_time] = True
        bct_slot_index[local_camera, local_time] = int(slot)

    return LogicalBCTSlotMapping(
        camera_valid=camera_valid,
        time_valid=time_valid,
        bct_slot_index=bct_slot_index,
        canonical_slot_camera=canonical_camera,
        canonical_slot_time=canonical_time,
    )


def cross_camera_tubelet_indices(
    slot_camera: np.ndarray,
    valid_slot: np.ndarray,
    tubelet_size: int = 2,
) -> np.ndarray:
    """Return tubelet indices containing valid slots from multiple cameras."""
    camera = np.asarray(slot_camera, dtype=np.int64).reshape(-1)
    valid = np.asarray(valid_slot, dtype=bool).reshape(-1)
    if camera.shape != valid.shape:
        raise ValueError("slot_camera and valid_slot must have identical shapes")
    if camera.size % tubelet_size:
        raise ValueError(
            f"slot count {camera.size} is not divisible by tubelet_size={tubelet_size}"
        )
    camera = camera.reshape(-1, tubelet_size)
    valid = valid.reshape(-1, tubelet_size)
    bad: list[int] = []
    for index, (camera_row, valid_row) in enumerate(zip(camera, valid)):
        if np.unique(camera_row[valid_row]).size > 1:
            bad.append(index)
    return np.asarray(bad, dtype=np.int64)


def assert_no_cross_camera_tubelets(
    slot_camera: np.ndarray,
    valid_slot: np.ndarray,
    tubelet_size: int = 2,
) -> None:
    bad = cross_camera_tubelet_indices(slot_camera, valid_slot, tubelet_size)
    if bad.size:
        raise ValueError(f"valid slots cross camera boundaries in tubelets {bad.tolist()}")
