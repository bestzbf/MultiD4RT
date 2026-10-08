"""Native dataset sampling and padding for the MultiCamera4RC contract."""

from __future__ import annotations

import json
import math
import os
from bisect import bisect_right
from itertools import chain
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset

from .registry import create_adapter

_NATIVE_MULTICAMERA = ("kubric_multicam", "egoexo4d", "syn4d")

class _InvalidDenseClip(RuntimeError):
    """A readable clip whose actual depth mask contains no supervised pixel."""


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def read_data_config(path):
    """Read JSON or safe YAML; dataset roots are relative to this config."""
    path = Path(path).resolve()
    if path.suffix.lower() == ".json":
        config = json.loads(path.read_text())
    else:
        try:
            import yaml
        except ImportError as error:
            raise ImportError("YAML data configs require pip install -e '.[data]'") from error
        config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("data config must be a mapping")
    config = dict(config)
    allowed = {"image_size", "image_mode", "num_times", "num_queries", "epoch_size", "seed",
               "include_visibility", "supervision", "sources", "sampling", "skip_invalid", "invalid_log",
               "kubric_background_policy"}
    unknown = set(config)-allowed
    if unknown:
        raise ValueError(f"unknown data config fields: {sorted(unknown)}")
    if config.get("invalid_log"):
        config["invalid_log"] = str((path.parent/str(config["invalid_log"])).resolve())
    sources = config.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("data config requires a nonempty sources list")
    resolved = []
    for source in sources:
        if not isinstance(source, dict) or not source.get("root"):
            raise ValueError("each data source requires adapter and root")
        source = dict(source)
        source["root"] = str((path.parent/str(source["root"])).resolve())
        if source.get("manifest"):
            source["manifest"] = str((path.parent/str(source["manifest"])).resolve())
        options = dict(source.get("options", {}))
        for key in ("cache_dir", "validation_stamp_dir", "precompute_root", "fast_root", "camera_metadata", "split_metadata", "metadata_root"):
            if options.get(key) is not None:
                options[key] = str((path.parent/str(options[key])).resolve())
        source["options"] = options
        resolved.append(source)
    config["sources"] = resolved
    return config


class MultiCameraAdapterDataset(Dataset):
    """Sample raw adapters before constructing current-model world supervision.

    Sources may choose different camera and target-time counts. Each local
    batch must share a target-time count; per-GPU batch 1 supports all layouts.
    The collator pads observations and camera labels when needed.
    Indices and epochs determine samples independently of worker scheduling.
    """

    def __init__(self, sources, image_size=(280, 504), num_times=3,
                 num_queries=None, epoch_size=1000, seed=0,
                 include_visibility=True, image_mode="fit", supervision="sparse", sampling="weighted",
                 skip_invalid=False, invalid_log=None, kubric_background_policy="auto"):
        from .dense import KUBRIC_BACKGROUND_POLICIES
        if kubric_background_policy not in KUBRIC_BACKGROUND_POLICIES:
            raise ValueError(f"unknown Kubric background policy: {kubric_background_policy}")
        self.kubric_background_policy = kubric_background_policy
        self.image_size = None if image_mode == "native" else tuple(image_size)
        if image_mode != "native" and (len(self.image_size) != 2 or any(isinstance(v, bool) or not isinstance(v, int)
                or v < 28 or v % 14 for v in self.image_size)):
            raise ValueError("image_size must be [H,W], multiples of 14 and at least 28")
        if image_mode not in ("resize", "fit", "pad", "native"):
            raise ValueError("image_mode must be 'native', 'fit', 'resize' or 'pad'")
        self.image_mode = image_mode
        if supervision not in ("sparse", "dense"):
            raise ValueError("supervision must be 'dense' or 'sparse'")
        self.supervision = supervision
        self.num_times = _positive_integer(num_times, "num_times")
        if supervision == "dense":
            if num_queries not in (None, 0) or isinstance(num_queries, bool):
                raise ValueError("dense supervision uses every valid pixel; omit num_queries or set it to 0")
            self.num_queries = 0
        else:
            self.num_queries = _positive_integer(256 if num_queries is None else num_queries, "num_queries")
        self.epoch_size = _positive_integer(epoch_size, "epoch_size")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.seed, self.epoch = seed, 0
        self.include_visibility = bool(include_visibility)
        self.skip_invalid, self.invalid_log = bool(skip_invalid), invalid_log
        if self.skip_invalid and not self.invalid_log:
            raise ValueError("skip_invalid requires a persistent invalid_log; GT exclusions cannot be silent")
        if sampling not in ("weighted", "coverage", "balanced_coverage"):
            raise ValueError("sampling must be weighted, coverage or balanced_coverage")
        self.sampling = sampling
        self.coverage_ends = []
        self.coverage_entries = []
        self.coverage_size = 0
        self.sources, self.weights = [], []
        if not isinstance(sources, list) or not sources:
            raise ValueError("sources must be a nonempty list")
        for source in sources:
            source = dict(source)
            allowed = {"adapter", "root", "split", "cameras", "num_times", "weight", "strides", "sequences", "options", "name", "dense_kind", "manifest", "view_mode", "layouts"}
            if set(source)-allowed:
                raise ValueError(f"unknown source fields: {sorted(set(source)-allowed)}")
            name = str(source["adapter"]).lower()
            partial = {"mugen_h5", "pointodyssey_geometry", "pointodyssey_tracks", "dynamic_replica", "vkitti2", "vkitti2_stereo", "tartanair_v1_zip", "tartanair_v1_stereo", "tartanair_v2_zip", "hypersim_zip"}
            if supervision == "dense" and name not in partial | {"kubric_multicam", "syn4d"}:
                raise ValueError(f"{name}: dense GT requires native per-pixel labels; sparse tracks cannot be densified")
            native_kind = {"dynamic_replica": "replica_flow", "vkitti2": "vkitti_scene_flow", "vkitti2_stereo": "vkitti_scene_flow", "pointodyssey_tracks": "pointodyssey_tracks", "kubric_multicam": "kubric", "syn4d": "syn4d"}.get(name, "geometry")
            kind = source.get("dense_kind", native_kind)
            if supervision == "dense" and kind not in ("kubric", "geometry", "replica_flow", "vkitti_scene_flow", "pointodyssey_tracks", "syn4d"):
                raise ValueError(f"unsupported dense_kind {kind}")
            if supervision == "dense" and kind != "geometry" and kind != native_kind:
                raise ValueError(f"{name} cannot provide {kind} dense labels")
            source_times = _positive_integer(source.get("num_times", self.num_times), "source num_times")
            view_mode = source.get("view_mode", "temporal")
            if view_mode not in ("temporal", "static_multiview"):
                raise ValueError("view_mode must be temporal or static_multiview")
            static_views = view_mode == "static_multiview"
            if static_views:
                if supervision != "dense" or kind != "geometry" or source_times != 1:
                    raise ValueError("static_multiview requires dense geometry and num_times=1")
                if name not in ("mugen_h5", "hypersim_zip"):
                    raise ValueError(f"{name}: static view grouping is not verified")
                if name == "mugen_h5" and source.get("options", {}).get("dataset_name") not in ("blended", "blendedmvs", "taskonomy"):
                    raise ValueError("packed static multiview is verified for BlendedMVS/Taskonomy only")
                from .static_views import camera_choices
                counts = camera_choices(source.get("cameras", 1))
                cameras = max(counts)
            else:
                cameras = _positive_integer(source.get("cameras", 1), "source cameras")
                counts = (cameras,)
            layouts = []
            if "layouts" in source:
                if static_views or sampling != "balanced_coverage":
                    raise ValueError("variable layouts require temporal balanced_coverage sampling")
                if not isinstance(source["layouts"], list) or not source["layouts"]:
                    raise ValueError("layouts must be a nonempty list")
                for layout in source["layouts"]:
                    if not isinstance(layout, dict) or set(layout) - {"cameras", "num_times", "weight"}:
                        raise ValueError("layout fields are cameras, num_times and weight")
                    C = _positive_integer(layout.get("cameras", cameras), "layout cameras")
                    T = _positive_integer(layout["num_times"], "layout num_times")
                    probability = float(layout.get("weight", 1.))
                    if not math.isfinite(probability) or probability <= 0:
                        raise ValueError("layout weights must be finite and positive")
                    layouts.append(dict(cameras=C, num_times=T, weight=probability))
                cameras = max(v["cameras"] for v in layouts)
                source_times = max(v["num_times"] for v in layouts)
                counts = tuple(sorted({v["cameras"] for v in layouts}))
                if len(counts) > 1 and name not in ("kubric_multicam", "syn4d"):
                    raise ValueError("variable camera layouts require kubric_multicam or syn4d; keep native stereo pairs intact")
            minimum_cameras = min(counts)
            minimum_layout_times = min((v["num_times"] for v in layouts), default=source_times)
            weight = float(source.get("weight", 1.0))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("source weights must be finite and positive")
            strides = source.get("strides", [1])
            if not isinstance(strides, list) or not strides:
                raise ValueError("source strides must be a nonempty list")
            strides = tuple(_positive_integer(v, "stride") for v in strides)
            if kind in ("replica_flow", "vkitti_scene_flow") and strides != (1,):
                raise ValueError("native adjacent flow supervision requires strides=[1]; composing flow across gaps is not verified")
            options = dict(source.get("options", {}))
            if "root" in options or "split" in options:
                raise ValueError("root/split belong in the source, not its options")
            if name in ("kubric_multicam", "egoexo4d"):
                fixed = dict(min_cameras=minimum_cameras, max_cameras=cameras,
                             min_times=minimum_layout_times, max_times=source_times,
                             encoder_tubelet_size=1)
                if name == "kubric_multicam":
                    fixed.update(sampler_units="time", pixel_coordinates="array_index_v1")
                    options.setdefault("use_direct_depth_queries", False)
                    options.setdefault("require_depth", True)
                    # The maintained revision-4/movi_tracking_v3 packs store
                    # OpenCV C2W, as used by the source branch's training data.
                    # Legacy Blender packs can opt into conversion explicitly.
                    options.setdefault("extrinsics_are_opengl", False)
                else:
                    options.setdefault("require_camera_pose", True)
                    options.setdefault("require_depth", True)
                for key, value in fixed.items():
                    if key in options and options[key] != value:
                        raise ValueError(f"{name}: {key} must be {value!r} for the configured layout")
                    options[key] = value
            elif name in ("vkitti2_stereo", "tartanair_v1_stereo"):
                if cameras != 2:
                    raise ValueError(f"{name} requires cameras=2")
            elif name == "tartanair_v2_zip":
                if cameras not in (1, 2):
                    raise ValueError("native V2 direction supports one camera or a stereo pair")
                if "stereo" in options and options["stereo"] != (cameras == 2):
                    raise ValueError("V2 stereo option disagrees with camera layout")
                options["stereo"] = cameras == 2
            elif name == "syn4d":
                if supervision != "dense":
                    raise ValueError("Syn4D provides dense per-pixel supervision; use supervision=dense")
                if kind == "geometry":
                    options.setdefault("load_tracking", False)
            elif cameras != 1 and not static_views:
                raise ValueError(f"{name} returns one camera stream; use kubric_multicam/egoexo4d for native multi-camera clips")
            if name == "dynamic_replica":
                options.setdefault("pixel_coordinates", "array_index_v1")
                if supervision == "dense":
                    options.setdefault("load_trajectories", False)
                    options.setdefault("rgb_from_trajectory", False)
                    options.setdefault("skip_depth_when_tracks", False)
            if supervision == "dense" and name == "vkitti2":
                options.setdefault("strict_calibration", True)
                options.setdefault("load_precomputed", False)
                options.setdefault("load_normals", False)
                options.setdefault("load_flow", False)
            if name == "mugen_h5":
                # Packed dynamic data must use measured trajectories. A depth
                # pixel alone does not establish identity across moving scenes.
                options.setdefault("generate_static_tracks", False)
                options.setdefault("prefer_precomputed_tracks", True)
                options.setdefault("require_generated_tracks", False)
                options.setdefault("assume_static_scene", False)
                if options["generate_static_tracks"] and not options["assume_static_scene"]:
                    raise ValueError("MUGEN static-track generation requires explicit assume_static_scene=true; use precomputed tracks for dynamic scenes")
                if supervision == "dense":
                    if options["generate_static_tracks"]:
                        raise ValueError("partial dense geometry must not generate sparse static tracks")
                    options["prefer_precomputed_tracks"] = False
                    if options.get("require_precomputed_tracks"):
                        raise ValueError("dense geometry does not consume precomputed sparse tracks")
            adapter = create_adapter(name, root=source["root"],
                                     split=source.get("split", "train"), **options)
            requested = source.get("sequences")
            sequences = adapter.list_sequences()
            frame_pool = {}
            if source.get("manifest"):
                manifest = json.loads(Path(source["manifest"]).read_text())
                if Path(manifest["root"]).resolve() != Path(source["root"]).resolve():
                    raise ValueError("data manifest root does not match its source")
                records = manifest["sequences"]
                unknown = set(records)-set(sequences)
                if unknown:
                    raise ValueError(f"manifest has unknown sequences: {sorted(unknown)[:5]}")
                sequences = sorted(records)
                for sequence, record in records.items():
                    if "valid_frames" in record:
                        frames = record["valid_frames"]
                        if not isinstance(frames, list) or not frames or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in frames) or frames != sorted(set(frames)):
                            raise ValueError("manifest valid_frames must be sorted unique nonnegative indices")
                        if frames[-1] >= adapter.get_num_frames(sequence):
                            raise ValueError("manifest frame exceeds adapter sequence")
                        frame_pool[sequence] = frames
            if requested is not None:
                if not isinstance(requested, list) or not requested:
                    raise ValueError("sequences must be a nonempty whitelist")
                absent = set(requested)-set(sequences)
                if absent:
                    raise ValueError(f"{name}: unknown requested sequences {sorted(absent)}")
                sequences = list(requested)
            pool = []
            for sequence in sequences:
                length = len(frame_pool[sequence]) if sequence in frame_pool else adapter.get_num_frames(sequence)
                minimum_times = min(minimum_layout_times, length) if sampling in ("coverage", "balanced_coverage") and name not in _NATIVE_MULTICAMERA else minimum_layout_times
                if length < 1+(minimum_times-1)*min(strides):
                    continue
                if name in _NATIVE_MULTICAMERA:
                    info = adapter.get_sequence_info(sequence)
                    available = info.get("num_cameras", len(info.get("cameras", [])))
                    if available < minimum_cameras:
                        continue
                pool.append((sequence, length))
            if not pool:
                raise ValueError(f"{name}: no sequence can supply {cameras} cameras and {source_times} unique times")
            self.sources.append(dict(name=name, adapter=adapter, cameras=cameras, num_times=source_times,
                                     strides=strides, pool=pool, frame_pool=frame_pool, dense_kind=kind,
                                     display_name=source.get("name", name), view_mode=view_mode,
                                     camera_choices=counts, layouts=layouts,
                                     minimum_times=minimum_layout_times))
            self.weights.append(weight)
            for sequence, length in pool:
                T = min(minimum_layout_times, length) if name not in _NATIVE_MULTICAMERA else minimum_layout_times
                windows = (math.ceil((length-1)/(T-1)) if T > 1 and kind in ("replica_flow", "vkitti_scene_flow")
                           else math.ceil(length/T))
                self.coverage_size += windows
                self.coverage_ends.append(self.coverage_size)
                self.coverage_entries.append((len(self.sources)-1, sequence, length, T))
        if sampling == "coverage":
            self.epoch_size = max(self.epoch_size, self.coverage_size)
        if sampling == "balanced_coverage":
            from .sampling import coverage_permutation, source_quotas
            self.quotas = source_quotas(self.weights, self.epoch_size)
            self.quota_ends = []
            total = 0
            for si, source in enumerate(self.sources):
                total += self.quotas[si]
                self.quota_ends.append(total)
                entries, ends, size = [], [], 0
                for entry in self.coverage_entries:
                    if entry[0] == si:
                        # Recompute per-source counts rather than indexing the
                        # global coverage partition, which contains other pools.
                        _, sequence, length, T = entry
                        windows = math.ceil((length-1)/(T-1)) if T > 1 and source["dense_kind"] in ("replica_flow", "vkitti_scene_flow") else math.ceil(length/T)
                        size += windows
                        entries.append((sequence, length, T))
                        ends.append(size)
                source.update(coverage_entries=entries, coverage_ends=ends, coverage_size=size,
                              coverage_permutation=coverage_permutation(size, self.seed + si * 100003))

    def __len__(self):
        return self.epoch_size

    def set_epoch(self, epoch):
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        rng = random.Random(self.seed+self.epoch*self.epoch_size+int(index))
        candidate = index
        invalid_indices = set()
        requested_times = None
        coverage_intervals = None
        for attempt in range(32):
            try:
                return self._get_item(candidate)
            except _InvalidDenseClip as error:
                if not self.skip_invalid:
                    raise
                if requested_times is None:
                    requested_times = self.sample_plan(index)["num_times"]
                    if self.sampling == "coverage":
                        starts = [0, *self.coverage_ends[:-1]]
                        coverage_intervals = [(start, end) for (start, end, entry) in
                            zip(starts, self.coverage_ends, self.coverage_entries)
                            if entry[3] == requested_times]
                invalid_indices.add(candidate)
                path = Path(self.invalid_log); path.parent.mkdir(parents=True, exist_ok=True)
                record = json.dumps(dict(epoch=self.epoch, requested_index=index, invalid_index=candidate,
                    reason=str(error), retry=attempt)) + "\n"
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                try:
                    os.write(fd, record.encode())
                finally:
                    os.close(fd)
                candidate = self._replacement_index(index, requested_times, invalid_indices,
                                                    rng, coverage_intervals)
        raise RuntimeError(f"32 dense clips with num_times={requested_times} had no valid depth; inspect {self.invalid_log}")

    def _replacement_index(self, index, num_times, invalid_indices, rng, coverage_intervals):
        if coverage_intervals is not None:
            ends, total = [], 0
            for start, end in coverage_intervals:
                total += end-start
                ends.append(total)
            for _ in range(min(128, total)):
                offset = rng.randrange(total)
                interval = bisect_right(ends, offset)
                previous = ends[interval-1] if interval else 0
                candidate = coverage_intervals[interval][0]+offset-previous
                if candidate not in invalid_indices:
                    return candidate
            for start, end in coverage_intervals:
                for candidate in range(start, min(end, start+len(invalid_indices)+1)):
                    if candidate not in invalid_indices:
                        return candidate
        else:
            if self.sampling == "balanced_coverage":
                si = bisect_right(self.quota_ends, index)
                start = self.quota_ends[si-1] if si else 0
                stop = self.quota_ends[si]
            else:
                start, stop = 0, len(self)
            for _ in range(min(128, stop-start)):
                candidate = rng.randrange(start, stop)
                if candidate not in invalid_indices and self.sample_plan(candidate)["num_times"] == num_times:
                    return candidate
            offset = rng.randrange(start, stop)
            for candidate in chain(range(offset, stop), range(start, offset)):
                if candidate not in invalid_indices and self.sample_plan(candidate)["num_times"] == num_times:
                    return candidate
        raise RuntimeError(f"No replacement dense clip with num_times={num_times} for index {index}; inspect {self.invalid_log}")

    def sample_plan(self, index):
        """Inspect sampling without loading RGB or GT; used by curriculum audits."""
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sample_seed = self.seed+self.epoch*self.epoch_size+int(index)
        rng = random.Random(sample_seed)
        requested_cameras = None
        if self.sampling == "balanced_coverage":
            si = bisect_right(self.quota_ends, index)
            source = self.sources[si]
            previous = self.quota_ends[si-1] if si else 0
            ordinal = self.epoch*self.quotas[si]+index-previous
            a, b = source["coverage_permutation"]
            window = (a*ordinal+b) % source["coverage_size"]
            entry = bisect_right(source["coverage_ends"], window)
            sequence, length, minimum_times = source["coverage_entries"][entry]
            layouts = source["layouts"] or [dict(cameras=source["cameras"], num_times=source["num_times"], weight=1.)]
            available = source["cameras"]
            if source["name"] in _NATIVE_MULTICAMERA:
                info = source["adapter"].get_sequence_info(sequence)
                available = info.get("num_cameras", len(info.get("cameras", [])))
            eligible = [v for v in layouts if v["cameras"] <= available and 1+(v["num_times"]-1)*min(source["strides"]) <= length]
            if not eligible:
                # Retain admitted short monocular sequences, as legacy coverage
                # does. Never repeat frames to manufacture the requested T.
                if source["name"] in _NATIVE_MULTICAMERA:
                    raise ValueError("native multiview sequence has no eligible layout")
                eligible = [dict(cameras=source["cameras"], num_times=min(length, minimum_times), weight=1.)]
            layout = rng.choices(eligible, weights=[v["weight"] for v in eligible], k=1)[0]
            requested_cameras, num_times = layout["cameras"], layout["num_times"]
            stride = rng.choice([v for v in source["strides"] if 1+(num_times-1)*v <= length])
            previous_window = source["coverage_ends"][entry-1] if entry else 0
            window_stride = minimum_times-1 if minimum_times > 1 and source["dense_kind"] in ("replica_flow", "vkitti_scene_flow") else minimum_times
            start = min((window-previous_window)*window_stride, length-1-(num_times-1)*stride)
            times = [start+t*stride for t in range(num_times)]
        elif self.sampling == "coverage" and index < self.coverage_size:
            entry = bisect_right(self.coverage_ends, index)
            si, sequence, length, num_times = self.coverage_entries[entry]
            source = self.sources[si]
            previous = self.coverage_ends[entry-1] if entry else 0
            window_stride = num_times-1 if num_times > 1 and source["dense_kind"] in ("replica_flow", "vkitti_scene_flow") else num_times
            start = min((index-previous)*window_stride, length-num_times)
            times = list(range(start, start+num_times))
        else:
            source = rng.choices(self.sources, weights=self.weights, k=1)[0]
            sequence, length = rng.choice(source["pool"])
            num_times = min(source["num_times"], length) if self.sampling == "coverage" and source["name"] not in _NATIVE_MULTICAMERA else source["num_times"]
            strides = [v for v in source["strides"] if 1+(num_times-1)*v <= length]
            if not strides:
                raise ValueError(f"{source['name']}/{sequence}: configured strides cannot fit {num_times} times")
            stride = rng.choice(strides)
            start = rng.randrange(length-(num_times-1)*stride)
            times = [start+t*stride for t in range(num_times)]
        if sequence in source["frame_pool"]:
            times = [source["frame_pool"][sequence][i] for i in times]
        requested_cameras = source["cameras"] if requested_cameras is None else requested_cameras
        return dict(source=source, sequence=sequence, times=times, cameras=requested_cameras,
                    num_times=num_times, sample_seed=sample_seed, rng=rng, length=length)

    def _get_item(self, index):
        from .bridge import prepare_clip
        plan = self.sample_plan(index)
        source, sequence, times = plan["source"], plan["sequence"], plan["times"]
        requested_cameras, num_times = plan["cameras"], plan["num_times"]
        sample_seed, rng, length = plan["sample_seed"], plan["rng"], plan["length"]
        static_info = None
        if source["view_mode"] == "static_multiview":
            from .static_views import select_static_views
            admitted = source["frame_pool"].get(sequence, range(length))
            frame_indices, static_info = select_static_views(source["adapter"], sequence, admitted,
                times[0], source["camera_choices"], rng)
            requested_cameras = len(frame_indices)
        else:
            frame_indices = times*requested_cameras
        try:
            if source["name"] in ("kubric_multicam", "syn4d"):
                clip = source["adapter"].load_clip(
                    sequence, frame_indices, layout_num_cameras=requested_cameras,
                    layout_num_times=num_times, sampling_seed=sample_seed)
            else:
                clip = source["adapter"].load_clip(sequence, frame_indices)
            if static_info is not None:
                from .static_views import as_static_multiview
                clip = as_static_multiview(clip, static_info)
            elif requested_cameras == 1:
                # Some older single-camera readers omit source frame labels.
                # The sampler knows them; retain gaps rather than arange(T).
                clip.metadata.setdefault("frame_indices", times)
            sample = prepare_clip(clip, image_size=self.image_size,
                                  num_queries=self.num_queries, seed=sample_seed,
                                  include_visibility=self.include_visibility,
                                  image_mode=self.image_mode,
                                  all_track_queries=self.supervision == "dense" and source["dense_kind"] == "pointodyssey_tracks")
            if self.supervision == "dense" and not sample["targets"]["depth_valid"].any():
                raise _InvalidDenseClip(f"{source['display_name']}/{sequence}, frames={times}: depth mask has no valid pixels")
            if self.supervision == "dense":
                if source["dense_kind"] == "kubric":
                    from .dense import attach_kubric_dense_recipe
                    attach_kubric_dense_recipe(clip, sample, image_mode=self.image_mode,
                                              include_visibility=self.include_visibility,
                                              background_policy=self.kubric_background_policy)
                elif source["dense_kind"] == "syn4d":
                    from .syn4d_dense import attach_syn4d_dense_recipe
                    attach_syn4d_dense_recipe(clip, sample, image_mode=self.image_mode,
                                             include_visibility=self.include_visibility)
                else:
                    from .partial_dense import attach_partial_dense_recipe
                    attach_partial_dense_recipe(clip, sample, kind=source["dense_kind"],
                                                image_mode=self.image_mode, adapter=source["adapter"])
            if len(sample["query_time_idx"]) != num_times:
                raise ValueError("adapter changed the requested number of unique target times")
            if sample["camera_id"].unique().numel() != requested_cameras:
                raise ValueError("adapter changed the requested number of cameras")
        except _InvalidDenseClip:
            raise
        except Exception as error:
            raise RuntimeError(f"Data preparation failed at sample {index}: {source['name']}/{sequence}, frames={times}: {error}") from error
        sample["dataset_name"] = source["display_name"]
        if "depth_filter" in clip.metadata:
            sample["depth_filter"] = clip.metadata["depth_filter"]
        if "native_track_checks" in clip.metadata:
            sample["native_tracks"] = {
                "trajectory_source": clip.metadata["trajectory_source"],
                "hidden_3d_supervision": clip.metadata["hidden_3d_supervision"],
                "source_checks": clip.metadata["native_track_checks"],
                "queries": len(sample["targets"]["sparse_queries"]),
            }
        if static_info is not None:
            sample["static_multiview"] = static_info
        return sample


def create_dataset(config):
    """Build a native dataset from a JSON/YAML file or configuration mapping."""
    if isinstance(config, (str, Path)):
        config = read_data_config(config)
    return MultiCameraAdapterDataset(**dict(config))


def _pad_axis(value, size, axis, fill=0):
    if value.shape[axis] == size:
        return value
    shape = list(value.shape)
    shape[axis] = size
    output = value.new_full(shape, fill)
    region = [slice(None)]*value.ndim
    region[axis] = slice(0, value.shape[axis])
    output[tuple(region)] = value
    return output


def collate_adapter_samples(batch):
    """Pad varying camera counts without padding or fabricating target times."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    times = len(batch[0]["query_time_idx"])
    if any(len(x["query_time_idx"]) != times for x in batch):
        raise ValueError("native batch must have the same unique target-time count")
    if any(x["images"].shape[1:] != batch[0]["images"].shape[1:] for x in batch):
        raise ValueError("native images must share prepared [3,H,W]")
    slots = max(len(x["images"]) for x in batch)
    cameras = max(x["camera_id"][x["observation_valid"]].unique().numel() for x in batch)
    output = {"clip_id": [x["clip_id"] for x in batch],
              "dataset_name": [x.get("dataset_name", "unknown") for x in batch]}
    if any("depth_filter" in x for x in batch):
        output["depth_filter"] = [x.get("depth_filter") for x in batch]
    if any("static_multiview" in x for x in batch):
        output["static_multiview"] = [x.get("static_multiview") for x in batch]
    if any("native_tracks" in x for x in batch):
        output["native_tracks"] = [x.get("native_tracks") for x in batch]
    if any("syn4d" in x for x in batch):
        output["syn4d"] = [x.get("syn4d") for x in batch]
    for key in ("images", "camera_id", "time_idx", "observation_valid"):
        fill = -1 if key in ("camera_id", "time_idx") else 0
        output[key] = torch.stack([_pad_axis(x[key], slots, 0, fill) for x in batch])
    output["query_time_idx"] = torch.stack([x["query_time_idx"] for x in batch])
    keys = set(batch[0]["targets"])
    if any(set(x["targets"]) != keys for x in batch):
        raise ValueError("native target fields differ; use one include_visibility setting")
    output["targets"] = {}
    observation_fields = {"depth", "depth_valid", "K", "c2w", "camera_valid"}
    observation_fields.update({"dense_segmentation", "dense_pair_delta", "dense_pair_valid", "dense_pair_time"})
    observation_fields.update({"dense_syn4d_face_map", "dense_syn4d_environment", "dense_motion_source_valid",
                               "dense_syn4d_content_hw"})
    visibility_fields = {"sparse_visibility", "sparse_visibility_valid"}
    for key in sorted(keys):
        values = []
        for sample in batch:
            value = sample["targets"][key]
            if key in observation_fields:
                value = _pad_axis(value, slots, 0, -1 if key == "dense_syn4d_face_map" else 0)
            elif key == "dense_syn4d_vertices":
                value = _pad_axis(value, max(x["targets"][key].shape[1] for x in batch), 1)
            elif key in ("dense_syn4d_faces", "dense_syn4d_face_valid"):
                value = _pad_axis(value, max(x["targets"][key].shape[0] for x in batch), 0)
            elif key in visibility_fields:
                value = _pad_axis(value, cameras, 2)
                value = _pad_axis(value, max(x["targets"]["sparse_queries"].shape[0] for x in batch), 0)
            elif key.startswith("sparse_"):
                fill = -1 if key == "sparse_track_id" else 0
                value = _pad_axis(value, max(x["targets"]["sparse_queries"].shape[0] for x in batch), 0, fill)
            elif key.startswith("dense_object_"):
                objects = max(x["targets"]["dense_object_valid"].shape[0] for x in batch)
                axis = 1 if key in ("dense_object_R", "dense_object_t") else 0
                value = _pad_axis(value, objects, axis)
            values.append(value)
        if any(x.shape != values[0].shape for x in values):
            raise ValueError(f"native target {key} shapes differ; use fixed num_queries and num_times")
        output["targets"][key] = torch.stack(values)
    return output
