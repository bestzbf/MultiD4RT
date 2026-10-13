"""Execute upstream scoring/normalization functions without visualization imports.

Upstream code remains in the user's official checkout. We verify its hashes,
then compile the original function AST unchanged; metric code is not rewritten.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import importlib.machinery
import json
import logging
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import torch

MVTRACKER_COMMIT = "ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compile_functions(path, names, namespace):
    """Load selected top-level functions with their original bodies/defaults."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise ValueError(f"Official functions missing in {path}: {names}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class OfficialMVTracker:
    def __init__(self, root):
        self.root = Path(root).resolve()
        manifest = json.loads(Path(__file__).with_name("sources.json").read_text())["mvtracker"]
        for relative, expected in manifest["sha256"].items():
            actual = sha256_file(self.root / relative)
            if actual != expected:
                raise ValueError(f"Official source changed: {relative}; expected {expected}, got {actual}")
        self.provenance = {"repository": "https://github.com/ethz-vlg/mvtracker",
                           "commit": MVTRACKER_COMMIT, "sha256": manifest["sha256"]}
        path = self.root / "mvtracker/evaluation/metrics.py"
        spec = importlib.util.spec_from_file_location("_multicamera4rc_mvtracker_metrics", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace = {"np": np, "torch": torch, "logging": logging,
                     "evaluate_predictions": module.evaluate_predictions}
        self.evaluate_3dpt = compile_functions(self.root / "mvtracker/evaluation/evaluator_3dpt.py",
            ["evaluate_3dpt"], namespace)["evaluate_3dpt"]
        geometry = {"torch": torch, "np": np}
        compile_functions(self.root / "mvtracker/utils/basic.py",
                          ["to_homogeneous", "from_homogeneous"], geometry)
        self.transform_scene = compile_functions(self.root / "mvtracker/datasets/utils.py",
                                                  ["transform_scene"], geometry)["transform_scene"]

    def normalization(self, family):
        # These constants are from the hash-verified official dataset loaders.
        from scipy.spatial.transform import Rotation
        if family == "panoptic-multiview":
            return 2.5, torch.from_numpy(Rotation.from_euler("x", -90, degrees=True).as_matrix()), torch.zeros(3)
        if family == "dexycb-multiview":
            rx = Rotation.from_euler("x", 220, degrees=True).as_matrix()
            ry = Rotation.from_euler("y", 3, degrees=True).as_matrix()
            rz = Rotation.from_euler("z", -30, degrees=True).as_matrix()
            return 6.0, torch.from_numpy(rz @ ry @ rx), torch.tensor([0., 0., .5])
        if family == "kubric-multiview":
            return 1.0, torch.eye(3), torch.zeros(3)
        raise ValueError(f"Unknown evaluation setting: {family}")

    def cached_labels(self, path, protocol):
        with np.load(path, allow_pickle=False) as cache:
            query = torch.from_numpy(cache["query_points"].copy())
            tracks = torch.from_numpy(cache["traj3d_world"].copy())
            visible = cache["visibility"].astype(bool).any(axis=0)
            indices = cache["track_indices"].copy()
        scale, rotation, translation = self.normalization(protocol.family)
        _, _, query, tracks, _ = self.transform_scene(scale, rotation, translation,
            query_points=query, traj3d_world=tracks)
        return {"query_points": query.numpy(), "tracks": tracks.numpy(),
                "visible": visible, "track_indices": indices, "track_upscaling_factor": 1 / scale}

    def score(self, labels, tracks, visible, protocol):
        tracks = np.asarray(tracks, dtype=np.float32)
        visible = np.asarray(visible)
        if tracks.shape != labels["tracks"].shape or visible.shape != labels["visible"].shape:
            raise ValueError("Predictions must cover every frozen frame and query in original order")
        if visible.dtype != np.bool_ or not np.isfinite(tracks).all():
            raise ValueError("Predictions need finite 3D coordinates and boolean any-view visibility")
        return self.evaluate_3dpt(labels["tracks"], labels["visible"], tracks, visible,
            evaluation_setting=protocol.family, track_upscaling_factor=labels["track_upscaling_factor"],
            query_points=labels["query_points"], prefix="3dpt", verbose=False, add_per_track_results=False)

    def dataset(self, protocol, root):
        # Import only the official dataset module, never the GPU predictor/viz stack.
        existing = sys.modules.get("mvtracker")
        if existing is not None and Path(existing.__file__).resolve().parent != self.root / "mvtracker":
            raise RuntimeError("Another MVTracker checkout is already imported")
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        importlib.import_module("mvtracker")
        # Upstream's aggregate datasets/__init__.py also imports the unrelated
        # single-view TAP-Vid reader and its Rerun UI dependencies. Give the
        # three original reader modules a package namespace without executing
        # that aggregate initializer; no reader/helper function is replaced.
        if "mvtracker.datasets" not in sys.modules:
            package = ModuleType("mvtracker.datasets")
            package.__path__ = [str(self.root / "mvtracker/datasets")]
            package.__file__ = str(self.root / "mvtracker/datasets/__init__.py")
            package.__package__ = "mvtracker.datasets"
            package.__spec__ = importlib.machinery.ModuleSpec("mvtracker.datasets", loader=None, is_package=True)
            sys.modules["mvtracker.datasets"] = package
        modules = {
            "panoptic-multiview": ("panoptic_studio_multiview_dataset", "PanopticStudioMultiViewDataset"),
            "dexycb-multiview": ("dexycb_multiview_dataset", "DexYCBMultiViewDataset"),
            "kubric-multiview": ("kubric_multiview_dataset", "KubricMultiViewDataset"),
        }
        module, name = modules[protocol.family]
        factory = getattr(importlib.import_module(f"mvtracker.datasets.{module}"), name)
        dataset = factory.from_name(protocol.official_name, str(root))
        if not dataset.use_cached_tracks or dataset.cache_name + ".npz" != protocol.cache_name:
            raise ValueError("Official loader configuration does not match the frozen query protocol")
        return dataset
