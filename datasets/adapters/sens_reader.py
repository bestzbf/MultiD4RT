"""ScanNet .sens file reader.

Based on ScanNet official reader.py:
https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image


class SensReader:
    """Read ScanNet .sens files and extract RGB images, depth maps, and camera parameters."""

    def __init__(self, sens_path: str | Path):
        self.sens_path = Path(sens_path)
        if not self.sens_path.exists():
            raise FileNotFoundError(f".sens file not found: {self.sens_path}")

        self.version = None
        self.color_width = None
        self.color_height = None
        self.depth_width = None
        self.depth_height = None
        self.depth_shift = None
        self.num_frames = None
        self.color_intrinsics = None
        self.depth_intrinsics = None
        self.poses = None  # camera-to-world poses [N, 4, 4]

        self._read_header()

    def _read_header(self):
        """Read .sens file header."""
        with open(self.sens_path, "rb") as f:
            self.version = struct.unpack("I", f.read(4))[0]
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = f.read(strlen).decode("utf-8")

            # Color camera parameters
            self.color_intrinsics = self._read_intrinsics(f)
            self.color_extrinsics = self._read_intrinsics(f)  # 4x4 extrinsics

            # Depth camera parameters
            self.depth_intrinsics = self._read_intrinsics(f)
            self.depth_extrinsics = self._read_intrinsics(f)  # 4x4 extrinsics

            # Compression types
            self.color_compression = struct.unpack("i", f.read(4))[0]  # 2=jpeg
            self.depth_compression = struct.unpack("i", f.read(4))[0]  # 1=zlib_ushort

            # Image dimensions
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]

            # Number of frames
            self.num_frames = struct.unpack("Q", f.read(8))[0]

    def _read_intrinsics(self, f: BinaryIO) -> np.ndarray:
        """Read 4x4 intrinsic matrix."""
        mat = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            mat[i // 4, i % 4] = struct.unpack("f", f.read(4))[0]
        return mat

    def _read_pose(self, f: BinaryIO) -> np.ndarray:
        """Read 4x4 pose matrix."""
        mat = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            mat[i // 4, i % 4] = struct.unpack("f", f.read(4))[0]
        return mat

    def extract_all(self, output_dir: str | Path, max_frames: int | None = None):
        """Extract all frames to output_dir/images and output_dir/depths.

        Args:
            output_dir: Output directory
            max_frames: Maximum number of frames to extract (None = all)
        """
        output_dir = Path(output_dir)
        images_dir = output_dir / "images"
        depths_dir = output_dir / "depths"
        images_dir.mkdir(parents=True, exist_ok=True)
        depths_dir.mkdir(parents=True, exist_ok=True)

        poses = []
        num_to_extract = min(self.num_frames, max_frames) if max_frames else self.num_frames

        with open(self.sens_path, "rb") as f:
            # Skip header
            self._skip_header(f)

            for frame_idx in range(num_to_extract):
                # Read camera-to-world pose (4x4)
                pose = self._read_pose(f)
                poses.append(pose)

                # Read timestamps
                timestamp_color = struct.unpack("Q", f.read(8))[0]
                timestamp_depth = struct.unpack("Q", f.read(8))[0]

                # Read color image
                color_size_bytes = struct.unpack("Q", f.read(8))[0]
                depth_size_bytes = struct.unpack("Q", f.read(8))[0]

                color_data = f.read(color_size_bytes)
                depth_data = f.read(depth_size_bytes)

                # Decompress and save color (JPEG)
                if self.color_compression == 2:  # JPEG
                    from io import BytesIO
                    color_img = Image.open(BytesIO(color_data))
                    color_img.save(images_dir / f"{frame_idx:06d}.jpg")
                else:
                    raise ValueError(f"Unsupported color compression: {self.color_compression}")

                # Decompress and save depth (zlib uint16)
                if self.depth_compression == 1:  # zlib_ushort
                    depth_decompressed = zlib.decompress(depth_data)
                    depth_img = np.frombuffer(depth_decompressed, dtype=np.uint16).reshape(
                        (self.depth_height, self.depth_width)
                    )
                    depth_pil = Image.fromarray(depth_img, mode="I;16")
                    depth_pil.save(depths_dir / f"{frame_idx:06d}.png")
                else:
                    raise ValueError(f"Unsupported depth compression: {self.depth_compression}")

        self.poses = np.stack(poses, axis=0)

        # Save metadata
        metadata = {
            "intrinsics": self.color_intrinsics[:3, :3].tolist(),
            "extrinsics": self.poses.tolist(),
            "depth_scale": 1000.0,  # ScanNet depth is in millimeters
            "width": self.color_width,
            "height": self.color_height,
        }

        import json
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return len(poses)

    def _skip_header(self, f: BinaryIO):
        """Skip to frame data section."""
        f.seek(0)
        # Version
        f.read(4)
        # Sensor name
        strlen = struct.unpack("Q", f.read(8))[0]
        f.read(strlen)
        # Color intrinsics + extrinsics (4x4 each)
        f.read(4 * 16 * 2)
        # Depth intrinsics + extrinsics (4x4 each)
        f.read(4 * 16 * 2)
        # Compression types (2 ints)
        f.read(4 * 2)
        # Color width + height
        f.read(4 * 2)
        # Depth width + height
        f.read(4 * 2)
        # Depth shift
        f.read(4)
        # Num frames
        f.read(8)
