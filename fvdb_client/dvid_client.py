"""
DVID Client for fetching label topology data.

Supports two methods for fetching voxel coordinates:
1. sparsevol - Returns RLE-encoded voxel coordinates directly
2. index + blocks - Get block list, then fetch individual blocks (more parallelizable)
"""

import struct
from dataclasses import dataclass
from typing import Iterator, Optional
from urllib.parse import urljoin

import numpy as np
import requests


@dataclass
class LabelInfo:
    """Information about a label from DVID."""
    label: int
    voxel_count: int
    block_count: int
    min_voxel: tuple[int, int, int]
    max_voxel: tuple[int, int, int]


@dataclass
class SparsevolConfig:
    """Configuration for sparsevol requests."""
    supervoxels: bool = False
    scale: int = 0
    compression: Optional[str] = None  # None, "lz4", or "gzip"
    # Bounding box filters (optional)
    minx: Optional[int] = None
    maxx: Optional[int] = None
    miny: Optional[int] = None
    maxy: Optional[int] = None
    minz: Optional[int] = None
    maxz: Optional[int] = None


class DVIDClient:
    """Client for fetching label data from a DVID server."""

    def __init__(self, server: str, uuid: str, instance: str, timeout: int = 300):
        """
        Initialize DVID client.

        Args:
            server: DVID server URL (e.g., "http://mydvidserver.org:9000")
            uuid: Version UUID or branch name (e.g., ":master")
            instance: Labelmap data instance name (e.g., "segmentation")
            timeout: Request timeout in seconds
        """
        self.server = server.rstrip("/")
        self.uuid = uuid
        self.instance = instance
        self.timeout = timeout

    def _base_url(self) -> str:
        """Get base URL for API requests."""
        return f"{self.server}/api/node/{self.uuid}/{self.instance}"

    def _build_params(self, config: Optional[SparsevolConfig] = None) -> dict:
        """Build query parameters from config."""
        params = {}
        if config:
            if config.supervoxels:
                params["supervoxels"] = "true"
            if config.scale > 0:
                params["scale"] = str(config.scale)
            if config.compression:
                params["compression"] = config.compression
            if config.minx is not None:
                params["minx"] = str(config.minx)
            if config.maxx is not None:
                params["maxx"] = str(config.maxx)
            if config.miny is not None:
                params["miny"] = str(config.miny)
            if config.maxy is not None:
                params["maxy"] = str(config.maxy)
            if config.minz is not None:
                params["minz"] = str(config.minz)
            if config.maxz is not None:
                params["maxz"] = str(config.maxz)
        return params

    def get_label_info(self, label: int, config: Optional[SparsevolConfig] = None) -> Optional[LabelInfo]:
        """
        Get information about a label (voxel count, bounding box).

        Args:
            label: Label ID to query
            config: Optional configuration (supervoxels, scale)

        Returns:
            LabelInfo if label exists, None if not found
        """
        url = f"{self._base_url()}/sparsevol-size/{label}"
        params = self._build_params(config)

        response = requests.get(url, params=params, timeout=self.timeout)

        if response.status_code == 404:
            return None
        response.raise_for_status()

        data = response.json()
        return LabelInfo(
            label=label,
            voxel_count=data.get("voxels", 0),
            block_count=data.get("numblocks", 0),
            min_voxel=tuple(data.get("minvoxel", [0, 0, 0])),
            max_voxel=tuple(data.get("maxvoxel", [0, 0, 0])),
        )

    def label_exists(self, label: int, config: Optional[SparsevolConfig] = None) -> bool:
        """
        Check if a label exists.

        Args:
            label: Label ID to check
            config: Optional configuration

        Returns:
            True if label exists, False otherwise
        """
        url = f"{self._base_url()}/sparsevol/{label}"
        params = self._build_params(config)

        response = requests.head(url, params=params, timeout=self.timeout)
        return response.status_code == 200

    def get_sparsevol_rles(self, label: int, config: Optional[SparsevolConfig] = None) -> Iterator[tuple[int, int, int, int]]:
        """
        Get sparse volume as streaming RLEs.

        Each RLE is (x, y, z, length) where the run starts at (x, y, z) and
        extends 'length' voxels along the X axis.

        Args:
            label: Label ID to fetch
            config: Optional configuration

        Yields:
            Tuples of (x, y, z, length) for each RLE span
        """
        url = f"{self._base_url()}/sparsevol/{label}"
        params = self._build_params(config)
        params["format"] = "srles"

        response = requests.get(url, params=params, timeout=self.timeout, stream=True)

        if response.status_code == 404:
            return
        response.raise_for_status()

        # Streaming RLEs: each span is 4 int32 values (16 bytes)
        buffer = b""
        for chunk in response.iter_content(chunk_size=4096):
            buffer += chunk
            while len(buffer) >= 16:
                x, y, z, length = struct.unpack("<iiii", buffer[:16])
                buffer = buffer[16:]
                yield (x, y, z, length)

    def get_sparsevol_coords(self, label: int, config: Optional[SparsevolConfig] = None) -> np.ndarray:
        """
        Get all voxel coordinates for a label.

        This expands RLEs into individual coordinates.
        Warning: Can use significant memory for large labels!

        Args:
            label: Label ID to fetch
            config: Optional configuration

        Returns:
            numpy array of shape (N, 3) with voxel coordinates
        """
        coords = []
        for x, y, z, length in self.get_sparsevol_rles(label, config):
            for dx in range(length):
                coords.append([x + dx, y, z])

        if not coords:
            return np.zeros((0, 3), dtype=np.int32)
        return np.array(coords, dtype=np.int32)

    def get_sparsevol_coords_batched(self, label: int, config: Optional[SparsevolConfig] = None,
                                      batch_size: int = 100000) -> Iterator[np.ndarray]:
        """
        Get voxel coordinates in batches (memory efficient for large labels).

        Args:
            label: Label ID to fetch
            config: Optional configuration
            batch_size: Maximum coordinates per batch

        Yields:
            numpy arrays of shape (<=batch_size, 3) with voxel coordinates
        """
        coords = []
        for x, y, z, length in self.get_sparsevol_rles(label, config):
            for dx in range(length):
                coords.append([x + dx, y, z])
                if len(coords) >= batch_size:
                    yield np.array(coords, dtype=np.int32)
                    coords = []

        if coords:
            yield np.array(coords, dtype=np.int32)

    def get_coarse_sparsevol(self, label: int, config: Optional[SparsevolConfig] = None) -> list[tuple[int, int, int]]:
        """
        Get block coordinates that contain the label.

        This uses sparsevol-coarse which returns block coordinates (not voxel coordinates).

        Args:
            label: Label ID to fetch
            config: Optional configuration

        Returns:
            List of (bx, by, bz) block coordinates
        """
        url = f"{self._base_url()}/sparsevol-coarse/{label}"
        params = self._build_params(config)

        response = requests.get(url, params=params, timeout=self.timeout)

        if response.status_code == 404:
            return []
        response.raise_for_status()

        data = response.content
        if len(data) < 8:
            return []

        # Parse header
        # byte 0: set to 0
        # byte 1: num dimensions (3)
        # byte 2: dimension of run (0 = X)
        # byte 3: reserved
        # uint32: # blocks (TODO, 0 for now)
        # uint32: # spans
        offset = 0
        offset += 4  # skip first 4 bytes
        num_blocks = struct.unpack("<I", data[offset:offset+4])[0]
        offset += 4
        num_spans = struct.unpack("<I", data[offset:offset+4])[0]
        offset += 4

        blocks = []
        for _ in range(num_spans):
            if offset + 16 > len(data):
                break
            bx, by, bz, length = struct.unpack("<iiii", data[offset:offset+16])
            offset += 16
            # Expand the run
            for dbx in range(length):
                blocks.append((bx + dbx, by, bz))

        return blocks

    def get_instance_info(self) -> dict:
        """
        Get information about the labelmap instance.

        Returns:
            Dictionary with instance metadata
        """
        url = f"{self._base_url()}/info"
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_block_size(self) -> tuple[int, int, int]:
        """
        Get the block size for this labelmap instance.

        Returns:
            Tuple of (bx, by, bz) block dimensions
        """
        info = self.get_instance_info()
        block_size = info.get("Extended", {}).get("BlockSize", [64, 64, 64])
        return tuple(block_size)
