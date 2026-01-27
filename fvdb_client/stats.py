"""
Statistics computation and reporting for IndexGrid data.

Provides functions to compute and display statistics about label topology
including voxel counts, bounding boxes, and storage cost estimates.
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import fvdb
    FVDB_AVAILABLE = True
except ImportError:
    FVDB_AVAILABLE = False


@dataclass
class IndexGridStats:
    """Statistics for an IndexGrid."""
    # Basic counts
    voxel_count: int
    unique_voxel_count: int  # After deduplication

    # Bounding box (in voxel coordinates)
    min_coord: tuple[int, int, int]
    max_coord: tuple[int, int, int]

    # Dimensions
    extent: tuple[int, int, int]  # max - min + 1

    # Tree structure (if available from fVDB)
    leaf_count: Optional[int] = None
    internal_node_count: Optional[int] = None

    # Storage estimates
    file_size_bytes: Optional[int] = None
    estimated_memory_bytes: Optional[int] = None

    @property
    def bounding_box_volume(self) -> int:
        """Volume of the bounding box in voxels."""
        return self.extent[0] * self.extent[1] * self.extent[2]

    @property
    def fill_ratio(self) -> float:
        """Ratio of active voxels to bounding box volume."""
        if self.bounding_box_volume == 0:
            return 0.0
        return self.unique_voxel_count / self.bounding_box_volume

    @property
    def compression_ratio(self) -> Optional[float]:
        """Ratio of dense storage to actual IndexGrid storage."""
        if self.file_size_bytes is None or self.file_size_bytes == 0:
            return None
        # Dense storage would be 1 byte per voxel (just marking active)
        dense_size = self.bounding_box_volume
        return dense_size / self.file_size_bytes


def compute_stats(coords: np.ndarray, file_path: Optional[str] = None) -> IndexGridStats:
    """
    Compute statistics from voxel coordinates.

    Args:
        coords: numpy array of shape (N, 3) with voxel coordinates
        file_path: Optional path to saved .nvdb file for file size

    Returns:
        IndexGridStats with computed statistics
    """
    if len(coords) == 0:
        return IndexGridStats(
            voxel_count=0,
            unique_voxel_count=0,
            min_coord=(0, 0, 0),
            max_coord=(0, 0, 0),
            extent=(0, 0, 0),
        )

    # Remove duplicates
    unique_coords = np.unique(coords, axis=0)

    # Compute bounding box
    min_coord = tuple(unique_coords.min(axis=0).tolist())
    max_coord = tuple(unique_coords.max(axis=0).tolist())
    extent = tuple((max_coord[i] - min_coord[i] + 1) for i in range(3))

    # Get file size if available
    file_size = None
    if file_path and os.path.exists(file_path):
        file_size = os.path.getsize(file_path)

    # Estimate memory usage for IndexGrid
    # NanoVDB uses ~96 bytes per leaf node (8x8x8 voxels)
    # Plus overhead for internal nodes
    num_potential_leaves = (len(unique_coords) + 511) // 512  # Upper bound
    estimated_memory = num_potential_leaves * 96 + 1024  # Add header overhead

    return IndexGridStats(
        voxel_count=len(coords),
        unique_voxel_count=len(unique_coords),
        min_coord=min_coord,
        max_coord=max_coord,
        extent=extent,
        file_size_bytes=file_size,
        estimated_memory_bytes=estimated_memory,
    )


def compute_stats_from_grid(grid: "fvdb.GridBatch", file_path: Optional[str] = None) -> IndexGridStats:
    """
    Compute statistics from an fVDB GridBatch.

    Args:
        grid: fVDB GridBatch
        file_path: Optional path to saved .nvdb file for file size

    Returns:
        IndexGridStats with computed statistics
    """
    if not FVDB_AVAILABLE:
        raise ImportError("fVDB is not installed")

    # Get voxel coordinates
    ijk = grid.ijk[0].jdata  # First grid in batch
    coords = ijk.cpu().numpy()

    # Get total voxels
    total_voxels = grid.total_voxels

    # Compute bounding box
    if len(coords) == 0:
        min_coord = (0, 0, 0)
        max_coord = (0, 0, 0)
    else:
        min_coord = tuple(coords.min(axis=0).tolist())
        max_coord = tuple(coords.max(axis=0).tolist())

    extent = tuple((max_coord[i] - min_coord[i] + 1) for i in range(3))

    # Get file size if available
    file_size = None
    if file_path and os.path.exists(file_path):
        file_size = os.path.getsize(file_path)

    # Get tree structure info if available
    leaf_count = None
    internal_count = None
    try:
        # fVDB might expose these properties
        leaf_count = grid.num_voxels  # This is per-grid
    except AttributeError:
        pass

    return IndexGridStats(
        voxel_count=total_voxels,
        unique_voxel_count=total_voxels,  # fVDB already deduplicates
        min_coord=min_coord,
        max_coord=max_coord,
        extent=extent,
        leaf_count=leaf_count,
        internal_node_count=internal_count,
        file_size_bytes=file_size,
    )


def format_bytes(num_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def print_stats(stats: IndexGridStats, label: Optional[int] = None):
    """
    Print statistics in a formatted manner.

    Args:
        stats: IndexGridStats to display
        label: Optional label ID for display
    """
    print("=" * 60)
    if label is not None:
        print(f"Label {label} Statistics")
    else:
        print("IndexGrid Statistics")
    print("=" * 60)

    print(f"\nVoxel Counts:")
    print(f"  Total voxels:        {stats.voxel_count:,}")
    if stats.voxel_count != stats.unique_voxel_count:
        print(f"  Unique voxels:       {stats.unique_voxel_count:,}")
        print(f"  Duplicates removed:  {stats.voxel_count - stats.unique_voxel_count:,}")

    print(f"\nBounding Box:")
    print(f"  Min coordinate:      ({stats.min_coord[0]}, {stats.min_coord[1]}, {stats.min_coord[2]})")
    print(f"  Max coordinate:      ({stats.max_coord[0]}, {stats.max_coord[1]}, {stats.max_coord[2]})")
    print(f"  Extent (W x H x D):  {stats.extent[0]} x {stats.extent[1]} x {stats.extent[2]}")
    print(f"  Bounding box volume: {stats.bounding_box_volume:,} voxels")
    print(f"  Fill ratio:          {stats.fill_ratio:.4%}")

    if stats.leaf_count is not None:
        print(f"\nTree Structure:")
        print(f"  Leaf nodes:          {stats.leaf_count:,}")
        if stats.internal_node_count is not None:
            print(f"  Internal nodes:      {stats.internal_node_count:,}")

    print(f"\nStorage:")
    if stats.file_size_bytes is not None:
        print(f"  File size:           {format_bytes(stats.file_size_bytes)}")
        # Bytes per voxel
        bytes_per_voxel = stats.file_size_bytes / max(1, stats.unique_voxel_count)
        print(f"  Bytes per voxel:     {bytes_per_voxel:.2f}")

    if stats.estimated_memory_bytes is not None:
        print(f"  Est. memory:         {format_bytes(stats.estimated_memory_bytes)}")

    if stats.compression_ratio is not None:
        print(f"  Compression ratio:   {stats.compression_ratio:.2f}x vs dense")

    # Compare to alternative storage formats
    print(f"\nStorage Comparison (theoretical):")
    # Dense 3D array would be extent[0] * extent[1] * extent[2] bits
    dense_bits = stats.bounding_box_volume
    dense_bytes = (dense_bits + 7) // 8
    print(f"  Dense bitmap:        {format_bytes(dense_bytes)}")

    # RLE encoding: worst case is 16 bytes per voxel (x, y, z, length=1)
    # Best case for contiguous: 16 bytes per run
    rle_worst = stats.unique_voxel_count * 16
    print(f"  RLE (worst case):    {format_bytes(rle_worst)}")

    # Coordinate list: 12 bytes per voxel (3 x int32)
    coord_list = stats.unique_voxel_count * 12
    print(f"  Coord list (int32):  {format_bytes(coord_list)}")

    if stats.file_size_bytes:
        print(f"  IndexGrid (.nvdb):   {format_bytes(stats.file_size_bytes)}")

    print("=" * 60)
