#!/usr/bin/env python3
"""Verify an fVDB IndexGrid exported from DVID."""
import sys
import os
import fvdb
import torch

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <file.nvdb>")
    sys.exit(1)

path = sys.argv[1]
file_size = os.path.getsize(path)
print(f"Loading: {path} ({file_size:,} bytes)")

grid, data, names = fvdb.GridBatch.from_nanovdb(path, device="cpu")

print(f"Grid names: {names}")
print(f"Grids in batch: {grid.grid_count}")

total_voxels = grid.total_voxels
print(f"Total active voxels: {total_voxels:,}")

if data is not None and data.jdata.numel() > 0:
    print(f"Voxel data shape: {data.jdata.shape}")
else:
    print("Voxel data: None (IndexGrid - topology only)")

if total_voxels == 0:
    print("\nWARNING: Grid has 0 active voxels.")
    print("The NanoVDB container is valid but the topology is empty.")
    print("The Go exporter may not be writing active voxel masks correctly.")
    os._exit(1)

# Get bounding box from grid properties (avoids materializing coordinates)
try:
    bbox = grid.bbox_at(0)  # Shape [2, 3] - [[min_i, min_j, min_k], [max_i, max_j, max_k]]
    bbox_min = bbox[0]
    bbox_max = bbox[1]
    extent = bbox_max - bbox_min + 1

    print(f"Bounding box min: ({bbox_min[0].item()}, {bbox_min[1].item()}, {bbox_min[2].item()})")
    print(f"Bounding box max: ({bbox_max[0].item()}, {bbox_max[1].item()}, {bbox_max[2].item()})")
    print(f"Extent: {extent[0].item()} x {extent[1].item()} x {extent[2].item()}")
    print(f"Fill ratio: {total_voxels / extent.prod().item():.6f}")
except Exception as e:
    print(f"Could not get bounding box: {e}")

print("\nVerification complete.")
os._exit(0)
