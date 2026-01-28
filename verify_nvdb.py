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
print(f"Total active voxels: {grid.total_voxels:,}")

if data is not None:
    print(f"Voxel data shape: {data.jdata.shape}")
else:
    print("Voxel data: None")

if grid.total_voxels == 0:
    print("\nWARNING: Grid has 0 active voxels.")
    print("The NanoVDB container is valid but the topology is empty.")
    print("The Go exporter may not be writing active voxel masks correctly.")
    sys.exit(1)

ijk = grid.ijk
coords = ijk.jdata

mins = coords.min(dim=0).values
maxs = coords.max(dim=0).values
extent = maxs - mins + 1

print(f"Bounding box min: ({mins[0]}, {mins[1]}, {mins[2]})")
print(f"Bounding box max: ({maxs[0]}, {maxs[1]}, {maxs[2]})")
print(f"Extent: {extent[0]} x {extent[1]} x {extent[2]}")
print(f"Fill ratio: {grid.total_voxels / extent.prod().item():.6f}")
