# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

fVDB Client is a Python CLI tool and library for fetching label topology from DVID servers and creating fVDB IndexGrids for sparse voxel representation and visualization. Used in connectomics research.

## Build and Development Commands

```bash
# Install in development mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Install specific extras
pip install -e ".[fvdb]"   # For fVDB support
pip install -e ".[render]" # For static rendering (matplotlib)
```

**Prerequisites**: fVDB must be installed separately following https://github.com/openvdb/fvdb-core instructions. Requires Python 3.10+, Linux with NVIDIA GPU for visualization.

## CLI Entry Point

The `fvdb-client` command is defined in `pyproject.toml` as `fvdb_client.cli:main`.

Commands: `info`, `fetch`, `stats`, `view`, `render`

## Architecture

```
CLI (cli.py)
    └── Subcommands: info, fetch, stats, view, render
        ├── DVIDClient (dvid_client.py) - HTTP client for DVID servers
        ├── IndexGridBuilder (indexgrid.py) - Builds fVDB GridBatch from coordinates
        ├── Statistics (stats.py) - Computes IndexGridStats from grids
        └── Visualization (visualize.py) - Interactive viewer and static rendering
```

**Data Flow**: DVID Server → RLE-encoded data → DVIDClient (streams coords) → IndexGridBuilder → fVDB GridBatch → NanoVDB file (.nvdb) → Stats/Visualization

## Key Modules

- **`dvid_client.py`**: `DVIDClient` class fetches label data via HTTP. Key methods: `get_label_info()`, `get_sparsevol_rles()`, `get_sparsevol_coords()`, `get_coarse_sparsevol()`
- **`indexgrid.py`**: `IndexGridBuilder` accumulates coordinates and builds sparse grids. Uses PyTorch tensors and fVDB JaggedTensor. Functions: `save_indexgrid()`, `load_indexgrid()`, `coords_to_grid()`
- **`stats.py`**: `IndexGridStats` dataclass and `compute_stats()` / `compute_stats_from_grid()` functions
- **`visualize.py`**: `visualize_grid()` for interactive 3D viewer, `render_to_image()` for static output

## Code Patterns

- **Graceful degradation**: Functions check `FVDB_AVAILABLE` before using fVDB features
- **Device management**: Automatic fallback from CUDA to CPU
- **Memory efficiency**: Streaming RLE parsing for large labels
- **Builder pattern**: `IndexGridBuilder` for grid construction

## Important Constants

- Default DVID block size: 64×64×64 voxels
- Default viewer port: 8080
- Default color (RGB): (0.2, 0.6, 1.0)

## Public API

Import from `fvdb_client`:
- `DVIDClient`, `IndexGridBuilder`, `save_indexgrid`, `load_indexgrid`, `compute_stats`, `print_stats`

---

## fVDB API Reference

The `llm-research/` directory contains a snapshot of the fvdb-core repository for API reference. Use this to understand fVDB patterns when extending this codebase.

### Core Concepts

**Three coordinate systems**:
- **World space**: Continuous 3D coordinates where geometry exists
- **Voxel space**: Discrete integer grid coordinates (i, j, k)
- **Index space**: Linear indexing into internal sparse storage

**Key classes**:
- `Grid`: Single sparse voxel grid (topology only, no data)
- `GridBatch`: Collection of grids with different resolutions/origins
- `JaggedTensor`: Variable-length sequences with GPU support

### Grid Creation Patterns

```python
# From point clouds (most common)
grid = Grid.from_points(points, voxel_size=0.01, origin=[0.0, 0.0, 0.0])

# From voxel coordinates (used in this project)
ijk_coords = torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int64)
grid = Grid.from_ijk(ijk_coords, voxel_size=1.0, origin=0.0)

# Batch of grids
point_batches = JaggedTensor([points_1, points_2])
grid_batch = GridBatch.from_points(point_batches, voxel_sizes=[0.01, 0.015])
```

### Data Association Pattern

Grids store ONLY structure; data stored separately as tensors:

```python
grid = Grid.from_points(points, voxel_size=0.01)
num_voxels = grid.num_voxels

# Create feature data separately
features = torch.randn(num_voxels, 16)  # [num_voxels, channels]

# For GridBatch
features_flat = torch.ones(grid_batch.total_voxels, 16)
features_jagged = grid_batch.jagged_like(features_flat)
```

### Coordinate Conversions

```python
# World to voxel
voxel_coords = grid.world_to_voxel(world_points)

# Voxel to world
world_coords = grid.voxel_to_world(voxel_coords)

# Voxel to index (for data access)
indices = grid.ijk_to_index(voxel_coords)  # -1 for inactive voxels

# Get all active voxel coordinates
all_ijk = grid.ijk  # Shape [num_voxels, 3]
```

### Safe Index Access

Always check for -1 (inactive voxel marker):

```python
indices = grid.ijk_to_index(query_ijk)
valid_mask = indices >= 0
valid_data = features[indices[valid_mask]]
```

### Type-Safe Operations

Use `fvdb.*` functions instead of `torch.*` to preserve JaggedTensor types:

```python
import fvdb

# Correct: preserves JaggedTensor type
result = fvdb.relu(jagged_tensor)
result = fvdb.add(jagged_tensor, 1.0)

# Available: relu, sigmoid, tanh, exp, log, sqrt, sum, mean, etc.
```

### NanoVDB Serialization

```python
# Save
grid.save_nanovdb("grid.nvdb", voxel_data=features)

# Load
loaded_grid, loaded_data, name = Grid.from_nanovdb("grid.nvdb")
```

### Key Reference Files in llm-research/

- `fvdb-core/fvdb/grid.py` - Grid class implementation
- `fvdb-core/fvdb/grid_batch.py` - GridBatch class
- `fvdb-core/fvdb/jagged_tensor.py` - JaggedTensor class
- `fvdb-core/examples/grid_building.py` - Grid creation patterns
- `fvdb-core/examples/sample_trilinear.py` - Interpolation patterns
