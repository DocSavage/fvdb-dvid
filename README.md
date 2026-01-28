# fVDB Client

A Python tool to fetch label topology from [DVID](https://github.com/janelia-flyem/dvid) servers and create [fVDB](https://github.com/openvdb/fvdb-core) IndexGrids for efficient sparse voxel representation and visualization.

## Features

- **Fetch label data from DVID**: Connect to any DVID server and download label voxel topology
- **Create IndexGrids**: Convert label topology to fVDB's efficient sparse representation
- **Statistics reporting**: Understand voxel counts, bounding boxes, and storage costs
- **Interactive 3D visualization**: View labels using fVDB's web-based viewer with trackball rotation
- **Static rendering**: Generate images for documentation or animation

## Requirements

- Python 3.10+
- Linux with NVIDIA GPU (for visualization and GPU-accelerated operations)
- [fVDB](https://github.com/openvdb/fvdb-core) with PyTorch and CUDA support

## Installation

### 1. Install fVDB (required for full functionality)

Follow the [fVDB installation instructions](https://github.com/openvdb/fvdb-core). Typically:

```bash
# Create conda environment
conda create -n fvdb python=3.12
conda activate fvdb

# Install PyTorch with CUDA
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

# Install fVDB
pip install fvdb
```

### 2. Install fvdb-client

```bash
# Clone the repository
git clone <repo-url> fvdb-dvid
cd fvdb-dvid

# Install in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Usage

### Quick Info (no download)

Get information about a label without downloading the full data:

```bash
fvdb-client info \
    --server http://dvid.example.org:8000 \
    --uuid abc123 \
    --instance segmentation \
    --label 12345
```

### Fetch and Create IndexGrid

Download label topology and save as NanoVDB file:

```bash
fvdb-client fetch \
    --server http://dvid.example.org:8000 \
    --uuid abc123 \
    --instance segmentation \
    --label 12345 \
    --output label_12345.nvdb
```

Options:
- `--supervoxels`: Interpret label as supervoxel ID
- `--scale N`: Use scale level N (0 = highest resolution)
- `--compress`: Use Blosc compression for smaller files
- `--device cuda|cpu`: Device for grid building

### View Statistics

Analyze a saved IndexGrid:

```bash
fvdb-client stats label_12345.nvdb --label 12345
```

Output includes:
- Voxel counts
- Bounding box dimensions
- Fill ratio
- Storage size
- Comparison to alternative formats (dense, RLE, coordinate list)

### Interactive 3D Visualization

View the label topology with mouse-controllable rotation:

```bash
fvdb-client view label_12345.nvdb
```

This starts a web-based viewer at `http://127.0.0.1:8080` with:
- Trackball rotation (click and drag)
- Zoom (scroll)
- Pan (shift + drag)

Options:
- `--port N`: Use different port
- `--color R,G,B`: Set voxel color (0-1 range)
- `--point-size N`: Set point size in pixels

### Static Rendering

Render to an image file (requires matplotlib):

```bash
# Single image
fvdb-client render label_12345.nvdb --output render.png

# Rotation animation frames
fvdb-client render label_12345.nvdb --output frames/ --rotate --frames 72
```

Create a GIF from frames:
```bash
convert -delay 10 -loop 0 frames/frame_*.png animation.gif
```

## Python API

Use as a library in your own scripts:

```python
from fvdb_client import DVIDClient, IndexGridBuilder, compute_stats, print_stats

# Connect to DVID
client = DVIDClient(
    server="http://dvid.example.org:8000",
    uuid="abc123",
    instance="segmentation"
)

# Get label info
info = client.get_label_info(12345)
print(f"Label has {info.voxel_count:,} voxels")

# Build IndexGrid
builder = IndexGridBuilder(name="label_12345")
builder.add_coords_from_rles(client.get_sparsevol_rles(12345))
grid = builder.build()

# Save to file
from fvdb_client import save_indexgrid
save_indexgrid(grid, "label_12345.nvdb")

# Compute and print statistics
from fvdb_client.stats import compute_stats_from_grid
stats = compute_stats_from_grid(grid, "label_12345.nvdb")
print_stats(stats, label=12345)

# Visualize
from fvdb_client.visualize import visualize_grid
visualize_grid(grid, name="Label 12345")
```

## File Formats

### NanoVDB (.nvdb)

The output files use the [NanoVDB](https://github.com/AcademySoftwareFoundation/openvdb/tree/feature/nanovdb) format, which is:
- A compact, GPU-friendly sparse voxel format
- Compatible with OpenVDB ecosystem
- Loadable by fVDB, NanoVDB viewers, and other OpenVDB tools

### DVID Sparse Volumes

This tool uses DVID's `/sparsevol/<label>` endpoint with streaming RLEs format for efficient data transfer.

## Storage Cost Analysis

IndexGrids provide excellent compression for sparse label data:

| Format | Storage for 1M voxels (typical) |
|--------|--------------------------------|
| Dense 3D bitmap | ~125 MB (1000³ bounding box) |
| Coordinate list | ~12 MB (3 × int32 per voxel) |
| RLE worst case | ~16 MB (4 × int32 per span) |
| IndexGrid | ~1-5 MB (depends on structure) |

The actual compression depends on the spatial structure of the label. Compact, contiguous labels compress better than scattered ones.

## Troubleshooting

### fVDB not found

```
ImportError: fVDB is not installed
```

Install fVDB following the instructions at https://github.com/openvdb/fvdb-core

### CUDA not available

The tool will fall back to CPU if CUDA is not available, but visualization features require GPU support.

### Viewer not loading

- Check that Vulkan drivers are installed
- Try a different port: `--port 8081`
- Check firewall settings if accessing remotely

## License

BSD-3-Clause (same as DVID)

## See Also

- [DVID](https://github.com/janelia-flyem/dvid) - Distributed, Versioned, Image-oriented Dataservice
- [fVDB](https://github.com/openvdb/fvdb-core) - GPU-accelerated sparse voxel operations
- [NanoVDB](https://github.com/AcademySoftwareFoundation/openvdb/tree/feature/nanovdb) - Compact VDB format
