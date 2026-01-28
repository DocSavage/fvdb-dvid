# fvdb-dvid

A Python tool to fetch label topology from [DVID](https://github.com/janelia-flyem/dvid) servers and create [fVDB](https://github.com/openvdb/fvdb-core) IndexGrids for efficient sparse voxel representation and visualization.

## Features

- **Fetch label data from DVID**: Connect to any DVID server and download label voxel topology
- **Create IndexGrids**: Convert label topology to fVDB's efficient sparse representation
- **Load existing .nvdb files**: Work with previously saved IndexGrids without DVID access
- **Statistics reporting**: Understand voxel counts, bounding boxes, and storage costs
- **Interactive 3D visualization**: Web-based viewer with trackball rotation (remote-accessible)
- **Static rendering**: Generate images via matplotlib (no GPU display required)

## Requirements

- Linux with NVIDIA GPU (Ampere or newer, compute capability 8.0+)
- NVIDIA driver 550.0 or later
- Python 3.10-3.13
- Vulkan (for interactive viewer only)

## Installation

### 1. Create conda environment

```bash
conda create -n fvdb python=3.12
conda activate fvdb
```

### 2. Install fVDB and PyTorch

**Important:** The PyPI package `fvdb` is an unrelated project. The correct package is `fvdb-core` from NVIDIA's package index.

```bash
pip install fvdb-core torch==2.8.0 \
  --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple" \
  --extra-index-url="https://download.pytorch.org/whl/cu128"
```

Verify:

```bash
python -c "import fvdb; import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
# Expected: PyTorch: 2.8.0+cu128 CUDA: True
```

### 3. Install nanovdb-editor (required for interactive viewer)

The interactive 3D viewer depends on [nanovdb-editor](https://github.com/openvdb/nanovdb-editor), which must be built from source. This requires Vulkan and a C++ toolchain.

```bash
git clone https://github.com/openvdb/nanovdb-editor.git
cd nanovdb-editor
./build.sh -p
```

If you only need statistics and static rendering, this step can be skipped.

### 4. Install fvdb-dvid

```bash
git clone <repo-url> fvdb-dvid
cd fvdb-dvid
pip install -e .
```

## Usage

### Working with existing .nvdb files

```bash
# View statistics
fvdb-client stats label_12345.nvdb

# Interactive 3D visualization (requires nanovdb-editor)
fvdb-client view label_12345.nvdb

# Render to image (requires matplotlib, no Vulkan needed)
fvdb-client render label_12345.nvdb --output render.png
```

### Remote viewing

The interactive viewer serves a web UI that can be accessed remotely via SSH port forwarding:

```bash
# On the cluster:
fvdb-client view label.nvdb --ip 0.0.0.0 --port 8080 --no-browser

# From your local machine:
ssh -L 8080:localhost:8080 user@cluster

# Then open http://localhost:8080 in your local browser
```

### Fetching from DVID

Get information about a label (no download):

```bash
fvdb-client info \
    --server http://dvid.example.org:8000 \
    --uuid abc123 \
    --instance segmentation \
    --label 12345
```

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

### Visualization options

Interactive viewer (requires nanovdb-editor + Vulkan):
```bash
fvdb-client view label.nvdb --port 8080 --color 0.2,0.6,1.0 --point-size 3.0
```

Static rendering (requires matplotlib only):
```bash
# Single image
fvdb-client render label.nvdb --output render.png

# Rotation animation frames
fvdb-client render label.nvdb --output frames/ --rotate --frames 72

# Create GIF from frames
convert -delay 10 -loop 0 frames/frame_*.png animation.gif
```

## Python API

```python
from fvdb_client import DVIDClient, IndexGridBuilder, save_indexgrid, load_indexgrid

# Load existing .nvdb file
grid = load_indexgrid("label_12345.nvdb", device="cuda")

# Or fetch from DVID and build
client = DVIDClient(
    server="http://dvid.example.org:8000",
    uuid="abc123",
    instance="segmentation"
)

info = client.get_label_info(12345)
print(f"Label has {info.voxel_count:,} voxels")

builder = IndexGridBuilder(name="label_12345")
builder.add_coords_from_rles(client.get_sparsevol_rles(12345))
grid = builder.build()

save_indexgrid(grid, "label_12345.nvdb")

# Visualize (requires nanovdb-editor)
from fvdb_client.visualize import visualize_grid
visualize_grid(grid, name="Label 12345")
```

## File Formats

### NanoVDB (.nvdb)

Output files use [NanoVDB](https://github.com/AcademySoftwareFoundation/openvdb/tree/feature/nanovdb) format:
- Compact, GPU-friendly sparse voxel format
- Compatible with OpenVDB ecosystem
- Loadable by fVDB, NanoVDB viewers, and other OpenVDB tools

### DVID Sparse Volumes

This tool uses DVID's `/sparsevol/<label>` endpoint with streaming RLEs format for efficient data transfer.

## Troubleshooting

### Wrong fvdb package installed

If you see errors about MPNet or HuggingFace when importing fvdb:
```bash
pip uninstall fvdb
```
You installed the wrong package. Install `fvdb-core` from NVIDIA's index as shown above.

### CUDA not available

Check that:
1. NVIDIA driver is installed: `nvidia-smi`
2. PyTorch has CUDA: `python -c "import torch; print(torch.version.cuda)"`

If `torch.version.cuda` is `None`, reinstall PyTorch with CUDA support.

### Viewer fails with "No module named 'nanovdb_editor'"

Build and install [nanovdb-editor](https://github.com/openvdb/nanovdb-editor) from source (see installation step 3).

### Viewer fails with Vulkan error

The interactive viewer requires Vulkan. Check with `vulkaninfo`. If Vulkan is unavailable, use `fvdb-client render` for static images instead.

## License

BSD-3-Clause

## See Also

- [DVID](https://github.com/janelia-flyem/dvid) - Distributed, Versioned, Image-oriented Dataservice
- [fVDB](https://github.com/openvdb/fvdb-core) - GPU-accelerated sparse voxel operations
- [fVDB Installation Docs](https://fvdb.ai/installation.html) - Official fVDB installation guide
- [nanovdb-editor](https://github.com/openvdb/nanovdb-editor) - NanoVDB viewer (required for interactive visualization)
