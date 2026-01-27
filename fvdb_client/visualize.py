"""
Visualization module using fVDB's built-in viewer.

Provides functions to visualize IndexGrid data using fVDB's web-based
viewer with interactive trackball rotation.
"""

from typing import Optional, Tuple
import numpy as np

try:
    import torch
    import fvdb
    import fvdb.viz as fviz
    FVDB_AVAILABLE = True
except ImportError:
    FVDB_AVAILABLE = False


def check_fvdb():
    """Check if fVDB is available."""
    if not FVDB_AVAILABLE:
        raise ImportError(
            "fVDB is not installed. Please install fVDB following instructions at: "
            "https://github.com/openvdb/fvdb-core"
        )


def visualize_grid(
    grid: "fvdb.GridBatch",
    name: str = "Label Topology",
    color: Tuple[float, float, float] = (0.2, 0.6, 1.0),
    point_size: float = 3.0,
    ip_address: str = "127.0.0.1",
    port: int = 8080,
    auto_open: bool = True,
):
    """
    Visualize an fVDB GridBatch using the built-in web viewer.

    This opens a web-based 3D viewer with interactive trackball rotation.
    The viewer runs until interrupted with Ctrl+C.

    Args:
        grid: fVDB GridBatch to visualize
        name: Name for the scene
        color: RGB color tuple (0-1 range) for the voxels
        point_size: Size of points in pixels
        ip_address: IP address for the viewer server
        port: Port for the viewer server
        auto_open: Whether to automatically open browser
    """
    check_fvdb()

    # Initialize the viewer
    print(f"Starting fVDB viewer on http://{ip_address}:{port}")
    fviz.init(ip_address=ip_address, port=port)

    # Create scene
    scene = fviz.Scene(name)

    # Get voxel coordinates
    ijk = grid.ijk[0].jdata  # First grid in batch
    points = ijk.float()  # Convert to float for rendering

    # Create colors tensor (same color for all points)
    colors = torch.tensor([color], dtype=torch.float32)
    colors = colors.expand(points.shape[0], 3).contiguous()

    if points.is_cuda:
        colors = colors.cuda()

    # Add point cloud to scene
    scene.add_point_cloud(
        name="voxels",
        points=points,
        colors=colors,
        point_size=point_size,
    )

    # Show viewer
    if auto_open:
        fviz.show()

    print("\nViewer is running. Use mouse to rotate (trackball), scroll to zoom.")
    print("Press Ctrl+C to exit.\n")

    # Keep running until interrupted
    fviz.wait_for_interrupt()


def visualize_coords(
    coords: np.ndarray,
    name: str = "Label Topology",
    color: Tuple[float, float, float] = (0.2, 0.6, 1.0),
    point_size: float = 3.0,
    ip_address: str = "127.0.0.1",
    port: int = 8080,
    device: str = "cuda",
    auto_open: bool = True,
):
    """
    Visualize voxel coordinates using the fVDB web viewer.

    Args:
        coords: numpy array of shape (N, 3) with voxel coordinates
        name: Name for the scene
        color: RGB color tuple (0-1 range) for the voxels
        point_size: Size of points in pixels
        ip_address: IP address for the viewer server
        port: Port for the viewer server
        device: Device to use ("cuda" or "cpu")
        auto_open: Whether to automatically open browser
    """
    check_fvdb()

    # Create grid from coordinates
    from .indexgrid import coords_to_grid
    grid = coords_to_grid(coords, device=device)

    # Visualize
    visualize_grid(
        grid,
        name=name,
        color=color,
        point_size=point_size,
        ip_address=ip_address,
        port=port,
        auto_open=auto_open,
    )


def visualize_nvdb_file(
    path: str,
    name: Optional[str] = None,
    color: Tuple[float, float, float] = (0.2, 0.6, 1.0),
    point_size: float = 3.0,
    ip_address: str = "127.0.0.1",
    port: int = 8080,
    device: str = "cuda",
    auto_open: bool = True,
):
    """
    Load and visualize a NanoVDB file.

    Args:
        path: Path to .nvdb file
        name: Scene name (defaults to filename)
        color: RGB color tuple for voxels
        point_size: Size of points in pixels
        ip_address: IP address for viewer server
        port: Port for viewer server
        device: Device to load onto
        auto_open: Whether to automatically open browser
    """
    check_fvdb()

    from .indexgrid import load_indexgrid
    import os

    # Load the grid
    grid = load_indexgrid(path, device=device)

    # Use filename as default name
    if name is None:
        name = os.path.basename(path)

    # Visualize
    visualize_grid(
        grid,
        name=name,
        color=color,
        point_size=point_size,
        ip_address=ip_address,
        port=port,
        auto_open=auto_open,
    )


def render_to_image(
    grid: "fvdb.GridBatch",
    output_path: str,
    width: int = 1920,
    height: int = 1080,
    azimuth: float = 45.0,
    elevation: float = 30.0,
    color: Tuple[float, float, float] = (0.2, 0.6, 1.0),
    background: Tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """
    Render a grid to a static image file.

    Note: This is a placeholder. fVDB's visualization is currently
    web-based and may not support direct image export. For static
    rendering, consider using PyVista or matplotlib.

    Args:
        grid: fVDB GridBatch to render
        output_path: Path to save image
        width: Image width
        height: Image height
        azimuth: Camera azimuth angle in degrees
        elevation: Camera elevation angle in degrees
        color: RGB color for voxels
        background: RGB background color
    """
    check_fvdb()

    # fVDB's viz module is primarily for interactive viewing
    # For static rendering, we can use matplotlib or similar

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        raise ImportError("matplotlib is required for static rendering")

    # Get coordinates
    ijk = grid.ijk[0].jdata.cpu().numpy()

    # Subsample if too many points for matplotlib
    max_points = 50000
    if len(ijk) > max_points:
        indices = np.random.choice(len(ijk), max_points, replace=False)
        ijk = ijk[indices]
        print(f"Subsampled to {max_points} points for rendering")

    # Create figure
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(ijk[:, 0], ijk[:, 1], ijk[:, 2],
               c=[color], s=1, alpha=0.6)

    # Set viewing angle
    ax.view_init(elev=elevation, azim=azimuth)

    # Set background
    ax.set_facecolor(background)

    # Equal aspect ratio
    max_range = np.array([ijk[:, 0].max() - ijk[:, 0].min(),
                          ijk[:, 1].max() - ijk[:, 1].min(),
                          ijk[:, 2].max() - ijk[:, 2].min()]).max() / 2.0
    mid_x = (ijk[:, 0].max() + ijk[:, 0].min()) * 0.5
    mid_y = (ijk[:, 1].max() + ijk[:, 1].min()) * 0.5
    mid_z = (ijk[:, 2].max() + ijk[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Saved render to {output_path}")


def create_rotation_images(
    grid: "fvdb.GridBatch",
    output_dir: str,
    num_frames: int = 36,
    prefix: str = "frame",
    width: int = 800,
    height: int = 800,
    elevation: float = 30.0,
    color: Tuple[float, float, float] = (0.2, 0.6, 1.0),
):
    """
    Create a series of images rotating around the object.

    Useful for creating GIFs or videos without interactive viewer.

    Args:
        grid: fVDB GridBatch to render
        output_dir: Directory to save frames
        num_frames: Number of frames (angles)
        prefix: Filename prefix
        width: Image width
        height: Image height
        elevation: Camera elevation angle
        color: RGB color for voxels
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_frames):
        azimuth = i * (360.0 / num_frames)
        output_path = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
        render_to_image(
            grid,
            output_path,
            width=width,
            height=height,
            azimuth=azimuth,
            elevation=elevation,
            color=color,
        )
        print(f"Frame {i+1}/{num_frames}")

    print(f"\nAll frames saved to {output_dir}")
    print("To create a GIF, run:")
    print(f"  convert -delay 10 -loop 0 {output_dir}/{prefix}_*.png animation.gif")
