"""
IndexGrid builder using fVDB.

Creates fVDB GridBatch objects from voxel coordinates and provides
save/load functionality for NanoVDB files.
"""

from typing import Iterator, Optional
import numpy as np

# These imports require fVDB to be installed
try:
    import torch
    import fvdb
    from fvdb import JaggedTensor
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


class IndexGridBuilder:
    """
    Build fVDB IndexGrid from voxel coordinates.

    This class accumulates voxel coordinates and builds an fVDB GridBatch
    that represents the sparse voxel topology.
    """

    def __init__(self, name: str = "label", voxel_size: float = 1.0, device: str = "cuda"):
        """
        Initialize the IndexGrid builder.

        Args:
            name: Name for the grid (used when saving)
            voxel_size: Size of each voxel in world units
            device: PyTorch device to use ("cuda" or "cpu")
        """
        check_fvdb()
        self.name = name
        self.voxel_size = voxel_size
        self.device = device
        self._coords: list[np.ndarray] = []
        self._total_coords = 0

    def add_coords(self, coords: np.ndarray):
        """
        Add voxel coordinates to the builder.

        Args:
            coords: numpy array of shape (N, 3) with integer coordinates
        """
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coords must have shape (N, 3), got {coords.shape}")
        self._coords.append(coords.astype(np.int64))
        self._total_coords += len(coords)

    def add_coords_from_rles(self, rles: Iterator[tuple[int, int, int, int]]):
        """
        Add voxel coordinates from RLE iterator.

        Args:
            rles: Iterator yielding (x, y, z, length) tuples
        """
        batch = []
        batch_size = 100000

        for x, y, z, length in rles:
            for dx in range(length):
                batch.append([x + dx, y, z])
                if len(batch) >= batch_size:
                    self.add_coords(np.array(batch, dtype=np.int64))
                    batch = []

        if batch:
            self.add_coords(np.array(batch, dtype=np.int64))

    @property
    def total_coords(self) -> int:
        """Return the total number of coordinates added."""
        return self._total_coords

    def build(self) -> "fvdb.GridBatch":
        """
        Build the fVDB GridBatch from accumulated coordinates.

        Returns:
            fvdb.GridBatch containing the sparse voxel grid
        """
        check_fvdb()

        if not self._coords:
            raise ValueError("No coordinates added to builder")

        # Concatenate all coordinates
        all_coords = np.concatenate(self._coords, axis=0)

        # Remove duplicates
        all_coords = np.unique(all_coords, axis=0)

        # Convert to torch tensor
        coords_tensor = torch.from_numpy(all_coords).long()

        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            coords_tensor = coords_tensor.cuda()

        # Create JaggedTensor (single grid in batch)
        coords_jagged = JaggedTensor([coords_tensor])

        # Create GridBatch
        grid = fvdb.GridBatch.from_ijk(
            coords_jagged,
            voxel_sizes=[self.voxel_size] * 3,
            origins=[0.0] * 3,
        )

        return grid

    def clear(self):
        """Clear accumulated coordinates."""
        self._coords = []
        self._total_coords = 0


def save_indexgrid(grid: "fvdb.GridBatch", path: str, name: str = "label",
                   compressed: bool = False, verbose: bool = False):
    """
    Save an fVDB GridBatch to a NanoVDB file.

    Args:
        grid: fVDB GridBatch to save
        path: Output file path (should have .nvdb extension)
        name: Name for the grid in the file
        compressed: Whether to use Blosc compression
        verbose: Whether to print information about the saved grid
    """
    check_fvdb()
    grid.save_nanovdb(path, name=name, compressed=compressed, verbose=verbose)


def load_indexgrid(path: str, device: str = "cuda") -> "fvdb.GridBatch":
    """
    Load a GridBatch from a NanoVDB file.

    Args:
        path: Path to .nvdb file
        device: Device to load onto ("cuda" or "cpu")

    Returns:
        fvdb.GridBatch loaded from file
    """
    check_fvdb()

    # Determine actual device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    return fvdb.GridBatch.from_nanovdb(path, device=device)


def coords_to_grid(coords: np.ndarray, voxel_size: float = 1.0,
                   device: str = "cuda") -> "fvdb.GridBatch":
    """
    Create a GridBatch directly from numpy coordinates.

    Args:
        coords: numpy array of shape (N, 3) with integer coordinates
        voxel_size: Size of each voxel
        device: Device to create on

    Returns:
        fvdb.GridBatch
    """
    check_fvdb()

    # Remove duplicates
    coords = np.unique(coords, axis=0).astype(np.int64)

    # Convert to torch
    coords_tensor = torch.from_numpy(coords).long()

    if device == "cuda" and torch.cuda.is_available():
        coords_tensor = coords_tensor.cuda()

    # Create JaggedTensor
    coords_jagged = JaggedTensor([coords_tensor])

    # Create GridBatch
    return fvdb.GridBatch.from_ijk(
        coords_jagged,
        voxel_sizes=[voxel_size] * 3,
        origins=[0.0] * 3,
    )
