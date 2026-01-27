"""
fVDB Client - Fetch label topology from DVID and create fVDB IndexGrids.

This package provides tools to:
1. Fetch label voxel data from a DVID server
2. Create fVDB IndexGrid representations
3. Visualize the topology using fVDB's built-in viewer
4. Report statistics on voxel count and storage costs
"""

from .dvid_client import DVIDClient
from .indexgrid import IndexGridBuilder, save_indexgrid, load_indexgrid
from .stats import compute_stats, print_stats

__version__ = "0.1.0"

__all__ = [
    "DVIDClient",
    "IndexGridBuilder",
    "save_indexgrid",
    "load_indexgrid",
    "compute_stats",
    "print_stats",
]
