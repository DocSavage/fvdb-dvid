#!/usr/bin/env python3
"""
Command-line interface for fVDB Client.

Fetch label topology from DVID, create IndexGrids, view statistics,
and visualize the data.
"""

import argparse
import sys
import os


def cmd_fetch(args):
    """Fetch label data from DVID and save as IndexGrid."""
    from .dvid_client import DVIDClient, SparsevolConfig
    from .indexgrid import IndexGridBuilder, save_indexgrid
    from .stats import compute_stats_from_grid, print_stats

    print(f"Connecting to DVID server: {args.server}")
    print(f"UUID: {args.uuid}, Instance: {args.instance}, Label: {args.label}")

    client = DVIDClient(args.server, args.uuid, args.instance, timeout=args.timeout)

    # Configure request
    config = SparsevolConfig(
        supervoxels=args.supervoxels,
        scale=args.scale,
    )

    # Check if label exists
    print("\nChecking label info...")
    info = client.get_label_info(args.label, config)
    if info is None:
        print(f"Error: Label {args.label} not found")
        return 1

    print(f"  Voxels: {info.voxel_count:,}")
    print(f"  Blocks: {info.block_count:,}")
    print(f"  Bounding box: {info.min_voxel} to {info.max_voxel}")

    # Create IndexGrid builder
    print("\nFetching voxel data...")
    builder = IndexGridBuilder(
        name=f"label_{args.label}",
        voxel_size=args.voxel_size,
        device=args.device,
    )

    # Fetch and add coordinates
    rle_count = 0
    for x, y, z, length in client.get_sparsevol_rles(args.label, config):
        for dx in range(length):
            builder.add_coords(__import__('numpy').array([[x + dx, y, z]], dtype=__import__('numpy').int64))
        rle_count += 1
        if rle_count % 10000 == 0:
            print(f"  Processed {rle_count:,} RLE spans, {builder.total_coords:,} voxels...")

    print(f"  Total: {builder.total_coords:,} voxels from {rle_count:,} spans")

    # Build the grid
    print("\nBuilding IndexGrid...")
    grid = builder.build()
    print(f"  Grid created with {grid.total_voxels:,} unique voxels")

    # Save to file
    output_path = args.output or f"label_{args.label}.nvdb"
    print(f"\nSaving to {output_path}...")
    save_indexgrid(grid, output_path, name=f"label_{args.label}",
                   compressed=args.compress, verbose=True)

    # Print statistics
    if not args.quiet:
        stats = compute_stats_from_grid(grid, output_path)
        print()
        print_stats(stats, label=args.label)

    print(f"\nDone! Output saved to: {output_path}")
    return 0


def cmd_stats(args):
    """Show statistics for a .nvdb file."""
    from .indexgrid import load_indexgrid
    from .stats import compute_stats_from_grid, print_stats

    print(f"Loading {args.file}...")
    grid = load_indexgrid(args.file, device=args.device)

    stats = compute_stats_from_grid(grid, args.file)
    print_stats(stats, label=args.label)

    return 0


def cmd_view(args):
    """Visualize a .nvdb file with interactive 3D viewer."""
    from .visualize import visualize_nvdb_file

    # Parse color
    color = tuple(float(c) for c in args.color.split(','))

    visualize_nvdb_file(
        args.file,
        name=args.name,
        color=color,
        point_size=args.point_size,
        ip_address=args.ip,
        port=args.port,
        device=args.device,
        auto_open=not args.no_browser,
    )
    return 0


def cmd_render(args):
    """Render a .nvdb file to static image(s)."""
    from .indexgrid import load_indexgrid
    from .visualize import render_to_image, create_rotation_images

    print(f"Loading {args.file}...")
    grid = load_indexgrid(args.file, device=args.device)

    # Parse color
    color = tuple(float(c) for c in args.color.split(','))

    if args.rotate:
        # Create rotation animation frames
        create_rotation_images(
            grid,
            args.output,
            num_frames=args.frames,
            width=args.width,
            height=args.height,
            elevation=args.elevation,
            color=color,
        )
    else:
        # Single image
        render_to_image(
            grid,
            args.output,
            width=args.width,
            height=args.height,
            azimuth=args.azimuth,
            elevation=args.elevation,
            color=color,
        )

    return 0


def cmd_info(args):
    """Get info about a label from DVID (without downloading full data)."""
    from .dvid_client import DVIDClient, SparsevolConfig

    client = DVIDClient(args.server, args.uuid, args.instance, timeout=args.timeout)

    config = SparsevolConfig(
        supervoxels=args.supervoxels,
        scale=args.scale,
    )

    info = client.get_label_info(args.label, config)
    if info is None:
        print(f"Label {args.label} not found")
        return 1

    print(f"Label {info.label}:")
    print(f"  Voxels:       {info.voxel_count:,}")
    print(f"  Blocks:       {info.block_count:,}")
    print(f"  Min voxel:    {info.min_voxel}")
    print(f"  Max voxel:    {info.max_voxel}")

    # Estimate storage
    extent = tuple(info.max_voxel[i] - info.min_voxel[i] + 1 for i in range(3))
    print(f"  Extent:       {extent[0]} x {extent[1]} x {extent[2]}")

    # Block size
    block_size = client.get_block_size()
    print(f"  Block size:   {block_size}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="fVDB Client - Fetch label topology from DVID and create IndexGrids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Get info about a label (quick, no download)
  fvdb-client info --server http://dvid.example.org:8000 --uuid abc123 \\
      --instance segmentation --label 12345

  # Fetch label and create IndexGrid
  fvdb-client fetch --server http://dvid.example.org:8000 --uuid abc123 \\
      --instance segmentation --label 12345 --output label.nvdb

  # View statistics for a saved IndexGrid
  fvdb-client stats label.nvdb

  # Visualize with interactive 3D viewer (requires GPU)
  fvdb-client view label.nvdb

  # Render to static image
  fvdb-client render label.nvdb --output render.png

  # Create rotation animation frames
  fvdb-client render label.nvdb --output frames/ --rotate --frames 72
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Common DVID arguments
    def add_dvid_args(p):
        p.add_argument('--server', '-s', required=True,
                       help='DVID server URL (e.g., http://dvid.example.org:8000)')
        p.add_argument('--uuid', '-u', required=True,
                       help='Version UUID or branch (e.g., :master)')
        p.add_argument('--instance', '-i', required=True,
                       help='Labelmap instance name (e.g., segmentation)')
        p.add_argument('--label', '-l', type=int, required=True,
                       help='Label ID to fetch')
        p.add_argument('--supervoxels', action='store_true',
                       help='Interpret label as supervoxel ID')
        p.add_argument('--scale', type=int, default=0,
                       help='Scale level (0 = highest resolution)')
        p.add_argument('--timeout', type=int, default=300,
                       help='Request timeout in seconds (default: 300)')

    # info command
    info_parser = subparsers.add_parser('info', help='Get label info from DVID')
    add_dvid_args(info_parser)
    info_parser.set_defaults(func=cmd_info)

    # fetch command
    fetch_parser = subparsers.add_parser('fetch', help='Fetch label and create IndexGrid')
    add_dvid_args(fetch_parser)
    fetch_parser.add_argument('--output', '-o',
                              help='Output .nvdb file path (default: label_<id>.nvdb)')
    fetch_parser.add_argument('--device', default='cuda',
                              help='Device: cuda or cpu (default: cuda)')
    fetch_parser.add_argument('--voxel-size', type=float, default=1.0,
                              help='Voxel size in world units (default: 1.0)')
    fetch_parser.add_argument('--compress', action='store_true',
                              help='Use Blosc compression')
    fetch_parser.add_argument('--quiet', '-q', action='store_true',
                              help='Suppress statistics output')
    fetch_parser.set_defaults(func=cmd_fetch)

    # stats command
    stats_parser = subparsers.add_parser('stats', help='Show IndexGrid statistics')
    stats_parser.add_argument('file', help='Path to .nvdb file')
    stats_parser.add_argument('--label', '-l', type=int,
                              help='Label ID for display')
    stats_parser.add_argument('--device', default='cpu',
                              help='Device: cuda or cpu (default: cpu)')
    stats_parser.set_defaults(func=cmd_stats)

    # view command
    view_parser = subparsers.add_parser('view', help='Visualize with interactive 3D viewer')
    view_parser.add_argument('file', help='Path to .nvdb file')
    view_parser.add_argument('--name', help='Scene name')
    view_parser.add_argument('--device', default='cuda',
                             help='Device: cuda or cpu (default: cuda)')
    view_parser.add_argument('--color', default='0.2,0.6,1.0',
                             help='RGB color (default: 0.2,0.6,1.0)')
    view_parser.add_argument('--point-size', type=float, default=3.0,
                             help='Point size in pixels (default: 3.0)')
    view_parser.add_argument('--ip', default='127.0.0.1',
                             help='Viewer IP address (default: 127.0.0.1)')
    view_parser.add_argument('--port', type=int, default=8080,
                             help='Viewer port (default: 8080)')
    view_parser.add_argument('--no-browser', action='store_true',
                             help="Don't auto-open browser")
    view_parser.set_defaults(func=cmd_view)

    # render command
    render_parser = subparsers.add_parser('render', help='Render to static image')
    render_parser.add_argument('file', help='Path to .nvdb file')
    render_parser.add_argument('--output', '-o', required=True,
                               help='Output image path or directory (for --rotate)')
    render_parser.add_argument('--device', default='cpu',
                               help='Device: cuda or cpu (default: cpu)')
    render_parser.add_argument('--width', type=int, default=1920,
                               help='Image width (default: 1920)')
    render_parser.add_argument('--height', type=int, default=1080,
                               help='Image height (default: 1080)')
    render_parser.add_argument('--azimuth', type=float, default=45.0,
                               help='Camera azimuth in degrees (default: 45)')
    render_parser.add_argument('--elevation', type=float, default=30.0,
                               help='Camera elevation in degrees (default: 30)')
    render_parser.add_argument('--color', default='0.2,0.6,1.0',
                               help='RGB color (default: 0.2,0.6,1.0)')
    render_parser.add_argument('--rotate', action='store_true',
                               help='Create rotation animation frames')
    render_parser.add_argument('--frames', type=int, default=36,
                               help='Number of rotation frames (default: 36)')
    render_parser.set_defaults(func=cmd_render)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
