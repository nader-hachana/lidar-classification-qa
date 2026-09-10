"""Run the misclassified-vegetation detector against a real LAS/LAZ/COPC file."""

import argparse
import csv
import sys

import laspy
import numpy as np

from lidar_classification_qa.detect import flag_likely_structure
from lidar_classification_qa.streaming import estimate_ground_grid_streaming, scan_columns_streaming


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to a LAS/LAZ/COPC file")
    parser.add_argument("--cell-size", type=float, default=2.0, help="Column analysis grid cell size in meters")
    parser.add_argument(
        "--ground-cell-size",
        type=float,
        default=10.0,
        help="Ground estimation grid cell size in meters, kept coarser than --cell-size on purpose",
    )
    parser.add_argument("--chunk-size", type=int, default=2_000_000, help="Points read per chunk")
    parser.add_argument("--sample-size", type=int, default=3_000_000, help="Points sampled to build the ground grid")
    parser.add_argument("--ground-percentile", type=float, default=10.0)
    parser.add_argument(
        "--min-hag-for-stats",
        type=float,
        default=1.0,
        help="Drop points below this height above ground before computing column stats",
    )
    parser.add_argument("--min-height", type=float, default=10.0, help="Min height above ground to flag, in meters")
    parser.add_argument("--max-vertical-std", type=float, default=1.0, help="Max vertical spread to flag, in meters")
    parser.add_argument("--min-purity", type=float, default=0.7, help="Min fraction of a column in the majority class")
    parser.add_argument("--min-points", type=int, default=5, help="Min points in a column to trust a flag")
    parser.add_argument(
        "--min-real-ground-fraction",
        type=float,
        default=1.0,
        help="Min fraction of a column's points resting on real (not borrowed) ground data to trust a flag",
    )
    parser.add_argument("--output", help="Optional CSV path to write flagged columns to")
    args = parser.parse_args()

    print(f"Pass 1/2: estimating ground grid from a {args.sample_size:,}-point sample...", file=sys.stderr)
    ground_z, has_local_ground, xmin, ymin, ground_nx, ground_ny = estimate_ground_grid_streaming(
        args.path, args.ground_cell_size, args.sample_size, args.chunk_size, percentile=args.ground_percentile
    )
    print(
        f"Ground grid: {ground_nx} x {ground_ny} cells at {args.ground_cell_size}m, "
        f"{has_local_ground.sum():,} with real local ground data ({100 * has_local_ground.mean():.1f}%)",
        file=sys.stderr,
    )

    with laspy.open(args.path) as f:
        xmax, ymax = float(f.header.maxs[0]), float(f.header.maxs[1])
    nx = max(1, int(np.ceil((xmax - xmin) / args.cell_size)))
    ny = max(1, int(np.ceil((ymax - ymin) / args.cell_size)))
    print(f"Column grid: {nx} x {ny} = {nx * ny:,} cells at {args.cell_size}m", file=sys.stderr)

    print("Pass 2/2: scanning the full file into per-column statistics...", file=sys.stderr)
    columns = scan_columns_streaming(
        args.path,
        ground_z,
        has_local_ground,
        xmin,
        ymin,
        args.ground_cell_size,
        ground_nx,
        ground_ny,
        args.cell_size,
        nx,
        ny,
        args.chunk_size,
        min_hag=args.min_hag_for_stats,
    )

    flagged = flag_likely_structure(
        columns,
        min_height=args.min_height,
        max_vertical_std=args.max_vertical_std,
        min_purity=args.min_purity,
        min_points=args.min_points,
        min_real_ground_fraction=args.min_real_ground_fraction,
    )

    n_occupied = len(columns["cell_id"])
    n_flagged = int(flagged.sum())
    print(f"\n{n_occupied:,} occupied columns, {n_flagged} flagged as likely misclassified vegetation.")

    if n_flagged:
        flagged_x = columns["x_center"][flagged]
        flagged_y = columns["y_center"][flagged]
        flagged_hag = columns["hag_max"][flagged]
        flagged_std = columns["z_std"][flagged]
        flagged_class = columns["majority_class"][flagged]
        flagged_purity = columns["majority_fraction"][flagged]

        order = np.argsort(-flagged_hag)
        print("\nTop flagged columns by height above ground:")
        print(f"{'x':>12} {'y':>12} {'height (m)':>12} {'vert. std (m)':>14} {'class':>6} {'purity':>7}")
        for i in order[:20]:
            print(
                f"{flagged_x[i]:12.1f} {flagged_y[i]:12.1f} {flagged_hag[i]:12.1f} "
                f"{flagged_std[i]:14.3f} {flagged_class[i]:6d} {flagged_purity[i]:7.2f}"
            )

    if args.output:
        with open(args.output, "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(
                [
                    "x_center",
                    "y_center",
                    "hag_max",
                    "z_std",
                    "majority_class",
                    "majority_fraction",
                    "n_points",
                    "real_ground_fraction",
                ]
            )
            for i in np.flatnonzero(flagged):
                writer.writerow(
                    [
                        columns["x_center"][i],
                        columns["y_center"][i],
                        columns["hag_max"][i],
                        columns["z_std"][i],
                        columns["majority_class"][i],
                        columns["majority_fraction"][i],
                        columns["n_points"][i],
                        columns["real_ground_fraction"][i],
                    ]
                )
        print(f"\nWrote {n_flagged} flagged columns to {args.output}")


if __name__ == "__main__":
    main()
