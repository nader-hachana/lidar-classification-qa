"""Two-pass scan over a LAS/LAZ/COPC file, memory bounded independent of point count.

Pass 1 takes a probabilistic subsample of points, its size is a fixed
target, not a fraction tied to the file, to build a per-cell ground grid.
Pass 2 streams the file again in fixed-size chunks, folding each chunk into
running per-column accumulators (sum, sum of squares, count, max,
classification counts, real-ground fraction). Peak memory in both passes is
bounded by the number of grid columns and the sample size, not by how many
points are in the file, the same principle the ground-up streaming design
in copc-pointcloud-pipeline
(github.com/nader-hachana/copc-pointcloud-pipeline) is built on.
"""

import laspy
import numpy as np

from lidar_classification_qa.enrich import compute_cell_id, compute_ground_grid


def estimate_ground_grid_streaming(
    path: str,
    ground_cell_size: float,
    sample_size: int = 3_000_000,
    chunk_size: int = 2_000_000,
    ground_class: int = 2,
    percentile: float = 10.0,
    seed: int = 0,
    max_class: int = 32,
) -> tuple[np.ndarray, np.ndarray, float, float, int, int]:
    """Ground grid from a probabilistic subsample sized independent of the file's total point count.

    ground_cell_size is deliberately meant to be coarser than the cell size
    used later for column analysis. A cell fully covered by a flat elevated
    structure has few or no real ground returns of its own, a coarse cell
    reaches past a structure's own footprint to real ground nearby.

    Returns (ground_z, has_local_ground, xmin, ymin, nx, ny). See
    enrich.compute_ground_grid() for what has_local_ground means and why it
    matters: cells with no real ground-classified points nearby get an
    estimate borrowed from elsewhere, which on hilly or heavily forested
    terrain can be badly wrong, and flagging shouldn't trust it.
    """
    with laspy.open(path) as f:
        header = f.header
        xmin, ymin = float(header.mins[0]), float(header.mins[1])
        xmax, ymax = float(header.maxs[0]), float(header.maxs[1])
        total_points = header.point_count
        fraction = min(1.0, sample_size / max(total_points, 1))
        rng = np.random.default_rng(seed)

        sample_x: list[np.ndarray] = []
        sample_y: list[np.ndarray] = []
        sample_z: list[np.ndarray] = []
        sample_c: list[np.ndarray] = []
        for chunk in f.chunk_iterator(chunk_size):
            keep = rng.random(len(chunk.x)) < fraction
            if keep.any():
                sample_x.append(np.asarray(chunk.x)[keep])
                sample_y.append(np.asarray(chunk.y)[keep])
                sample_z.append(np.asarray(chunk.z)[keep])
                sample_c.append(np.clip(np.asarray(chunk.classification)[keep].astype(np.int64), 0, max_class - 1))

    x = np.concatenate(sample_x)
    y = np.concatenate(sample_y)
    z = np.concatenate(sample_z)
    classification = np.concatenate(sample_c)

    ground_z, has_local_ground, ground_nx, ground_ny = compute_ground_grid(
        x, y, z, classification, xmin, ymin, xmax, ymax, ground_cell_size, ground_class, percentile
    )
    return ground_z, has_local_ground, xmin, ymin, ground_nx, ground_ny


def scan_columns_streaming(
    path: str,
    ground_z: np.ndarray,
    has_local_ground: np.ndarray,
    xmin: float,
    ymin: float,
    ground_cell_size: float,
    ground_nx: int,
    ground_ny: int,
    cell_size: float,
    nx: int,
    ny: int,
    chunk_size: int = 2_000_000,
    max_class: int = 32,
    min_hag: float = 1.0,
) -> dict:
    """Fold the file through in fixed-size chunks into running per-column accumulators.

    Shaped to match aggregate_columns() in detect.py, so flag_likely_structure
    works on the result of either function without caring which one produced it.

    ground_cell_size/ground_nx/ground_ny describe the (coarser) grid ground_z
    and has_local_ground were estimated on, cell_size/nx/ny describe the
    (finer) grid used for the column statistics below, the two are
    deliberately different resolutions, see estimate_ground_grid_streaming().

    Points near the ground (hag < min_hag) are dropped from every chunk
    before folding it into the running totals, for the same reason
    aggregate_columns() does it: a column's own ground returns shouldn't
    count toward the vertical-spread statistic of whatever is elevated
    above them.
    """
    n_cells = nx * ny
    count = np.zeros(n_cells, dtype=np.int64)
    sum_z = np.zeros(n_cells)
    sum_z2 = np.zeros(n_cells)
    sum_hag = np.zeros(n_cells)
    max_hag = np.zeros(n_cells)
    sum_real_ground = np.zeros(n_cells)
    class_counts = np.zeros(n_cells * max_class, dtype=np.int64)

    with laspy.open(path) as f:
        for chunk in f.chunk_iterator(chunk_size):
            x = np.asarray(chunk.x)
            y = np.asarray(chunk.y)
            z = np.asarray(chunk.z)
            classification = np.clip(np.asarray(chunk.classification).astype(np.int64), 0, max_class - 1)

            ground_cell_id = compute_cell_id(x, y, xmin, ymin, ground_cell_size, ground_nx, ground_ny)
            hag_all = np.maximum(z - ground_z[ground_cell_id], 0.0)
            real_ground_all = has_local_ground[ground_cell_id]
            cell_id_all = compute_cell_id(x, y, xmin, ymin, cell_size, nx, ny)

            keep = hag_all >= min_hag
            cell_id = cell_id_all[keep]
            z = z[keep]
            hag = hag_all[keep]
            classification = classification[keep]
            real_ground = real_ground_all[keep]

            count += np.bincount(cell_id, minlength=n_cells)
            sum_z += np.bincount(cell_id, weights=z, minlength=n_cells)
            sum_z2 += np.bincount(cell_id, weights=z**2, minlength=n_cells)
            sum_hag += np.bincount(cell_id, weights=hag, minlength=n_cells)
            sum_real_ground += np.bincount(cell_id, weights=real_ground.astype(np.float64), minlength=n_cells)

            chunk_max = np.zeros(n_cells)
            np.maximum.at(chunk_max, cell_id, hag)
            np.maximum(max_hag, chunk_max, out=max_hag)

            class_key = cell_id * max_class + classification
            class_counts += np.bincount(class_key, minlength=n_cells * max_class)

    has_points = count > 0
    z_mean = np.divide(sum_z, count, out=np.zeros(n_cells), where=has_points)
    variance = np.divide(sum_z2, count, out=np.zeros(n_cells), where=has_points) - z_mean**2
    z_std = np.sqrt(np.maximum(variance, 0.0))
    hag_mean = np.divide(sum_hag, count, out=np.zeros(n_cells), where=has_points)
    real_ground_fraction = np.divide(sum_real_ground, count, out=np.zeros(n_cells), where=has_points)

    class_counts = class_counts.reshape(n_cells, max_class)
    majority_class = class_counts.argmax(axis=1)
    majority_fraction = np.divide(class_counts.max(axis=1), count, out=np.zeros(n_cells), where=has_points)

    cell_ids = np.arange(n_cells)
    cy = cell_ids // nx
    cx = cell_ids % nx

    return {
        "cell_id": cell_ids[has_points],
        "x_center": (xmin + (cx + 0.5) * cell_size)[has_points],
        "y_center": (ymin + (cy + 0.5) * cell_size)[has_points],
        "n_points": count[has_points],
        "z_mean": z_mean[has_points],
        "z_std": z_std[has_points],
        "hag_mean": hag_mean[has_points],
        "hag_max": max_hag[has_points],
        "majority_class": majority_class[has_points],
        "majority_fraction": majority_fraction[has_points],
        "real_ground_fraction": real_ground_fraction[has_points],
    }
