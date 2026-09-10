"""Ground surface and height-above-ground, computed independently of the file's own classification.

Adapted from the ground-extraction logic in copc-pointcloud-pipeline
(github.com/nader-hachana/copc-pointcloud-pipeline): a low percentile of Z
per grid cell, robust to the occasional stray low point, filled from the
nearest non-empty cell so a tile's ground estimate has no holes.

Every function here works on plain numpy arrays in and out, so all of it is
unit tested with small made-up point clouds instead of a real file.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_cell_id(x: np.ndarray, y: np.ndarray, xmin: float, ymin: float, cell_size: float, nx: int, ny: int) -> np.ndarray:
    """Which flat 2D grid cell each point falls into, as one integer id per point."""
    cx = np.clip(((x - xmin) / cell_size).astype(np.int64), 0, nx - 1)
    cy = np.clip(((y - ymin) / cell_size).astype(np.int64), 0, ny - 1)
    return cy * nx + cx


def compute_ground_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    cell_size: float,
    percentile: float = 10.0,
) -> tuple[np.ndarray, int, int]:
    """A low percentile of Z per cell, a robust stand-in for ground elevation.

    A percentile instead of the minimum on purpose: the minimum is exactly
    one noisy point away from being wrong, a percentile needs several points
    to agree before it moves.
    """
    nx = max(1, int(np.ceil((xmax - xmin) / cell_size)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell_size)))
    cell_id = compute_cell_id(x, y, xmin, ymin, cell_size, nx, ny)

    ground_z = np.full(nx * ny, np.nan)
    for cid in np.unique(cell_id):
        ground_z[cid] = np.percentile(z[cell_id == cid], percentile)

    ground_z = ground_z.reshape(ny, nx)
    empty = np.isnan(ground_z)
    if empty.any() and not empty.all():
        _, nearest = distance_transform_edt(empty, return_indices=True)
        ground_z = ground_z[tuple(nearest)]

    return ground_z.reshape(-1), nx, ny


def height_above_ground(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    ground_z: np.ndarray,
    xmin: float,
    ymin: float,
    cell_size: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Each point's elevation minus the ground estimate for its own cell, never negative."""
    cell_id = compute_cell_id(x, y, xmin, ymin, cell_size, nx, ny)
    hag = z - ground_z[cell_id]
    return np.maximum(hag, 0.0)
