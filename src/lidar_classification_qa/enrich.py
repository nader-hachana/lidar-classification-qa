"""Ground surface and height-above-ground, using only the file's ground-classified points.

Ground is estimated from a low percentile of ground-classified (class 2)
points per grid cell, not from every point regardless of class. Cells with
no ground points nearby borrow an estimate from the closest cell that has
one, which is tracked so callers can avoid trusting a borrowed value.
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_cell_id(x: np.ndarray, y: np.ndarray, xmin: float, ymin: float, cell_size: float, nx: int, ny: int) -> np.ndarray:
    """Flat 2D grid cell index for each point."""
    cx = np.clip(((x - xmin) / cell_size).astype(np.int64), 0, nx - 1)
    cy = np.clip(((y - ymin) / cell_size).astype(np.int64), 0, ny - 1)
    return cy * nx + cx


def compute_ground_grid(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classification: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    cell_size: float,
    ground_class: int = 2,
    percentile: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Low percentile of ground-classified Z per cell.

    Returns (ground_z, has_local_ground, nx, ny). has_local_ground is False
    for cells with no ground-classified points, where the value was
    borrowed from the nearest cell that has one instead.
    """
    nx = max(1, int(np.ceil((xmax - xmin) / cell_size)))
    ny = max(1, int(np.ceil((ymax - ymin) / cell_size)))
    n_cells = nx * ny

    is_ground = classification == ground_class
    gx, gy, gz = x[is_ground], y[is_ground], z[is_ground]
    cell_id = compute_cell_id(gx, gy, xmin, ymin, cell_size, nx, ny)

    ground_z = np.full(n_cells, np.nan)
    if len(cell_id) > 0:
        order = np.argsort(cell_id, kind="stable")
        cid_sorted = cell_id[order]
        z_sorted = gz[order]
        boundaries = np.searchsorted(cid_sorted, np.arange(n_cells + 1))
        for i in range(n_cells):
            lo, hi = boundaries[i], boundaries[i + 1]
            if hi > lo:
                ground_z[i] = np.percentile(z_sorted[lo:hi], percentile)

    has_local_ground = ~np.isnan(ground_z)
    grid = ground_z.reshape(ny, nx)
    empty = np.isnan(grid)
    if empty.any() and not empty.all():
        _, nearest = distance_transform_edt(empty, return_indices=True)
        grid = grid[tuple(nearest)]
    elif empty.all():
        grid = np.zeros_like(grid)

    return grid.reshape(-1), has_local_ground, nx, ny


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
    """Elevation minus the ground estimate for each point's cell, clipped at 0."""
    cell_id = compute_cell_id(x, y, xmin, ymin, cell_size, nx, ny)
    hag = z - ground_z[cell_id]
    return np.maximum(hag, 0.0)


def on_real_ground(
    x: np.ndarray,
    y: np.ndarray,
    has_local_ground: np.ndarray,
    xmin: float,
    ymin: float,
    cell_size: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """Whether each point's ground estimate is real (not borrowed)."""
    cell_id = compute_cell_id(x, y, xmin, ymin, cell_size, nx, ny)
    return has_local_ground[cell_id]
