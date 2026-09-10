"""Ground surface and height-above-ground, computed independently of the file's own classification of anything except the ground itself.

Adapted from the ground-extraction logic in copc-pointcloud-pipeline
(github.com/nader-hachana/copc-pointcloud-pipeline): a low percentile of Z
per grid cell, robust to the occasional stray low point, filled from the
nearest non-empty cell so a tile's ground estimate has no holes.

The one thing this does trust from the file's own classification is which
points are ground (class 2 by default). That's a deliberate, narrower kind
of trust than trusting the whole classification field: ground is the one
class almost every LiDAR delivery gets right (SoFi Stadium included, its
real classification bug is a roof mislabeled as vegetation, not its ground
points), and anchoring height above ground to it is what makes the rest of
this project's flagging trustworthy instead of guessing at what ground even
is. Testing against a hillside/forest file (Autzen) with real gaps in
ground coverage under dense canopy is what surfaced the need for this: a
percentile of every point in a cell, regardless of class, quietly grabs a
non-ground point as "ground" wherever real ground returns are sparse, and
on hilly terrain that error can be enormous.

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
    classification: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    cell_size: float,
    ground_class: int = 2,
    percentile: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """A low percentile of real ground-classified points' Z, per cell.

    A percentile instead of the minimum on purpose: the minimum is exactly
    one noisy point away from being wrong, a percentile needs several points
    to agree before it moves.

    Returns (ground_z, has_local_ground, nx, ny). has_local_ground marks
    which cells actually had ground-classified points to measure from,
    False where the value was instead borrowed from the nearest cell that
    does, false for a cell entirely under dense canopy that never got a
    real ground return, for instance. Callers that flag things relative to
    height above ground should treat a flag built on borrowed ground as
    unreliable and skip it, borrowed ground can be right next to correct
    but, on real hilly terrain, can also be tens of meters off.
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
        # no ground-classified points anywhere in range: nothing real to
        # borrow from either, has_local_ground is all False so nothing
        # downstream will trust this, the zeros here are just a placeholder
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
    """Each point's elevation minus the ground estimate for its own cell, never negative."""
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
    """Whether each point's height above ground rests on a real local ground measurement, not a borrowed one."""
    cell_id = compute_cell_id(x, y, xmin, ymin, cell_size, nx, ny)
    return has_local_ground[cell_id]
