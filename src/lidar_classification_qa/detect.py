"""Flag points labeled as vegetation that geometrically behave like flat structures.

The idea: real vegetation canopy has vertical structure, returns scatter
across the trunk, branches, and an uneven top, so a column of "vegetation"
points has real spread in Z. A flat roof or other hard structure does not,
every return in that column sits close to the same height. A column that is
both tall (height above ground far past what vegetation in the area
realistically reaches) and flat (low vertical spread) but still labeled
vegetation is very unlikely to actually be vegetation.

This isn't a guess: it's how a real 138m stadium roof canopy showed up
labeled class 4 ("medium vegetation") in a public LiDAR file, see the
write-up in this repo's README.

ASPRS LAS classification codes used here: 2 = ground, 3/4/5 = low/medium/
high vegetation, 6 = building.
"""

import numpy as np

VEGETATION_CLASSES = frozenset({3, 4, 5})


def aggregate_columns(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    hag: np.ndarray,
    classification: np.ndarray,
    xmin: float,
    ymin: float,
    cell_size: float,
    nx: int,
    ny: int,
    max_class: int = 32,
    min_hag: float = 1.0,
) -> dict:
    """Collapse points into 2D (x, y) columns and summarize each one.

    Per column: point count, vertical spread of Z (the flatness signal),
    mean and max height above ground, and the majority classification with
    how pure that majority is (a column split 50/50 between two classes is
    a much weaker signal than one that's 95% a single class).

    Points near the ground (hag < min_hag) are dropped before aggregating.
    Without this, a column's own real ground returns get folded into the
    same "vertical spread" statistic as whatever is above them, ground at
    z=0 plus a flat roof at z=40 looks exactly like a huge vertical spread,
    the opposite of what a flat roof should look like. The stats here
    describe the shape of what's elevated in a column, not the column's
    full ground-to-sky range.
    """
    from lidar_classification_qa.enrich import compute_cell_id

    if min_hag > 0:
        keep = hag >= min_hag
        x, y, z, hag, classification = x[keep], y[keep], z[keep], hag[keep], classification[keep]

    cell_id = compute_cell_id(x, y, xmin, ymin, cell_size, nx, ny)
    unique_cells, inverse, n_points = np.unique(cell_id, return_inverse=True, return_counts=True)
    n = len(unique_cells)

    def mean_and_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.bincount(inverse, weights=values, minlength=n)
        s2 = np.bincount(inverse, weights=values**2, minlength=n)
        mean = s / n_points
        variance = np.maximum(s2 / n_points - mean**2, 0.0)
        return mean, np.sqrt(variance)

    z_mean, z_std = mean_and_std(z)
    hag_mean, _ = mean_and_std(hag)

    hag_max = np.zeros(n)
    np.maximum.at(hag_max, inverse, hag)

    class_key = inverse * max_class + classification.astype(np.int64)
    class_counts = np.bincount(class_key, minlength=n * max_class).reshape(n, max_class)
    majority_class = class_counts.argmax(axis=1)
    majority_fraction = class_counts.max(axis=1) / n_points

    cy = unique_cells // nx
    cx = unique_cells % nx

    return {
        "cell_id": unique_cells,
        "x_center": xmin + (cx + 0.5) * cell_size,
        "y_center": ymin + (cy + 0.5) * cell_size,
        "n_points": n_points,
        "z_mean": z_mean,
        "z_std": z_std,
        "hag_mean": hag_mean,
        "hag_max": hag_max,
        "majority_class": majority_class,
        "majority_fraction": majority_fraction,
    }


def flag_likely_structure(
    columns: dict,
    vegetation_classes: frozenset[int] = VEGETATION_CLASSES,
    min_height: float = 10.0,
    max_vertical_std: float = 1.0,
    min_purity: float = 0.7,
) -> np.ndarray:
    """Columns labeled vegetation that are too tall and too flat to be real vegetation.

    Defaults are deliberately conservative (10m, a real height a tree could
    reach, and 1m of vertical spread, a real roof or wall is usually flatter
    than this): this is a coarse first pass meant to surface candidates for
    a human to check, not a silent auto-correction of the classification.
    """
    is_vegetation = np.isin(columns["majority_class"], list(vegetation_classes))
    is_tall = columns["hag_max"] >= min_height
    is_flat = columns["z_std"] <= max_vertical_std
    is_pure = columns["majority_fraction"] >= min_purity
    return is_vegetation & is_tall & is_flat & is_pure
