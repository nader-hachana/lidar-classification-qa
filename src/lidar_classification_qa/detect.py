"""Flag points labeled as vegetation that geometrically behave like flat structures.

Real vegetation has vertical structure: returns scatter across trunk,
branches, and an uneven top. A flat roof or other hard structure doesn't,
every return sits close to the same height. A column that's both taller
than real vegetation realistically reaches and flatter than real
vegetation ever is, but still labeled vegetation, is very unlikely to
actually be vegetation.

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
    real_ground: np.ndarray,
    xmin: float,
    ymin: float,
    cell_size: float,
    nx: int,
    ny: int,
    max_class: int = 32,
    min_hag: float = 1.0,
) -> dict:
    """Collapse points into 2D (x, y) columns and summarize each one.

    Per column: point count, vertical spread of Z, mean/max height above
    ground, majority classification and its purity, and the fraction of
    points whose height above ground rests on real (not borrowed) ground
    data, see enrich.on_real_ground().

    Points near the ground (hag < min_hag) are dropped first, otherwise a
    column's own ground returns get folded into its vertical-spread
    statistic along with whatever is elevated above them.
    """
    from lidar_classification_qa.enrich import compute_cell_id

    if min_hag > 0:
        keep = hag >= min_hag
        x, y, z, hag, classification, real_ground = (
            x[keep],
            y[keep],
            z[keep],
            hag[keep],
            classification[keep],
            real_ground[keep],
        )

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
    real_ground_fraction, _ = mean_and_std(real_ground.astype(np.float64))

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
        "real_ground_fraction": real_ground_fraction,
    }


def flag_likely_structure(
    columns: dict,
    vegetation_classes: frozenset[int] = VEGETATION_CLASSES,
    min_height: float = 10.0,
    max_vertical_std: float = 1.0,
    min_purity: float = 0.7,
    min_points: int = 5,
    min_real_ground_fraction: float = 1.0,
) -> np.ndarray:
    """Columns labeled vegetation that are too tall and too flat to be real vegetation.

    min_points filters out single stray points, which are flat and tall by
    definition since they have no spread to measure. min_real_ground_fraction
    requires every contributing point to rest on real, not borrowed, ground
    data, borrowed ground can be off by tens of meters on hilly terrain and
    otherwise fakes "tall flat" columns that aren't real.
    """
    is_vegetation = np.isin(columns["majority_class"], list(vegetation_classes))
    is_tall = columns["hag_max"] >= min_height
    is_flat = columns["z_std"] <= max_vertical_std
    is_pure = columns["majority_fraction"] >= min_purity
    has_enough_points = columns["n_points"] >= min_points
    has_real_ground = columns["real_ground_fraction"] >= min_real_ground_fraction
    return is_vegetation & is_tall & is_flat & is_pure & has_enough_points & has_real_ground
