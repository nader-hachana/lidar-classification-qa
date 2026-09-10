import numpy as np

from lidar_classification_qa.enrich import compute_ground_grid, height_above_ground


def test_ground_grid_uses_low_percentile_not_minimum():
    # one cell, a real cluster of ground points around z=0, one noisy point way below
    rng = np.random.default_rng(0)
    cluster = rng.normal(loc=0.0, scale=0.05, size=20)
    z = np.concatenate([cluster, [-50.0]])
    x = np.full(z.shape, 0.5)
    y = np.full(z.shape, 0.5)

    ground_z, nx, ny = compute_ground_grid(x, y, z, xmin=0, ymin=0, xmax=1, ymax=1, cell_size=1.0, percentile=10.0)

    assert nx == 1 and ny == 1
    # the percentile should stay close to the real ground cluster, not get dragged to -50
    assert ground_z[0] > -1


def test_empty_cells_fill_from_nearest_neighbour():
    # two cells side by side, only the left one has points
    x = np.array([0.5])
    y = np.array([0.5])
    z = np.array([10.0])

    ground_z, nx, ny = compute_ground_grid(x, y, z, xmin=0, ymin=0, xmax=2, ymax=1, cell_size=1.0, percentile=10.0)

    assert nx == 2 and ny == 1
    assert not np.isnan(ground_z).any()
    assert ground_z[1] == ground_z[0]


def test_height_above_ground_never_negative():
    x = np.array([0.5, 0.5])
    y = np.array([0.5, 0.5])
    z = np.array([5.0, -1.0])
    ground_z = np.array([0.0])

    hag = height_above_ground(x, y, z, ground_z, xmin=0, ymin=0, cell_size=1.0, nx=1, ny=1)

    assert hag[0] == 5.0
    assert hag[1] == 0.0
