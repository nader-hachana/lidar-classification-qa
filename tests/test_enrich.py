import numpy as np

from lidar_classification_qa.enrich import compute_ground_grid, height_above_ground, on_real_ground


def test_ground_grid_uses_low_percentile_not_minimum():
    # one cell, a real cluster of ground points around z=0, one noisy point way below
    rng = np.random.default_rng(0)
    cluster = rng.normal(loc=0.0, scale=0.05, size=20)
    z = np.concatenate([cluster, [-50.0]])
    x = np.full(z.shape, 0.5)
    y = np.full(z.shape, 0.5)
    classification = np.full(z.shape, 2)

    ground_z, has_local_ground, nx, ny = compute_ground_grid(
        x, y, z, classification, xmin=0, ymin=0, xmax=1, ymax=1, cell_size=1.0, percentile=10.0
    )

    assert nx == 1 and ny == 1
    assert has_local_ground[0]
    # the percentile should stay close to the real ground cluster, not get dragged to -50
    assert ground_z[0] > -1


def test_empty_cells_fill_from_nearest_neighbour_and_are_marked_unreliable():
    # two cells side by side, only the left one has a ground-classified point
    x = np.array([0.5])
    y = np.array([0.5])
    z = np.array([10.0])
    classification = np.array([2])

    ground_z, has_local_ground, nx, ny = compute_ground_grid(
        x, y, z, classification, xmin=0, ymin=0, xmax=2, ymax=1, cell_size=1.0, percentile=10.0
    )

    assert nx == 2 and ny == 1
    assert not np.isnan(ground_z).any()
    assert ground_z[1] == ground_z[0]
    # the left cell has a real measurement, the right one only has a borrowed one
    assert list(has_local_ground) == [True, False]


def test_non_ground_points_are_ignored_when_estimating_ground():
    # a cell with one real ground point at z=0 and a pile of "vegetation"
    # points at z=40 shouldn't have its ground estimate dragged up to 40,
    # this is the exact bug a naive "percentile of everything" approach hits
    x = np.full(21, 0.5)
    y = np.full(21, 0.5)
    z = np.concatenate([[0.0], np.full(20, 40.0)])
    classification = np.concatenate([[2], np.full(20, 4)])

    ground_z, has_local_ground, nx, ny = compute_ground_grid(
        x, y, z, classification, xmin=0, ymin=0, xmax=1, ymax=1, cell_size=1.0, percentile=10.0
    )

    assert has_local_ground[0]
    assert ground_z[0] < 1.0


def test_height_above_ground_never_negative():
    x = np.array([0.5, 0.5])
    y = np.array([0.5, 0.5])
    z = np.array([5.0, -1.0])
    ground_z = np.array([0.0])

    hag = height_above_ground(x, y, z, ground_z, xmin=0, ymin=0, cell_size=1.0, nx=1, ny=1)

    assert hag[0] == 5.0
    assert hag[1] == 0.0


def test_on_real_ground_looks_up_per_point():
    has_local_ground = np.array([True, False])
    x = np.array([0.5, 1.5])
    y = np.array([0.5, 0.5])

    result = on_real_ground(x, y, has_local_ground, xmin=0, ymin=0, cell_size=1.0, nx=2, ny=1)

    assert list(result) == [True, False]
