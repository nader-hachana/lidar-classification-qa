import numpy as np

from lidar_classification_qa.detect import aggregate_columns, flag_likely_structure

# Four columns, 2m apart on x so compute_cell_id keeps them in separate cells.
# Ground is flat (z == height above ground) to keep the synthetic case simple.


def _scene():
    # column 0, x~1: a real tree, class 4, real vertical spread from returns
    # scattering off trunk, branches, and an uneven canopy top
    tree_x = np.full(6, 1.0)
    tree_y = np.full(6, 1.0)
    tree_hag = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0])
    tree_class = np.full(6, 4)

    # column 1, x~3: the real bug this repo is built around, a 138m roof
    # canopy labeled class 4 ("medium vegetation"), flat because it's a roof
    roof_x = np.full(5, 3.0)
    roof_y = np.full(5, 1.0)
    roof_hag = np.array([137.9, 138.0, 138.1, 137.95, 138.05])
    roof_class = np.full(5, 4)

    # column 2, x~5: a real building, correctly labeled class 6
    building_x = np.full(5, 5.0)
    building_y = np.full(5, 1.0)
    building_hag = np.array([19.9, 20.0, 20.1, 19.95, 20.05])
    building_class = np.full(5, 6)

    # column 3, x~7: real short grass, class 3, flat but nowhere near tall
    grass_x = np.full(5, 7.0)
    grass_y = np.full(5, 1.0)
    grass_hag = np.array([0.2, 0.25, 0.3, 0.28, 0.22])
    grass_class = np.full(5, 3)

    x = np.concatenate([tree_x, roof_x, building_x, grass_x])
    y = np.concatenate([tree_y, roof_y, building_y, grass_y])
    hag = np.concatenate([tree_hag, roof_hag, building_hag, grass_hag])
    z = hag  # ground is flat at z=0 in this synthetic scene
    classification = np.concatenate([tree_class, roof_class, building_class, grass_class])

    return x, y, z, hag, classification


def test_flags_only_the_tall_flat_mislabeled_column():
    x, y, z, hag, classification = _scene()

    columns = aggregate_columns(x, y, z, hag, classification, xmin=0.0, ymin=0.0, cell_size=2.0, nx=4, ny=1)
    flagged = flag_likely_structure(columns)

    flagged_x = columns["x_center"][flagged]
    assert len(flagged_x) == 1
    assert 2.0 <= flagged_x[0] < 4.0  # the roof column, not the tree, building, or grass


def test_real_tree_is_not_flagged_despite_being_tall():
    x, y, z, hag, classification = _scene()
    columns = aggregate_columns(x, y, z, hag, classification, xmin=0.0, ymin=0.0, cell_size=2.0, nx=4, ny=1)
    flagged = flag_likely_structure(columns)

    tree_column = (columns["x_center"] >= 0.0) & (columns["x_center"] < 2.0)
    assert not flagged[tree_column].any()


def test_correctly_labeled_building_is_not_flagged():
    x, y, z, hag, classification = _scene()
    columns = aggregate_columns(x, y, z, hag, classification, xmin=0.0, ymin=0.0, cell_size=2.0, nx=4, ny=1)
    flagged = flag_likely_structure(columns)

    building_column = (columns["x_center"] >= 4.0) & (columns["x_center"] < 6.0)
    assert not flagged[building_column].any()


def test_short_flat_vegetation_is_not_flagged():
    x, y, z, hag, classification = _scene()
    columns = aggregate_columns(x, y, z, hag, classification, xmin=0.0, ymin=0.0, cell_size=2.0, nx=4, ny=1)
    flagged = flag_likely_structure(columns)

    grass_column = (columns["x_center"] >= 6.0) & (columns["x_center"] < 8.0)
    assert not flagged[grass_column].any()
