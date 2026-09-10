# lidar-classification-qa

Flags LiDAR points labeled as vegetation that geometrically behave like flat structures instead, an automated check for a real, concrete kind of classification error.

## The bug that started this

While building [copc-pointcloud-pipeline](https://github.com/nader-hachana/copc-pointcloud-pipeline), a streaming point cloud processing pipeline over a public SoFi Stadium LiDAR file (364,384,576 points), I found that only two classification codes exist anywhere in the file: class 2 (ground, 94%) and class 4 (medium vegetation, 6%). There's no building class at all, and the stadium's own roof, roughly 90+ meters above the ground beneath it, is labeled "medium vegetation."

A roof and a tree canopy do not look alike to a laser. Real vegetation scatters returns across a trunk, branches, and an uneven top, so a column of true vegetation points has real vertical spread. A roof does not, every return off it sits close to the same height. That difference is measurable, and checking for it doesn't require trusting the file's own classification field at all.

## What this tool does

For every 2m x 2m column of the point cloud:

1. Computes height above ground independently, using an estimated ground surface, not the file's own classification.
2. Measures the column's vertical spread (a real tree canopy is rough, a roof is flat) and its majority classification.
3. Flags any column that is labeled vegetation, but is both taller than real vegetation realistically reaches, and flatter than real vegetation ever is.

Points near the ground are excluded from a column's own statistics before any of this, otherwise a column's real ground returns get blended into the same "how flat is this" measurement as whatever is sitting above them, which quietly erases the exact signal the check is looking for. That's a real bug I hit and fixed while building this, worth mentioning since it's the kind of mistake this tool is meant to catch, just one level up: garbage classification in, garbage analysis out, unless something checks first.

Ground itself is estimated on a coarser grid than the one used for column analysis, for a related reason: a 2m cell sitting entirely under a solid roof has few or no real ground returns of its own, a coarse cell reaches past a structure's footprint to real ground nearby instead.

## Results on the real file

Run against the full 364M-point SoFi Stadium file:

```
100,973 occupied columns, 913 flagged as likely misclassified vegetation
```

- 913 columns labeled vegetation that are too tall and too flat to actually be vegetation
- 752 of those are backed by 10+ points, not single stray returns, median flagged column has 214 points
- The top hit: a column 92.4 meters above the estimated ground, perfectly flat (0.000m vertical spread), 100% labeled class 4, sitting exactly where the stadium roof is
- Mean height across all flagged columns: 13.9m, well past what vegetation in the surrounding area reaches, mean classification purity: 92%

95 of the 913 flagged columns are backed by a single point each, real candidates worth a human's second look, but not strong evidence on their own since a lone point can't establish flatness. A stricter run (`--min-purity` higher, or filtering the output CSV by `n_points`) narrows to the high-confidence set.

## Try it

```bash
uv sync
uv run lidar-classification-qa path/to/file.laz \
  --cell-size 2.0 --ground-cell-size 10.0 \
  --min-height 10 --max-vertical-std 1.0 \
  --output flagged_columns.csv
```

Works on any LAS/LAZ/COPC file with a `classification` field. Memory stays bounded independent of file size: ground estimation uses a fixed-size probabilistic sample rather than every point, and the full scan folds the file through in fixed-size chunks into running per-column accumulators, never holding the whole point cloud in memory at once.

## Why this matters beyond one file

Nobody manually audits every point in a LiDAR delivery. A wrong classification like this one would ship silently into anything built on top of it, canopy height models, vegetation encroachment monitoring near infrastructure, digital twins, all trusting a field that, in this file, is wrong for a meaningful chunk of the tallest structure in it. This isn't about vegetation specifically either, the same geometric reasoning (tall, flat, and labeled as something that shouldn't be tall and flat) generalizes to catching other classification errors before they reach whatever's built on top of them.

## Development

```bash
uv sync
uv run pytest
```

Every detection function works on plain numpy arrays in and out, so the core logic is unit tested against small made-up point clouds (a real tree, a mislabeled roof, a correctly labeled building, real short grass) rather than the full file.
