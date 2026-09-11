# lidar-classification-qa

Flags LiDAR points labeled as vegetation that are geometrically flat structures instead.

## Background

A public SoFi Stadium LiDAR file (364M points) has no building classification at all, its roof is labeled "medium vegetation." Real vegetation has vertical spread, returns scatter across trunk, branches, and canopy. A flat roof doesn't. That difference is measurable without trusting the file's classification field.

## How it works

For each 2m x 2m column of the point cloud:

1. Estimate ground height from the file's ground-classified points only.
2. Measure vertical spread and majority classification.
3. Flag columns labeled vegetation that are too tall and too flat to be real vegetation, backed by enough points, and sitting on ground measured locally rather than borrowed from a distant cell.

The ground-reliability check matters on hilly or forested terrain: cells with no ground returns nearby (dense canopy blocks the laser) can borrow an estimate that's tens of meters off, which fakes "tall flat" columns. Testing against a forested hillside dataset surfaced this and cutting it out dropped the false positive rate by 98.5%.

## Results

| Dataset | Points | Flagged | Notes |
|---|---|---|---|
| SoFi Stadium | 364M | 781 | Flat terrain, the original bug |
| Autzen Stadium | 10.7M | 2,990 | Forested hillside, candidates for review, not confirmed errors |
| Trestle Bridge | 46.7M | 0 | Clean negative, real vegetation present |
| Wolverine Glacier | 45.7M (cropped) | 0 | Clean negative, real vegetation present |
| Red Rocks Amphitheatre | 4.0M | 0 | Inconclusive, file has no classification at all |
| Mount St. Helens | 12.4M | 0 | Inconclusive, file has no vegetation classes |

Hobbs, NM (~24B points total) wasn't tested, a 1km² crop alone came to 658M points and the source has corrupted tiles.

## Usage

```bash
uv sync
uv run lidar-classification-qa path/to/file.laz \
  --cell-size 2.0 --ground-cell-size 10.0 \
  --min-height 10 --max-vertical-std 1.0 \
  --min-points 5 --min-real-ground-fraction 1.0 \
  --output flagged_columns.csv
```

Works on any LAS/LAZ/COPC file with a `classification` field. Memory stays bounded independent of file size, ground estimation uses a fixed-size sample and the full scan streams through in chunks.

## Development

```bash
uv sync
uv run pytest
```
