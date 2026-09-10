# lidar-classification-qa

Flags LiDAR points labeled as vegetation that geometrically behave like flat structures instead, an automated check for a real, concrete kind of classification error.

## The bug that started this

While building [copc-pointcloud-pipeline](https://github.com/nader-hachana/copc-pointcloud-pipeline), a streaming point cloud processing pipeline over a public SoFi Stadium LiDAR file (364,384,576 points), I found that only two classification codes exist anywhere in the file: class 2 (ground, 94%) and class 4 (medium vegetation, 6%). There's no building class at all, and the stadium's own roof is labeled "medium vegetation."

A roof and a tree canopy do not look alike to a laser. Real vegetation scatters returns across a trunk, branches, and an uneven top, so a column of true vegetation points has real vertical spread. A roof does not, every return off it sits close to the same height. That difference is measurable, and checking for it doesn't require trusting the file's own classification field at all.

## What this tool does

For every 2m x 2m column of the point cloud:

1. Computes height above ground independently, using an estimated ground surface built only from the file's ground-classified points, not from trusting the whole classification field.
2. Measures the column's vertical spread (a real tree canopy is rough, a roof is flat) and its majority classification.
3. Flags a column only when it's labeled vegetation, taller than real vegetation realistically reaches, flatter than real vegetation ever is, backed by enough points to trust, and sitting on ground that was actually measured nearby, not borrowed from somewhere else.

That last condition exists because of a real failure this project hit during testing, worth walking through since it's the kind of thing that would otherwise just quietly produce wrong results.

### Two bugs found by testing against a second real file

The SoFi result alone would have been easy to overstate. Testing against a second public dataset, Autzen Stadium (a forested Oregon hillside, a very different kind of terrain), immediately broke the first version of this tool: 19% of all occupied columns got flagged, including "vegetation" columns over 150 meters tall, taller than any tree that has ever existed.

Two real bugs were behind that:

1. **Points near the ground being folded into a column's own flatness measurement.** A column's real ground returns at z=0 plus a flat roof at z=40 look like one huge vertical spread, exactly the opposite of what a flat roof should register as. Fixed by excluding near-ground points from a column's own statistics before measuring anything.
2. **Ground estimated from whatever points exist in a cell, not specifically from real ground.** On a flat SoFi parking lot this never showed up. On a forested hillside, dense canopy blocks the laser from reaching true ground over large contiguous areas: 21,137 of roughly 149,000 ground cells at Autzen had zero ground-classified returns at all. Where that happens, the estimate has to borrow from the nearest cell that does have real ground data, and on hilly terrain "nearest" can sit tens of meters off, in the worst measured case, 130 meters. That fabricates "tall and flat" columns out of nothing.

The fix: ground is now estimated only from points the file itself classifies as ground, and every cell tracks whether its estimate is real or borrowed. A flagged column is only trusted if every contributing point's height above ground rests on a real, local measurement, not a borrowed one. A minimum point count (5, by default) was also added, a single stray return is flat and tall by definition, it has no spread to measure, and one of SoFi's own original top results turned out to be exactly that: a single-point column, correctly dropped by this fix rather than a real finding.

With that fix, Autzen's flagged count dropped from 196,911 to 2,990, a 98.5% reduction, while SoFi's real detection got *more* confident, not less (see below).

## Results on the real files

### SoFi Stadium (364,384,576 points, flat terrain)

```
100,037 occupied columns, 781 flagged as likely misclassified vegetation
```

- Median flagged column: 341 points (up from 214 before the fix, since removing noise-driven singletons raises the average quality of what's left)
- 622 of 781 flagged columns have 50+ points, 341 have 500+ points
- The tallest well-backed columns sit at 35-40m, perfectly flat (vertical spread as low as 0.008m), 100% labeled class 4, clustered tightly together, consistent with one contiguous flat structure, not scattered noise
- Mean classification purity across flagged columns: 91%

### Autzen Stadium (10,653,336 points, forested hillside, 209m of real elevation change)

```
1,144,016 occupied columns, 2,990 flagged as likely misclassified vegetation
```

A useful negative control: Autzen is a reference "classified" dataset, not one already known to have a labeling bug like SoFi. The flagged columns here are presented as candidates for review, not confirmed errors, their vertical spread (typically 0.3-1.0m) is less pristine than SoFi's near-perfectly-flat roof (often under 0.05m), consistent with either real classification mistakes in a million-plus-column dataset, or residual imperfections in a 10m-resolution ground model on genuinely sloped terrain. The 98.5% drop from the pre-fix run shows the ground-reliability gate is doing real work, this remainder is the harder, more ambiguous tail.

### Trestle Bridge (46,748,348 points, small drone survey)

```
2,621 occupied columns, 0 flagged as likely misclassified vegetation
```

A clean negative, and not a trivial one: this file has real vegetation in it (class 3, 2.0% of all points), so there was something to get wrong, and nothing was flagged. One data point toward the tool not just being trigger-happy on data it hasn't seen before.

## Try it

```bash
uv sync
uv run lidar-classification-qa path/to/file.laz \
  --cell-size 2.0 --ground-cell-size 10.0 \
  --min-height 10 --max-vertical-std 1.0 \
  --min-points 5 --min-real-ground-fraction 1.0 \
  --output flagged_columns.csv
```

Works on any LAS/LAZ/COPC file with a `classification` field. Memory stays bounded independent of file size: ground estimation uses a fixed-size probabilistic sample rather than every point, and the full scan folds the file through in fixed-size chunks into running per-column accumulators, never holding the whole point cloud in memory at once.

## Why this matters beyond one file

Nobody manually audits every point in a LiDAR delivery. A wrong classification like SoFi's would ship silently into anything built on top of it, canopy height models, vegetation encroachment monitoring near infrastructure, digital twins, all trusting a field that's wrong for a meaningful chunk of the tallest structure in the file. This isn't about vegetation specifically either, the same geometric reasoning (tall, flat, labeled as something that shouldn't be tall and flat, and backed by real local ground data) generalizes to catching other classification errors before they reach whatever's built on top of them.

## Development

```bash
uv sync
uv run pytest
```

Every detection function works on plain numpy arrays in and out, so the core logic is unit tested against small made-up point clouds (a real tree, a mislabeled roof, a correctly labeled building, real short grass, a column sitting on borrowed ground, a handful of stray points) rather than the full file.
